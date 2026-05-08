"""Train and evaluate a selected grocery product classification model with W&B logging."""

from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import *

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DATA = ROOT.parent / "data"
CONFIG = ROOT / "train_config.json"
PRETRAINED_MODELS = ROOT.parent / "pretrained_models"
UNKNOWN_CLASS = "unknown"
WANDB_PROJECT = "Product-Classification"
WANDB_RUN = None

TORCH_MODELS = {
    "swin-tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "convnext-tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv4s": "mobilenetv4_conv_small.e2400_r224_in1k",
    "mobilenetv4m": "mobilenetv4_conv_medium.e500_r256_in1k",
    "mobilenetv4hm": "mobilenetv4_hybrid_medium.e500_r224_in1k",
}
YOLO_MODELS = {
    "yolo11n": "yolo11n-cls.pt",
    "yolo11s": "yolo11s-cls.pt",
    "yolo11m": "yolo11m-cls.pt",
    "yolo26n": "yolo26n-cls.pt",
    "yolo26s": "yolo26s-cls.pt",
    "yolo26m": "yolo26m-cls.pt",
    "yolov8n": "yolov8n-cls.pt",
    "yolov8s": "yolov8s-cls.pt",
    "yolov8m": "yolov8m-cls.pt",
}


def wandb_threshold_tag(value) -> str:
    digits = f"{float(value):.2f}".split(".", 1)[1].rstrip("0") or "0"
    return f"thr{digits.zfill(2)}"


def wandb_run_name(config: dict) -> str:
    model = safe_name(model_key(config))
    batch = int(config.get("batch", 0))
    threshold = wandb_threshold_tag(config.get("unknown_threshold", 0.5))
    return f"{model}_B{batch}_{threshold}"


def wandb_scalar(value):
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def wandb_start(config: dict) -> None:
    global WANDB_RUN
    import wandb

    WANDB_RUN = wandb.init(project=config.get("wandb_project", WANDB_PROJECT), name=wandb_run_name(config), config=config)
    WANDB_RUN.define_metric("epoch")
    WANDB_RUN.define_metric("*", step_metric="epoch")


def wandb_log_row(row: dict, previous_time: float | None = None) -> float | None:
    if WANDB_RUN is None or "epoch" not in row:
        return previous_time

    epoch = int(float(row["epoch"]))
    log_row = {"epoch": epoch}
    if row.get("time") not in (None, ""):
        current_time = float(row["time"])
        log_row["time-per-epoch"] = current_time if previous_time is None else max(0.0, current_time - previous_time)
        previous_time = current_time

    for source, target in (
        ("train/loss", "loss/train-loss"),
        ("val/loss", "loss/val-loss"),
        ("metrics/accuracy_top1", "metric/accuracy-top1"),
        ("metrics/accuracy_top5", "metric/accuracy-top5"),
        ("lr", "lr"),
    ):
        if row.get(source) not in (None, ""):
            log_row[target] = wandb_scalar(row[source])
    if "lr" not in log_row and row.get("lr/pg0") not in (None, ""):
        log_row["lr"] = wandb_scalar(row["lr/pg0"])

    WANDB_RUN.log(log_row, step=epoch)
    return previous_time


def wandb_log_results_csv(path: Path) -> None:
    if WANDB_RUN is None or not path.exists():
        return
    previous_time = None
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            previous_time = wandb_log_row(row, previous_time)
    WANDB_RUN.save(str(path))


def wandb_log_summary(summary: dict) -> None:
    if WANDB_RUN is None:
        return
    WANDB_RUN.summary.update(
        {
            key: value
            for key, value in summary.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    )


def wandb_finish() -> None:
    if WANDB_RUN is not None:
        WANDB_RUN.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--run-folder", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--threshold", "--unknown-threshold", dest="unknown_threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))["train"]
    return dict(config)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    for key in ("epochs", "imgsz", "batch", "unknown_threshold"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.model_name:
        config["model_name"] = args.model_name
    if args.run_folder:
        config["run_folder"] = str(args.run_folder.expanduser().resolve())
    if args.device:
        config["device"] = args.device
    config["wandb"] = args.wandb
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project
    return config


def device_from(config: dict) -> torch.device:
    requested = config.get("device")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classes_from(folder: Path) -> list[str]:
    classes = sorted(path.name for path in folder.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found in: {folder}")
    return classes


def resolve_val_data(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if (path / "val").exists():
        return path, path / "val"
    classes_from(path)
    return path, path


def train_transform(config: dict):
    ops = [
        transforms.Resize((config["imgsz"], config["imgsz"])),
    ]
    if config.get("randaugment", True):
        ops.append(transforms.RandAugment())
    if config.get("horizontal_flip", 0.0) > 0:
        ops.append(transforms.RandomHorizontalFlip(config["horizontal_flip"]))
    if config.get("color_jitter", 0.0) > 0:
        jitter = config["color_jitter"]
        ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    if config.get("random_erasing", 0.0) > 0:
        ops.append(transforms.RandomErasing(p=config["random_erasing"]))
    return transforms.Compose(ops)


def eval_transform(config: dict):
    return transforms.Compose(
        [
            transforms.Resize((config["imgsz"], config["imgsz"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def model_key(config: dict) -> str:
    return str(config.get("model_name", "swin-tiny"))


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def is_yolo_model(name: str) -> bool:
    return name in YOLO_MODELS or name.endswith(".pt") or name.startswith("yolo")


def run_dir_for(config: dict) -> Path:
    return RUNS / safe_name(model_key(config))


def requested_run_dir(config: dict) -> Path:
    return Path(config["run_folder"]) if config.get("run_folder") else run_dir_for(config)


def build_model(config: dict, num_classes: int):
    key = model_key(config)
    model_name = TORCH_MODELS.get(key, key)
    pretrained = bool(config.get("pretrained", True))
    dropout = float(config.get("dropout", 0.0))

    try:
        import timm

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        return model, "timm"
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"timm model unavailable ({exc}); falling back to torchvision swin_t.")

    from torchvision.models import Swin_T_Weights, swin_t

    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = swin_t(weights=weights, dropout=dropout)
    except Exception as exc:
        print(f"Pretrained TorchVision weights unavailable ({exc}); using random initialization.")
        model = swin_t(weights=None, dropout=dropout)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model, "torchvision"


def topk_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    k = min(5, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    top1 = (top[:, 0] == targets).sum().item()
    top5 = (top == targets.unsqueeze(1)).any(dim=1).sum().item()
    return top1, top5


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(train)
    total_loss = total = top1 = top5 = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bsz = targets.shape[0]
        c1, c5 = topk_counts(logits.detach(), targets)
        total_loss += loss.item() * bsz
        total += bsz
        top1 += c1
        top5 += c5
    return {
        "loss": total_loss / max(1, total),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
    }


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    classes: list[str],
    config: dict,
    model_source: str,
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "classes": classes,
            "config": config,
            "model_source": model_source,
            "metric_top1": float(metric),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    config = checkpoint["config"]
    model, model_source = build_model(config, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes, config, model_source, checkpoint


def unknown_threshold(config: dict) -> float:
    return float(config.get("unknown_threshold", 0.5))


def open_set_correct(true_name: str, pred_name: str, train_classes: list[str]) -> bool:
    target_name = open_set_target_name(true_name, train_classes)
    return pred_name == target_name


def open_set_target_name(true_name: str, train_classes: list[str]) -> str:
    return true_name if true_name in train_classes else UNKNOWN_CLASS


def open_set_details(rows: list[dict], train_classes: list[str], true_classes: list[str], threshold: float) -> dict:
    shared_classes = sorted(set(train_classes) & set(true_classes))
    train_only_classes = sorted(set(train_classes) - set(true_classes))
    validation_only_classes = sorted(set(true_classes) - set(train_classes))
    eval_classes = shared_classes + [UNKNOWN_CLASS]
    class_to_i = {name: i for i, name in enumerate(eval_classes)}
    matrix = [[0 for _ in eval_classes] for _ in eval_classes]

    for row in rows:
        target_name = open_set_target_name(row["true_class"], train_classes)
        pred_name = row["predicted_class"] if row["predicted_class"] in class_to_i else UNKNOWN_CLASS
        matrix[class_to_i[target_name]][class_to_i[pred_name]] += 1

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    top5_correct = sum(
        row["true_class"] in row["top5"].split("|")
        if row["true_class"] in train_classes
        else row["predicted_class"] == UNKNOWN_CLASS
        for row in rows
    )
    per_class = {}
    for class_name in eval_classes:
        row_i = class_to_i[class_name]
        col_i = class_to_i[class_name]
        tp = matrix[row_i][col_i]
        support = sum(matrix[row_i])
        predicted = sum(row[col_i] for row in matrix)
        precision = divide(tp, predicted)
        recall = divide(tp, support)
        per_class[class_name] = {
            "support": support,
            "correct": tp,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall),
        }
    return {
        "true_classes": true_classes,
        "train_classes": train_classes,
        "shared_classes": shared_classes,
        "train_only_classes": train_only_classes,
        "validation_only_classes": validation_only_classes,
        "unknown_class": UNKNOWN_CLASS,
        "unknown_threshold": threshold,
        "evaluation_classes": eval_classes,
        "predicted_classes": eval_classes,
        "confusion_matrix": matrix,
        "total_val_images": total,
        "overall_accuracy_from_predictions": divide(correct, total),
        "overall_top5_accuracy_from_predictions": divide(top5_correct, total),
        "per_class": per_class,
    }


def predict_val(
    model,
    classes: list[str],
    val_dir: Path,
    config: dict,
    device: torch.device,
) -> tuple[list[dict], dict]:
    model.eval()
    tfm = eval_transform(config)
    true_classes = classes_from(val_dir)
    threshold = 0.0 if set(classes) == set(true_classes) else unknown_threshold(config)
    rows = []

    for true_name in true_classes:
        for image_path in sorted((val_dir / true_name).glob("*.jpg")):
            image = Image.open(image_path).convert("RGB")
            tensor = tfm(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            order = probs.argsort(descending=True)
            raw_pred_name = classes[int(order[0])]
            confidence = float(probs[int(order[0])])
            pred_name = raw_pred_name if confidence >= threshold else UNKNOWN_CLASS
            top5 = [classes[int(i)] for i in order[: min(5, len(classes))]]
            rows.append(
                {
                    "image": str(image_path),
                    "true_class": true_name,
                    "target_class": open_set_target_name(true_name, classes),
                    "predicted_class": pred_name,
                    "raw_predicted_class": raw_pred_name,
                    "confidence": confidence,
                    "unknown_threshold": threshold,
                    "top5": "|".join(top5),
                    "correct": open_set_correct(true_name, pred_name, classes),
                }
            )

    return rows, open_set_details(rows, classes, true_classes, threshold)


def evaluate_open_set_metrics(
    model,
    classes: list[str],
    val_dir: Path,
    config: dict,
    device: torch.device,
    criterion,
) -> dict:
    model.eval()
    _, details = predict_val(model, classes, val_dir, config, device)
    class_to_i = {name: i for i, name in enumerate(classes)}
    tfm = eval_transform(config)
    loss_total = loss_count = 0

    for true_name in details["shared_classes"]:
        for image_path in sorted((val_dir / true_name).glob("*.jpg")):
            image = Image.open(image_path).convert("RGB")
            tensor = tfm(image).unsqueeze(0).to(device)
            target = torch.tensor([class_to_i[true_name]], device=device)
            with torch.no_grad():
                loss = criterion(model(tensor), target)
            loss_total += float(loss.item())
            loss_count += 1

    return {
        "loss": divide(loss_total, loss_count),
        "top1": details["overall_accuracy_from_predictions"],
        "top5": details["overall_top5_accuracy_from_predictions"],
    }


def validate(
    weights: Path,
    data_root: Path,
    val_dir: Path,
    config: dict,
    out_dir: Path,
    device: torch.device,
) -> dict:
    model, classes, _, model_source, checkpoint = load_checkpoint(weights, device)
    predictions, details = predict_val(model, classes, val_dir, config, device)
    summary = {
        "data": str(data_root),
        "val_dir": str(val_dir),
        "weights": str(weights),
        "model_source": model_source,
        "epoch_max": int(checkpoint.get("epoch", -1)),
        "top1_accuracy": details["overall_accuracy_from_predictions"],
        "top5_accuracy": details["overall_top5_accuracy_from_predictions"],
        **details,
    }
    save_eval_outputs(out_dir, summary, predictions)
    wandb_log_summary(summary)
    return summary


def train(data: Path, config: dict, device: torch.device) -> Path:
    train_dir = RUNS / "train"
    weights_dir = train_dir / "weights"
    train_ds = datasets.ImageFolder(data / "train", transform=train_transform(config))
    preview_train_ds = datasets.ImageFolder(data / "train", transform=preview_transform(config))
    preview_val_ds = datasets.ImageFolder(data / "val", transform=preview_transform(config))
    classes = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=config["batch"], shuffle=True, num_workers=config["num_workers"])
    preview_train_loader = DataLoader(preview_train_ds, batch_size=config["batch"], shuffle=True, num_workers=0)
    preview_val_loader = DataLoader(preview_val_ds, batch_size=config["batch"], shuffle=True, num_workers=0)
    save_two_batches(preview_train_loader, classes, train_dir, "train")
    save_two_batches(preview_val_loader, preview_val_ds.classes, train_dir, "val")

    model, model_source = build_model(config, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["min_lr"],
    )

    best_top1, best_epoch, stale = -math.inf, -1, 0
    results = []
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = evaluate_open_set_metrics(model, classes, data / "val", config, device, criterion)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "time": epoch_time,
            "train/loss": train_metrics["loss"],
            "val/loss": val_metrics["loss"],
            "metrics/accuracy_top1": val_metrics["top1"],
            "metrics/accuracy_top5": val_metrics["top5"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        results.append(row)
        write_csv(train_dir / "results.csv", results)
        wandb_log_row(row)
        plot_training_results(results, train_dir / "results.png")
        save_checkpoint(weights_dir / "last.pt", model, epoch, classes, config, model_source, val_metrics["top1"])
        if val_metrics["top1"] > best_top1:
            best_top1, best_epoch, stale = val_metrics["top1"], epoch, 0
            save_checkpoint(weights_dir / "best.pt", model, epoch, classes, config, model_source, val_metrics["top1"])
        else:
            stale += 1
        print(
            f"epoch={epoch} time={epoch_time:.2f}s train_loss={train_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f}"
        )
        if stale >= config["patience"]:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break
    return weights_dir / "best.pt"


def yolo_model_path(config: dict) -> Path:
    name = YOLO_MODELS.get(model_key(config), model_key(config))
    path = Path(name)
    if not path.is_absolute():
        path = PRETRAINED_MODELS / name
    if not path.exists():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    return path.resolve()


def name_of(names: dict[int, str] | list[str], index: int) -> str:
    return names[index] if isinstance(names, list) else names[int(index)]


def keep_two_batch_images(folder: Path, prefix: str) -> None:
    for image in sorted(folder.glob(f"{prefix}_batch*.*"))[2:]:
        image.unlink(missing_ok=True)


def checkpoint_epoch_summary(train_dir: Path) -> dict:
    results_path = train_dir / "results.csv"
    if not results_path.exists():
        return {"epoch_max": None, "epoch_last": None}
    with results_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"epoch_max": None, "epoch_last": None}

    def score(row: dict) -> tuple[float, float, float]:
        return (
            float(row.get("metrics/accuracy_top1") or 0),
            float(row.get("metrics/accuracy_top5") or 0),
            -float(row.get("val/loss") or 0),
        )

    best_row = max(rows, key=score)
    last_row = rows[-1]
    return {
        "epoch_max": int(float(best_row["epoch"])),
        "epoch_last": int(float(last_row["epoch"])),
        "epoch_max_metric": "metrics/accuracy_top1",
        "epoch_max_top1": float(best_row.get("metrics/accuracy_top1") or 0),
        "epoch_last_top1": float(last_row.get("metrics/accuracy_top1") or 0),
    }


def update_checkpoint_epoch(path: Path, epoch: int | None) -> None:
    if epoch is None or not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        checkpoint["epoch"] = int(epoch)
        torch.save(checkpoint, path)


def update_weight_epochs(train_dir: Path, epoch_info: dict) -> None:
    weights_dir = train_dir / "weights"
    update_checkpoint_epoch(weights_dir / "best.pt", epoch_info.get("epoch_max"))
    update_checkpoint_epoch(weights_dir / "last.pt", epoch_info.get("epoch_last"))


def yolo_train_kwargs(config: dict, data: Path) -> dict:
    keys = {
        "epochs",
        "patience",
        "imgsz",
        "batch",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "dropout",
        "auto_augment",
        "erasing",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "scale",
        "flipud",
        "fliplr",
        "degrees",
        "translate",
        "shear",
        "perspective",
        "bgr",
        "mixup",
        "cutmix",
        "device",
    }
    kwargs = {key: config[key] for key in keys if key in config}
    kwargs.update({
        "data": str(data),
        "project": str(RUNS),
        "name": "train",
        "exist_ok": True,
        "plots": True,
        "val": set(classes_from(data / "train")) == set(classes_from(data / "val")),
    })
    return kwargs


def yolo_evaluate_images(
    model,
    val_dir: Path,
    imgsz: int,
    device: str | None,
    config: dict,
) -> tuple[list[dict], dict]:
    true_classes = classes_from(val_dir)
    model_classes = [name_of(model.names, i) for i in range(len(model.names))]
    threshold = 0.0 if set(model_classes) == set(true_classes) else unknown_threshold(config)
    rows = []

    for true_name in true_classes:
        for image in sorted((val_dir / true_name).glob("*.jpg")):
            result = model.predict(str(image), imgsz=imgsz, device=device, verbose=False)[0]
            raw_pred_name = name_of(model.names, int(result.probs.top1))
            confidence = float(result.probs.top1conf)
            pred_name = raw_pred_name if confidence >= threshold else UNKNOWN_CLASS
            top5 = [name_of(model.names, int(i)) for i in result.probs.top5]
            rows.append({
                "image": str(image),
                "true_class": true_name,
                "target_class": open_set_target_name(true_name, model_classes),
                "predicted_class": pred_name,
                "raw_predicted_class": raw_pred_name,
                "confidence": confidence,
                "unknown_threshold": threshold,
                "top5": "|".join(top5),
                "correct": open_set_correct(true_name, pred_name, model_classes),
            })

    return rows, open_set_details(rows, model_classes, true_classes, threshold)


def yolo_validate(model, data_root: Path, val_dir: Path, config: dict, out_dir: Path) -> dict:
    predictions, details = yolo_evaluate_images(
        model,
        val_dir,
        int(config["imgsz"]),
        config.get("device"),
        config,
    )
    val_error = "Skipped Ultralytics closed-set val; evaluated the provided val folder with open-set unknown logic."

    summary = {
        "data": str(data_root),
        "val_dir": str(val_dir),
        "weights": str(model.ckpt_path) if getattr(model, "ckpt_path", None) else None,
        "ultralytics_val_dir": None,
        "ultralytics_val_error": val_error,
        "top1_accuracy": details["overall_accuracy_from_predictions"],
        "top5_accuracy": details["overall_top5_accuracy_from_predictions"],
        **details,
    }
    save_eval_outputs(out_dir, summary, predictions)
    wandb_log_summary(summary)
    return summary


def run_yolo(args: argparse.Namespace, config: dict) -> None:
    from ultralytics import YOLO

    global RUNS
    RUNS = requested_run_dir(config)
    RUNS.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    val_root, val_dir = resolve_val_data((args.val_data or data).resolve())
    eval_dir = RUNS / ("validation_only" if args.val_only else "evaluation")

    if args.val_only:
        model_path = args.weights.resolve() if args.weights else yolo_model_path(config)
        summary = yolo_validate(YOLO(str(model_path)), val_root, val_dir, config, eval_dir)
        print(json.dumps(summary, indent=2))
        return

    if not (data / "train").exists() or not (data / "val").exists():
        raise FileNotFoundError(f"Expected train/ and val/ folders in: {data}")
    train_result = YOLO(str(yolo_model_path(config))).train(**yolo_train_kwargs(config, data))
    train_dir = Path(train_result.save_dir)
    wandb_log_results_csv(train_dir / "results.csv")
    keep_two_batch_images(train_dir, "train")
    epoch_info = checkpoint_epoch_summary(train_dir)
    update_weight_epochs(train_dir, epoch_info)
    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = train_dir / "weights" / "last.pt"
    summary = yolo_validate(YOLO(str(best_weights)), data, data / "val", config, eval_dir)
    summary.update(
        {
            "start_model": str(yolo_model_path(config)),
            "best_weights": str(best_weights),
            "train_dir": str(train_dir),
            "train_config": config,
            **epoch_info,
        }
    )
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config.resolve()), args)
    if config.get("wandb", False) and not args.val_only:
        wandb_start(config)
    try:
        if is_yolo_model(model_key(config)):
            run_yolo(args, config)
            return

        global RUNS
        RUNS = requested_run_dir(config)
        device = device_from(config)
        RUNS.mkdir(parents=True, exist_ok=True)
        data = args.data.resolve()
        val_root, val_dir = resolve_val_data((args.val_data or data).resolve())

        if args.val_only:
            if not args.weights:
                raise FileNotFoundError("--val-only requires --weights")
            summary = validate(args.weights.resolve(), val_root, val_dir, config, RUNS / "validation_only", device)
            print(json.dumps(summary, indent=2))
            return

        if not (data / "train").exists() or not (data / "val").exists():
            raise FileNotFoundError(f"Expected train/ and val/ folders in: {data}")
        best_weights = train(data, config, device)
        summary = validate(best_weights, data, data / "val", config, RUNS / "evaluation", device)
        last_path = RUNS / "train" / "weights" / "last.pt"
        if last_path.exists():
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            summary["epoch_last"] = int(last.get("epoch", -1))
            (RUNS / "evaluation" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        wandb_finish()


if __name__ == "__main__":
    main()
