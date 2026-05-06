"""Plotting and visual helpers for final_solution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def denormalize(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def preview_transform(config: dict):
    return transforms.Compose(
        [
            transforms.Resize((config["imgsz"], config["imgsz"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def save_batch_image(images: torch.Tensor, labels: torch.Tensor, classes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(8, images.shape[0])
    tile_w, tile_h = 190, 230
    image_h, label_h = 170, 52
    cols, rows = 4, 2
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(14)
    for i in range(count):
        x, y = (i % cols) * tile_w, (i // cols) * tile_h
        thumb = denormalize(images[i]).resize((image_h, image_h))
        canvas.paste(thumb, (x + 10, y + 8))
        draw.rectangle(
            (x + 8, y + 6, x + 10 + image_h, y + 8 + image_h),
            outline="#222222",
            width=1,
        )
        draw.rectangle(
            (x + 8, y + 184, x + tile_w - 8, y + 184 + label_h),
            fill="#F4F6F8",
            outline="#D0D7DE",
        )
        draw_centered_lines(
            draw,
            wrapped_label(classes[int(labels[i])], max_chars=18),
            (x + 12, y + 188, x + tile_w - 12, y + 184 + label_h),
            label_font,
            "#111111",
        )
    canvas.save(path)


def save_two_batches(loader, classes: list[str], out_dir: Path, prefix: str) -> None:
    for idx, (images, labels) in enumerate(loader):
        if idx >= 2:
            break
        save_batch_image(images, labels, classes, out_dir / f"{prefix}_batch{idx}.jpg")


def font(size: int, bold: bool = False):
    names = ["Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold else ["Arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw, text: str, used_font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=used_font)
    return box[2] - box[0], box[3] - box[1]


def wrapped_label(text: str, max_chars: int = 16) -> list[str]:
    parts = text.replace("_", " ").split()
    lines, current = [], ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return lines or [text]


def draw_centered_lines(draw, lines: list[str], box: tuple[int, int, int, int], used_font, fill: str) -> None:
    x0, y0, x1, y1 = box
    gap = 4
    heights = [text_size(draw, line, used_font)[1] for line in lines]
    total_h = sum(heights) + gap * max(0, len(lines) - 1)
    y = y0 + ((y1 - y0) - total_h) / 2
    for line, height in zip(lines, heights):
        width, _ = text_size(draw, line, used_font)
        draw.text((x0 + ((x1 - x0) - width) / 2, y), line, fill=fill, font=used_font)
        y += height + gap


def blend(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(int(a + (b - a) * ratio) for a, b in zip(start, end))


def plot_overall(summary: dict, path: Path) -> None:
    metrics = [
        ("Pred Acc", summary["overall_accuracy_from_predictions"], "#009E73"),
        ("Pred Top-5", summary["overall_top5_accuracy_from_predictions"], "#CC79A7"),
    ]
    image = Image.new("RGB", (1120, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, note_font = font(30, True), font(18), font(15)
    title = "Validation Overall Metrics"
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((1120 - title_w) / 2, 28), title, fill="black", font=title_font)
    left, top, width, height = 110, 110, 900, 340
    draw.line((left, top + height, left + width, top + height), fill="black", width=2)
    draw.line((left, top, left, top + height), fill="black", width=2)
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + height - int(value * height)
        draw.line((left - 6, y, left + width, y), fill="#E0E0E0", width=1)
        draw.text((left - 52, y - 9), f"{value:.1f}", fill="#333333", font=note_font)
    for i, (label, value, color) in enumerate(metrics):
        x0 = left + 80 + i * 215
        y0 = top + height - int(value * height)
        draw.rectangle((x0, y0, x0 + 135, top + height), fill=color, outline="black", width=2)
        value_text = f"{value:.3f}"
        tw, _ = text_size(draw, value_text, label_font)
        draw.text((x0 + (135 - tw) / 2, y0 - 30), value_text, fill="black", font=label_font)
        draw_centered_lines(
            draw,
            wrapped_label(label, 9),
            (x0 - 20, top + height + 14, x0 + 155, top + height + 66),
            label_font,
            "black",
        )
    image.save(path)


def plot_per_class(per_class: dict, path: Path) -> None:
    labels = list(per_class)
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    image = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = font(30, True), font(14), font(13)
    title = "Validation Per-Class Metrics"
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((1500 - title_w) / 2, 24), title, fill="#111111", font=title_font)
    left, top, width, height = 90, 110, 1320, 440
    draw.line((left, top + height, left + width, top + height), fill="#111111", width=2)
    draw.line((left, top, left, top + height), fill="#111111", width=2)
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + height - int(value * height)
        draw.line((left - 6, y, left + width, y), fill="#E0E0E0", width=1)
        draw.text((left - 48, y - 8), f"{value:.1f}", fill="#333333", font=small_font)
    group_w = width / max(1, len(labels))
    bar_w, bar_gap = min(22, max(12, int(group_w / 6))), 9
    for group_i, label in enumerate(labels):
        group_left = left + group_i * group_w
        base_x = group_left + group_w / 2 - (len(metrics) * bar_w + bar_gap * (len(metrics) - 1)) / 2
        for metric_i, (metric, color) in enumerate(zip(metrics, colors)):
            value = per_class[label][metric]
            x0 = int(base_x + metric_i * (bar_w + bar_gap))
            y0 = top + height - int(value * height)
            draw.rectangle((x0, y0, x0 + bar_w, top + height), fill=color, outline="black")
            draw.text((x0 - 4, y0 - 18), f"{value:.2f}", fill="#111111", font=small_font)
        draw_centered_lines(
            draw,
            wrapped_label(label, 16),
            (int(group_left), top + height + 18, int(group_left + group_w), top + height + 92),
            label_font,
            "#111111",
        )
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        x = 420 + i * 170
        draw.rectangle((x, 700, x + 26, 720), fill=color, outline="black")
        draw.text((x + 34, 699), metric, fill="#111111", font=small_font)
    image.save(path)


def plot_confusion(summary: dict, path: Path, normalized: bool = False) -> None:
    raw_matrix = summary["confusion_matrix"]
    matrix = [[divide(value, sum(row)) for value in row] for row in raw_matrix] if normalized else raw_matrix
    rows = summary.get("evaluation_classes", summary["true_classes"])
    cols = summary["predicted_classes"]
    cell_w, cell_h = 112, 76
    left, top = 320, 210
    width, height = left + len(cols) * cell_w + 70, top + len(rows) * cell_h + 70
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, label_font, number_font = font(30, True), font(18, True), font(14), font(18, True)
    title = "Validation Confusion Matrix (Normalized)" if normalized else "Validation Confusion Matrix"
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((width - title_w) / 2, 24), title, fill="#111111", font=title_font)
    pred = "Predicted class"
    pred_w, _ = text_size(draw, pred, axis_font)
    draw.text((left + (len(cols) * cell_w - pred_w) / 2, 78), pred, fill="#111111", font=axis_font)
    draw.text((24, top + len(rows) * cell_h / 2 - 14), "Actual class", fill="#111111", font=axis_font)
    max_value = max([max(row) for row in matrix] or [0])
    for c, label in enumerate(cols):
        x0 = left + c * cell_w
        draw_centered_lines(
            draw,
            wrapped_label(label, 13),
            (x0 + 4, 112, x0 + cell_w - 4, top - 18),
            label_font,
            "#111111",
        )
    for r, label in enumerate(rows):
        y0 = top + r * cell_h
        draw_centered_lines(
            draw,
            wrapped_label(label, 22),
            (16, y0, left - 22, y0 + cell_h),
            label_font,
            "#111111",
        )
        for c, value in enumerate(matrix[r]):
            ratio = divide(value, max_value)
            fill = blend((247, 251, 255), (8, 81, 156), ratio)
            x0 = left + c * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=fill, outline="#2F3A45")
            text = f"{value:.2f}" if normalized else str(value)
            tw, th = text_size(draw, text, number_font)
            draw.text(
                (x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2),
                text,
                fill="#FFFFFF" if ratio > 0.45 else "#111111",
                font=number_font,
            )
    draw.rectangle((left, top, left + len(cols) * cell_w, top + len(rows) * cell_h), outline="#111111", width=2)
    image.save(path)


def save_eval_outputs(out_dir: Path, summary: dict, predictions: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"class": name, **values} for name, values in summary["per_class"].items()]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "predictions_val.csv", predictions)
    write_csv(out_dir / "per_class_metrics_val.csv", rows)
    plot_overall(summary, out_dir / "overall_metrics_val.png")
    plot_per_class(summary["per_class"], out_dir / "per_class_metrics_val.png")
    plot_confusion(summary, out_dir / "confusion_matrix_val.png")
    plot_confusion(summary, out_dir / "confusion_matrix_val_normalized.png", normalized=True)


def plot_line_panel(
    draw: ImageDraw.ImageDraw,
    points: list[float],
    box: tuple[int, int, int, int],
    title: str,
    color: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x0, y0, x1, y1 = box
    title_font, small_font = font(18, True), font(12)
    draw.rectangle(box, outline="#222222", width=2)
    draw.text((x0 + 12, y0 + 10), title, fill="#111111", font=title_font)
    if not points:
        return
    y_min = min(points) if y_min is None else y_min
    y_max = max(points) if y_max is None else y_max
    if y_max == y_min:
        y_max = y_min + 1
    plot_left, plot_top, plot_right, plot_bottom = x0 + 58, y0 + 48, x1 - 22, y1 - 42
    for tick in range(5):
        ratio = tick / 4
        y = plot_bottom - int(ratio * (plot_bottom - plot_top))
        value = y_min + ratio * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill="#E5E5E5")
        draw.text((x0 + 10, y - 7), f"{value:.2f}", fill="#333333", font=small_font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222222", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222222", width=2)
    coords = []
    for i, value in enumerate(points):
        x = plot_left + int(i * (plot_right - plot_left) / max(1, len(points) - 1))
        y = plot_bottom - int((value - y_min) * (plot_bottom - plot_top) / (y_max - y_min))
        coords.append((x, y))
    if len(coords) > 1:
        draw.line(coords, fill=color, width=3)
    for x, y in coords:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="#111111")
    draw.text((plot_left, plot_bottom + 12), "epoch 1", fill="#333333", font=small_font)
    draw.text((plot_right - 55, plot_bottom + 12), f"epoch {len(points)}", fill="#333333", font=small_font)


def plot_training_results(results: list[dict], path: Path) -> None:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(30, True)
    title = "Training Results"
    title_w, _ = text_size(draw, title, title_font)
    draw.text(((1500 - title_w) / 2, 24), title, fill="#111111", font=title_font)
    train_loss = [float(row["train/loss"]) for row in results]
    val_loss = [float(row["val/loss"]) for row in results]
    top1 = [float(row["metrics/accuracy_top1"]) for row in results]
    top5 = [float(row["metrics/accuracy_top5"]) for row in results]
    lr = [float(row["lr"]) for row in results]
    plot_line_panel(draw, train_loss, (60, 90, 720, 330), "Train Loss", "#0072B2")
    plot_line_panel(draw, val_loss, (780, 90, 1440, 330), "Validation Loss", "#D55E00")
    plot_line_panel(draw, top1, (60, 380, 720, 620), "Validation Top-1 Accuracy", "#009E73", 0.0, 1.0)
    plot_line_panel(draw, top5, (780, 380, 1440, 620), "Validation Top-5 Accuracy", "#CC79A7", 0.0, 1.0)
    plot_line_panel(draw, lr, (420, 670, 1080, 860), "Learning Rate", "#6A3D9A")
    image.save(path)

