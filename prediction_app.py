from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from train_eval import UNKNOWN_CLASS, eval_transform, unknown_threshold

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "trained_models"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

def build_checkpoint_model(state_dict: dict, num_classes: int, config: dict):
    from torch import nn

    if any(key.startswith("features.") for key in state_dict):
        from torchvision.models import swin_t

        model = swin_t(weights=None, dropout=float(config.get("dropout", 0.0)))
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource
def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state"]
    model = build_checkpoint_model(state_dict, len(classes), config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, config


def predict_image(model, classes: list[str], config: dict, image: Image.Image):
    tensor = eval_transform(config)(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    order = probs.argsort(descending=True)
    raw_class = classes[int(order[0])]
    confidence = float(probs[int(order[0])])
    threshold = unknown_threshold(config)
    predicted_class = raw_class if confidence >= threshold else UNKNOWN_CLASS
    top5 = [(classes[int(i)], float(probs[int(i)])) for i in order[: min(5, len(classes))]]
    return predicted_class, raw_class, confidence, threshold, top5

def main() -> None:
    st.set_page_config(page_title="Product Classifier", layout="centered")
    st.title("Product Classifier")

    model_files = sorted(MODEL_DIR.glob("*.pt"))
    if not model_files:
        st.error(f"No model checkpoints found in {MODEL_DIR}")
        return

    model_path = st.selectbox("Model", model_files, format_func=lambda path: path.name)
    st.selectbox("Model type", ["resnet50"])
    uploaded_image = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])

    if uploaded_image is None:
        return

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    model, classes, config = load_model(str(model_path))
    predicted_class, _, confidence, threshold, top5 = predict_image(model, classes, config, image)

    st.metric("Predicted class:", predicted_class)
    st.write(f"Confidence: `{confidence:.4f}`")
    st.write(f"Unknown threshold from checkpoint: `{threshold:.4f}`")

    st.subheader("Top 5 predictions")
    for class_name, score in top5:
        st.write(f"{class_name}: `{score:.4f}`")

if __name__ == "__main__":
    main()
