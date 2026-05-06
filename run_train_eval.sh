#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BASE_DIR/.venv"
REQUIREMENTS="$BASE_DIR/requirements.txt"

MODE="${1:-}"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
fi

source "$VENV_DIR/bin/activate"

if [ "$MODE" = "train" ]; then
  python "$BASE_DIR/train_eval.py" --device mps

elif [ "$MODE" = "validation" ]; then
  python "$BASE_DIR/train_eval.py" \
    --val-only \
    --weights "$BASE_DIR/../runs/runs-final/resnet50/train/weights/best.pt" \
    --val-data "$BASE_DIR/../data/val" \
    --device mps

elif [ "$MODE" = "streamlit" ]; then
  streamlit run "$BASE_DIR/streamlit_app.py"

else
  echo "Usage: ./run_train_eval.sh {train|validation|streamlit}"
  exit 1
fi