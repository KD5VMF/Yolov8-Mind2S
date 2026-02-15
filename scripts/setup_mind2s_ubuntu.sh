#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envYoloMind2S}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/6] System deps (Ubuntu/Debian)..."
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential ffmpeg v4l-utils

echo "[2/6] Create venv: $ENV_DIR"
python3 -m venv "$ENV_DIR"

echo "[3/6] Upgrade pip tooling..."
"$ENV_DIR/bin/python" -m pip install -U pip wheel setuptools

echo "[4/6] Install core Python deps..."
"$ENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo "[5/6] Optional OpenVINO (Intel CPU/GPU/NPU) deps..."
echo "      If you want Intel GPU/NPU acceleration, run:"
echo "        $ENV_DIR/bin/pip install -r $REPO_DIR/requirements-openvino.txt"
echo
echo "      NOTE: Intel NPU also needs the Intel NPU driver (see README)."

echo "[6/6] Done."
echo
echo "Next:"
echo "  source "$ENV_DIR/bin/activate""
echo "  python "$REPO_DIR/yolo_interactive.py""
