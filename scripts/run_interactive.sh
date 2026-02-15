#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/envYoloMind2S}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ENV_DIR/bin/activate"
python "$REPO_DIR/yolo_interactive.py"
