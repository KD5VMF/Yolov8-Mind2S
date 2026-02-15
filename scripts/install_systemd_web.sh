#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/envYoloMind2S}"
SERVICE_NAME="yolov8-mind2s-web"

echo "Installing systemd service: ${SERVICE_NAME}.service"
sudo mkdir -p /etc/systemd/system
sudo cp "$REPO_DIR/systemd/yolov8-web.service" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo sed -i "s|__REPO_DIR__|$REPO_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo sed -i "s|__ENV_PY__|$ENV_DIR/bin/python|g" "/etc/systemd/system/${SERVICE_NAME}.service"

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}.service"

echo
echo "Status:"
systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
