#!/usr/bin/env bash
# Legacy wrapper. Use the new Mind2S setup script.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$DIR/scripts/setup_mind2s_ubuntu.sh"
