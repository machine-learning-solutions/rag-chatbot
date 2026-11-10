#!/usr/bin/env bash
# systemd ExecStart helper — assestant Node BFF on :3001
set -euo pipefail
ROOT="/home/jadaboawwad/Files/Software/Repositories/Applications/Portfolio/chatbot/assestant"
cd "$ROOT"

# Node is installed via nvm; login PATH is not available under systemd.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
fi

export PORT="${PORT:-3001}"
exec node server.js
