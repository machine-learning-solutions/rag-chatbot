#!/usr/bin/env bash
# systemd ExecStart helper — avatar_service on :8010
set -euo pipefail
ROOT="/home/jadaboawwad/Files/Software/Repositories/Applications/Portfolio/chatbot/assestant/avatar_service"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8010
