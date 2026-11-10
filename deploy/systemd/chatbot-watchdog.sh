#!/usr/bin/env bash
# Periodic health check — fixes dead ngrok tunnels that systemd still marks active.
set -euo pipefail

RECOVER="/home/jadaboawwad/Files/Software/Repositories/Applications/Portfolio/chatbot/deploy/systemd/chatbot-recover.sh"

backend_ok() {
  curl -sf -o /dev/null --max-time 4 http://127.0.0.1:8000/docs
}

ngrok_ok() {
  curl -sf --max-time 4 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | grep -q '"public_url"'
}

avatar_ok() {
  # Optional until units are installed; treat missing unit as healthy skip.
  if ! systemctl list-unit-files chatbot-avatar.service &>/dev/null; then
    return 0
  fi
  curl -sf -o /dev/null --max-time 4 http://127.0.0.1:8010/health
}

if backend_ok && ngrok_ok && avatar_ok; then
  exit 0
fi

logger -t chatbot-watchdog "Unhealthy after wake/network change; running recover"
exec "$RECOVER"
