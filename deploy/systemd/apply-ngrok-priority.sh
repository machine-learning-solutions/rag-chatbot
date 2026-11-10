#!/usr/bin/env bash
# Make chatbot-ngrok win the free-ngrok boot race over voice-phone-control.
# Usage: sudo bash deploy/systemd/apply-ngrok-priority.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/chatbot-ngrok.service" /etc/systemd/system/chatbot-ngrok.service
cp "$SCRIPT_DIR/chatbot-avatar.service" /etc/systemd/system/chatbot-avatar.service
cp "$SCRIPT_DIR/chatbot-assestant.service" /etc/systemd/system/chatbot-assestant.service
systemctl daemon-reload
systemctl enable chatbot-ngrok.service chatbot-avatar.service chatbot-assestant.service
systemctl disable voice-phone-control-ngrok.service 2>/dev/null || true
systemctl stop voice-phone-control-ngrok.service 2>/dev/null || true
systemctl restart chatbot-ngrok.service
systemctl restart chatbot-avatar.service
systemctl restart chatbot-assestant.service

echo "Enabled/active:"
systemctl is-enabled chatbot-ngrok.service chatbot-avatar.service chatbot-assestant.service voice-phone-control-ngrok.service 2>&1 || true
systemctl is-active chatbot-ngrok.service chatbot-avatar.service chatbot-assestant.service voice-phone-control-ngrok.service 2>&1 || true
echo "Tunnels:"
sleep 2
curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tunnels', []):
    print(' ', t.get('name'), '->', t.get('public_url'), '=>', t.get('config', {}).get('addr'))
" 2>/dev/null || echo "  (ngrok API not up yet)"
