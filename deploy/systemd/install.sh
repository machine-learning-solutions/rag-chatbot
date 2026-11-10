#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECOVER="$SCRIPT_DIR/chatbot-recover.sh"

echo "==> Removing old services..."
for svc in wefix-all stop-ngrok-on-suspend; do
  if systemctl list-unit-files "${svc}.service" &>/dev/null; then
    systemctl stop "${svc}.service" 2>/dev/null || true
    systemctl disable "${svc}.service" 2>/dev/null || true
    rm -f "/etc/systemd/system/${svc}.service"
    echo "    Removed ${svc}.service"
  fi
done

echo "==> Installing scripts..."
chmod +x \
  "$SCRIPT_DIR/chatbot-start.sh" \
  "$SCRIPT_DIR/chatbot-recover.sh" \
  "$SCRIPT_DIR/chatbot-watchdog.sh" \
  "$SCRIPT_DIR/chatbot-resume.sh" \
  "$SCRIPT_DIR/wait-for-gateway.sh" \
  "$SCRIPT_DIR/wait-for-avatar.sh" \
  "$SCRIPT_DIR/run-avatar.sh" \
  "$SCRIPT_DIR/run-assestant.sh"

echo "==> Installing systemd units..."
cp "$SCRIPT_DIR/chatbot.service" /etc/systemd/system/chatbot.service
cp "$SCRIPT_DIR/chatbot-ngrok.service" /etc/systemd/system/chatbot-ngrok.service
cp "$SCRIPT_DIR/chatbot-avatar.service" /etc/systemd/system/chatbot-avatar.service
cp "$SCRIPT_DIR/chatbot-assestant.service" /etc/systemd/system/chatbot-assestant.service
cp "$SCRIPT_DIR/chatbot-resume.service" /etc/systemd/system/chatbot-resume.service
cp "$SCRIPT_DIR/chatbot-watchdog.service" /etc/systemd/system/chatbot-watchdog.service
cp "$SCRIPT_DIR/chatbot-watchdog.timer" /etc/systemd/system/chatbot-watchdog.timer

echo "==> Installing sleep hook (post-wake recover)..."
mkdir -p /etc/systemd/system-sleep
install -m 755 "$SCRIPT_DIR/chatbot-resume.sh" /etc/systemd/system-sleep/chatbot-resume

systemctl --user stop chatbot-ngrok.service 2>/dev/null || true
systemctl --user disable chatbot-ngrok.service 2>/dev/null || true

systemctl daemon-reload

echo "==> Enabling services..."
systemctl enable chatbot.service
systemctl enable chatbot-ngrok.service
systemctl enable chatbot-avatar.service
systemctl enable chatbot-assestant.service
systemctl enable chatbot-resume.service
systemctl enable chatbot-watchdog.timer

# Free ngrok = one tunnel. Keep voice-phone unit installed but do not
# auto-start it on boot so chatbot-ngrok always wins after reboot.
if systemctl list-unit-files voice-phone-control-ngrok.service &>/dev/null; then
  systemctl disable voice-phone-control-ngrok.service 2>/dev/null || true
  echo "    Disabled voice-phone-control-ngrok.service (start manually when needed)"
fi

echo "==> Starting stack + watchdog..."
systemctl start chatbot-watchdog.timer
RUN_USER="${SUDO_USER:-jadaboawwad}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
# Do not stop ngrok here — recover restarts it if needed.
su - "$RUN_USER" -c "env HOME='$RUN_HOME' USER='$RUN_USER' bash '$RECOVER'" || true
systemctl start chatbot-ngrok.service 2>/dev/null || true
systemctl start chatbot-avatar.service 2>/dev/null || true
systemctl start chatbot-assestant.service 2>/dev/null || true

echo ""
echo "Done. Status:"
systemctl is-active chatbot.service chatbot-ngrok.service chatbot-avatar.service chatbot-assestant.service chatbot-watchdog.timer
systemctl status chatbot-ngrok.service --no-pager -l | tail -5
systemctl status chatbot-avatar.service --no-pager -l | tail -5

echo ""
echo "Tunnel URLs:"
sleep 2
curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tunnels', []):
    print(' ', t.get('name','?'), '->', t.get('public_url','?'))
" 2>/dev/null || echo "  (check http://localhost:4040)"

echo ""
echo "==> Enabling host mode (chatbot stays online; system suspend blocked)..."
chmod +x "$SCRIPT_DIR/enable-host-mode.sh" "$SCRIPT_DIR/disable-host-mode.sh"
bash "$SCRIPT_DIR/enable-host-mode.sh" || true
systemctl start chatbot-host-mode.service 2>/dev/null || true

echo ""
echo "Watchdog: every 90s fixes ngrok if it drops."
echo "Manual recover: bash $RECOVER"
echo "Allow Suspend again: sudo bash $SCRIPT_DIR/disable-host-mode.sh"
echo ""
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
  echo "Skipping monitor alerts (run as your user: bash $SCRIPT_DIR/../monitor/install.sh)"
else
  echo "Installing client activity alerts..."
  bash "$SCRIPT_DIR/../monitor/install.sh" || true
fi
