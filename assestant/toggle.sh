#!/usr/bin/env bash
# Start/stop the AI Agent chat app (Node server + avatar service).
# Ollama is left alone -- it's a separate, always-on Docker container.
#
# Usage:
#   toggle.sh          toggle: stop if running, start if not
#   toggle.sh start    start only if not already running (idempotent)
#   toggle.sh stop     stop only if running (idempotent)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE_PATTERN="node server.js"
AVATAR_PATTERN="uvicorn main:app"
# 3000 is left free for the portfolio's own dev server (its default port).
NODE_PORT=3001

running() { pgrep -f "$1" > /dev/null 2>&1; }
# Check actual port liveness, not just process-list matching -- a stale,
# unkillable process can match the pgrep pattern without actually listening.
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- 3>&-; }

start() {
  if port_open "$NODE_PORT" && port_open 8010; then
    echo "AI Agent already running."
    return
  fi
  echo "Starting avatar service on :8010..."
  (cd "$DIR/avatar_service" && source .venv/bin/activate && nohup python -m uvicorn main:app --host 0.0.0.0 --port 8010 > "$DIR/.avatar.log" 2>&1 &)
  echo "Starting chat server on :$NODE_PORT..."
  (cd "$DIR" && PORT="$NODE_PORT" nohup node server.js > "$DIR/.node.log" 2>&1 &)
  sleep 1
  echo "Started. Logs: $DIR/.avatar.log , $DIR/.node.log"
  echo "Open http://localhost:$NODE_PORT"
}

stop() {
  if ! port_open "$NODE_PORT" && ! port_open 8010; then
    echo "AI Agent already stopped."
    return
  fi
  echo "Stopping AI Agent..."
  pkill -f "$NODE_PATTERN" 2>/dev/null
  pkill -f "$AVATAR_PATTERN" 2>/dev/null
  echo "Stopped."
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  "")
    if port_open "$NODE_PORT" || port_open 8010; then stop; else start; fi
    ;;
  *)
    echo "Usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
