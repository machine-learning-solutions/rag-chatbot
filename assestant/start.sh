#!/usr/bin/env bash
# Starts the avatar lip-sync service and the chat server together.
# Ollama (LFM3) must already be running separately (e.g. the chatbot-ollama-1 container).
set -e
cd "$(dirname "$0")"

echo "Starting avatar service on :8010 ..."
(cd avatar_service && source .venv/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 8010) &
AVATAR_PID=$!

echo "Starting chat server on :3001 ..."
node server.js &
NODE_PID=$!

trap "kill $AVATAR_PID $NODE_PID 2>/dev/null" EXIT INT TERM
wait
