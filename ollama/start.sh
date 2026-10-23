#!/usr/bin/env sh
set -eu

MODEL_NAME="${OLLAMA_MODEL:-gemma4:e2b}"

# Start the server in the background so we can pull a model.
ollama serve &
OLLAMA_PID="$!"

wait_for_server() {
  i=0
  until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
      echo "Ollama did not become ready in time" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_server

# Best-effort pull; don't crash the service if pull fails (e.g., model name typo).
ollama pull "$MODEL_NAME" || true

# Keep the main process alive as the server.
wait "$OLLAMA_PID"

