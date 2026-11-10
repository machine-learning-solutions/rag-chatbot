#!/usr/bin/env bash
# Wait until avatar_service accepts connections (model load can take minutes).
for _ in $(seq 1 180); do
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8010/health 2>/dev/null; then
    exit 0
  fi
  if ss -tln 2>/dev/null | grep -q ':8010 '; then
    # Port is open but /health may still be loading — keep waiting briefly.
    sleep 2
    if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8010/health 2>/dev/null; then
      exit 0
    fi
  fi
  sleep 2
done
logger -t chatbot-assestant "Avatar :8010 not ready after wait"
exit 1
