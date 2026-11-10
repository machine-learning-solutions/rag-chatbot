#!/usr/bin/env bash
# Full stack recovery after suspend, network drop, or stale ngrok tunnel.
set -euo pipefail

ROOT="/home/jadaboawwad/Files/Software/Repositories/Applications/Portfolio/chatbot"
ERR="/tmp/chatbot-compose.err"
LOCK_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/chatbot"
LOCK_FILE="$LOCK_DIR/recover.lock"
mkdir -p "$LOCK_DIR"

log() { logger -t chatbot-recover "$*"; }

(
  flock -n 9 || {
    log "Recover already running; skip"
    exit 0
  }

  cd "$ROOT"

  backend_ok() {
    curl -sf -o /dev/null --max-time 4 http://127.0.0.1:8000/docs
  }

  ngrok_ok() {
    curl -sf --max-time 4 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
      | grep -q '"public_url"'
  }

  wait_for_network() {
    for _ in $(seq 1 60); do
      ping -c1 -W2 1.1.1.1 >/dev/null 2>&1 && return 0
      sleep 2
    done
    log "Network still down; continuing anyway"
  }

  wait_for_docker() {
    for _ in $(seq 1 60); do
      docker info >/dev/null 2>&1 && return 0
      sleep 2
    done
    log "Docker not ready"
    return 1
  }

  reconcile_and_up() {
    log "Reconciling Docker stack"
    docker compose down --remove-orphans 2>/dev/null || true
    mapfile -t stale < <(docker ps -aq --filter "name=^chatbot-")
    if ((${#stale[@]})); then
      docker rm -f "${stale[@]}" 2>/dev/null || true
    fi
    docker compose up -d --remove-orphans
  }

  ensure_stack() {
    if backend_ok; then
      log "Backend already healthy"
      return 0
    fi
    if docker compose up -d --remove-orphans 2>"$ERR"; then
      log "Compose up succeeded"
    elif grep -qE "already in use|Conflict|network .+ not found|failed to set up container networking" "$ERR"; then
      # After reboot Docker often keeps container configs pointing at a
      # network ID that no longer exists -- only a full down/up recreates it.
      log "Stale compose state; reconciling"
      reconcile_and_up
    else
      cat "$ERR" >&2
      log "Compose failed: $(head -1 "$ERR")"
      return 1
    fi
    for _ in $(seq 1 45); do
      backend_ok && return 0
      sleep 2
    done
    log "Backend did not become healthy"
    return 1
  }

  try_systemctl() {
    # Recover runs as the login user; system units often need polkit auth.
    systemctl "$@" >/dev/null 2>&1
  }

  restart_ngrok() {
    log "Restarting ngrok tunnel"
    try_systemctl reset-failed chatbot-ngrok.service || true
    if ! try_systemctl restart chatbot-ngrok.service && ! try_systemctl start chatbot-ngrok.service; then
      log "systemctl ngrok failed (polkit?); starting ngrok directly"
      pkill -u "$(id -un)" -f '/usr/local/bin/ngrok start' 2>/dev/null || true
      sleep 1
      # Same tunnel name the unit uses; wait-for-gateway already passed via stack.
      nohup /usr/local/bin/ngrok start chatbot \
        >/tmp/chatbot-ngrok-direct.log 2>&1 &
    fi
    for _ in $(seq 1 30); do
      ngrok_ok && return 0
      sleep 2
    done
    log "ngrok API still not reporting tunnels"
    return 1
  }

  avatar_ok() {
    curl -sf -o /dev/null --max-time 4 http://127.0.0.1:8010/health
  }

  assestant_ok() {
    curl -sf -o /dev/null --max-time 4 http://127.0.0.1:3001/avatar/reference.png \
      || curl -sf -o /dev/null --max-time 4 http://127.0.0.1:3001/
  }

  ensure_avatar_stack() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if ! avatar_ok; then
      log "Starting avatar"
      try_systemctl reset-failed chatbot-avatar.service || true
      if ! try_systemctl start chatbot-avatar.service; then
        nohup "$SCRIPT_DIR/run-avatar.sh" >/tmp/chatbot-avatar-direct.log 2>&1 &
      fi
    fi
    # Give avatar a moment to bind before assestant (or wait-for-avatar if unit).
    for _ in $(seq 1 60); do
      avatar_ok && break
      sleep 2
    done
    if ! assestant_ok; then
      log "Starting assestant"
      try_systemctl reset-failed chatbot-assestant.service || true
      if ! try_systemctl start chatbot-assestant.service; then
        nohup "$SCRIPT_DIR/run-assestant.sh" >/tmp/chatbot-assestant-direct.log 2>&1 &
      fi
    fi
  }

  log "Recover started"
  wait_for_network
  wait_for_docker
  ensure_stack

  if ngrok_ok; then
    log "ngrok already healthy"
  else
    restart_ngrok || true
  fi

  ensure_avatar_stack || true

  log "Recover finished (backend=$(backend_ok && echo ok || echo fail) ngrok=$(ngrok_ok && echo ok || echo fail) avatar=$(avatar_ok && echo ok || echo fail))"
) 9>"$LOCK_FILE"
