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

  restart_ngrok() {
    log "Restarting ngrok tunnel"
    systemctl reset-failed chatbot-ngrok.service 2>/dev/null || true
    systemctl restart chatbot-ngrok.service 2>/dev/null \
      || systemctl start chatbot-ngrok.service
    for _ in $(seq 1 45); do
      ngrok_ok && return 0
      sleep 2
    done
    log "ngrok API still not reporting tunnels"
    return 1
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

  log "Recover finished (backend=$(backend_ok && echo ok || echo fail) ngrok=$(ngrok_ok && echo ok || echo fail))"
) 9>"$LOCK_FILE"
