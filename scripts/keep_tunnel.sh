#!/bin/bash
# Self-healing SSH tunnel: Mac localhost:8000 -> remote GPU box 127.0.0.1:8000.
# Plain `ssh -fN -L` drops on idle/network blips (vast.ai is especially flaky),
# which kills long batch runs. This respawns it with keepalives if it drops.
#
# Parameterized via env vars so switching between boxes (vast / Mithril / ...)
# is just an env change — no script edit. Defaults match the box we're using
# right now.
#
#   TUNNEL_HOST  required-ish (default below)
#   TUNNEL_USER  default: ubuntu
#   TUNNEL_PORT  default: 22
#   TUNNEL_KEY   default: ~/ulusha-key.pem  (set "" to fall back to ssh-config)
#   TUNNEL_LOCAL_PORT  default: 8000
#
# Examples:
#   TUNNEL_HOST=44.250.249.199 ./scripts/keep_tunnel.sh           # Mithril (default)
#   TUNNEL_HOST=81.166.173.12 TUNNEL_USER=root TUNNEL_PORT=10753 \
#     TUNNEL_KEY="" ./scripts/keep_tunnel.sh                       # old vast setup

TUNNEL_HOST="${TUNNEL_HOST:-44.250.249.199}"
TUNNEL_USER="${TUNNEL_USER:-ubuntu}"
TUNNEL_PORT="${TUNNEL_PORT:-22}"
TUNNEL_KEY="${TUNNEL_KEY-$HOME/ulusha-key.pem}"
TUNNEL_LOCAL_PORT="${TUNNEL_LOCAL_PORT:-8000}"

KEY_OPT=()
if [ -n "$TUNNEL_KEY" ] && [ -f "$TUNNEL_KEY" ]; then
  KEY_OPT=(-i "$TUNNEL_KEY")
fi

echo "[keep_tunnel] target=$TUNNEL_USER@$TUNNEL_HOST:$TUNNEL_PORT  key=${TUNNEL_KEY:-(none)}  local=$TUNNEL_LOCAL_PORT" >&2

while true; do
  ssh "${KEY_OPT[@]}" \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=15 -N \
      -L "$TUNNEL_LOCAL_PORT:127.0.0.1:8000" \
      -p "$TUNNEL_PORT" "$TUNNEL_USER@$TUNNEL_HOST"
  echo "[keep_tunnel] $(date +%H:%M:%S) tunnel dropped; reconnecting in 3s" >&2
  sleep 3
done
