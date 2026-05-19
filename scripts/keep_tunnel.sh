#!/bin/bash
# Self-healing SSH tunnel: Mac localhost:8000 -> box 127.0.0.1:8000.
# Plain `ssh -fN -L` drops on idle/network blips (vast.ai is flaky), which
# kills long batch runs. This respawns it with keepalives if it drops.
HOST="root@81.166.173.12"; PORT=10753
while true; do
  ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=15 -N -L 8000:127.0.0.1:8000 -p "$PORT" "$HOST"
  echo "[keep_tunnel] $(date +%H:%M:%S) tunnel dropped; reconnecting in 3s" >&2
  sleep 3
done
