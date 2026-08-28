#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$DIR/server.log"

echo "[H100 Qwen Deploy] Starting Qwen2.5-3B-Instruct API service on port 8000..."
nohup ~/bin/micromamba run -n mot-wam python "$DIR/service.py" > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$DIR/server.pid"
echo "[H100 Qwen Deploy] Service started with PID $PID, logs streaming to $LOG_FILE"
