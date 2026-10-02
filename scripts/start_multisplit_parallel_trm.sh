#!/usr/bin/env bash
# Start parallel multi-split workers across GPU 0 and GPU 1 for Moment-DETR-TRM
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE_DIR="$ROOT_DIR/results/moment_detr_trm/queue"
mkdir -p "$QUEUE_DIR"

# Clean any previous queue metadata
rm -f "$QUEUE_DIR"/*.claimed "$QUEUE_DIR"/*.done "$QUEUE_DIR"/*.failed "$QUEUE_DIR"/*.gpu
rmdir "$QUEUE_DIR/lock" 2>/dev/null || true

# Kill existing worker tmux sessions if running
tmux kill-session -t trm_worker_gpu0 2>/dev/null || true
tmux kill-session -t trm_worker_gpu1 2>/dev/null || true

echo "Starting Worker 0 on GPU 0 (tmux: trm_worker_gpu0)..."
tmux new-session -d -s trm_worker_gpu0 "bash scripts/queue_worker_trm.sh 0 2>&1 | tee results/moment_detr_trm/queue/worker_gpu0.log"

echo "Starting Worker 1 on GPU 1 (tmux: trm_worker_gpu1)..."
tmux new-session -d -s trm_worker_gpu1 "bash scripts/queue_worker_trm.sh 1 2>&1 | tee results/moment_detr_trm/queue/worker_gpu1.log"

echo "Both workers launched successfully."
echo "Active tmux sessions: trm_worker_gpu0, trm_worker_gpu1"
echo "Queue logs in: results/moment_detr_trm/queue/"
