#!/usr/bin/env bash
# Start dual-GPU parallel pipeline for Moment-DETR-TRM-GMR-Joint
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE_DIR="$ROOT_DIR/results/moment_detr_trm_gmr_joint/queue"
mkdir -p "$QUEUE_DIR"

# 1. Mark A1 as claimed by GPU 0 (it is already running smoothly on GPU 0)
touch "$QUEUE_DIR/A1.claimed"
echo "0" > "$QUEUE_DIR/A1.gpu"
printf 'claimed=%s\ngpu=0\n' "$(date -Is)" >> "$QUEUE_DIR/A1.claimed"

# 2. Launch Worker 1 on GPU 1 immediately in background (will claim A2_alt)
echo "Starting Worker 1 on GPU 1 (will claim next task, e.g. A2_alt)..."
bash scripts/queue_worker_joint.sh 1 > "$QUEUE_DIR/worker_gpu1.log" 2>&1 &
WORKER1_PID=$!
echo "Worker 1 PID: $WORKER1_PID on GPU 1"

# 3. Launch Watcher on GPU 0 in background (waits for A1, marks done, then joins queue worker on GPU 0)
echo "Starting Watcher on GPU 0 (will join queue worker when A1 completes)..."
bash scripts/watch_and_run_gpu0.sh > "$QUEUE_DIR/watcher_gpu0.log" 2>&1 &
WATCHER0_PID=$!
echo "Watcher 0 PID: $WATCHER0_PID on GPU 0"

echo "Dual-GPU parallel pipeline initialized successfully!"
echo "GPU 0: Running A1 -> then will pick up next available split"
echo "GPU 1: Running Worker 1 (A2_alt) -> then will pick up next available split"
echo "Queue logs in: $QUEUE_DIR"
