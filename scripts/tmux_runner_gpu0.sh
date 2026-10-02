#!/usr/bin/env bash
# Runner for tmux session joint_worker_gpu0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================================="
echo "[joint_worker_gpu0] Split A1 is currently training on GPU 0 (PID 800077)..."
echo "[joint_worker_gpu0] Streaming live output until A1 completes..."
echo "=========================================================="

TASK_LOG="/home/guoxiangyu/.gemini/antigravity-cli/brain/36709ee5-f56a-4cf1-8d79-4c594193d6b1/.system_generated/tasks/task-2049.log"
if [[ -f "$TASK_LOG" ]]; then
  tail -f --pid=800077 "$TASK_LOG" 2>/dev/null || true
fi

# Wait for A1 full pipeline (training + inference) to produce joint_summary.json
while [[ ! -f "$ROOT_DIR/results/moment_detr_trm_gmr_joint/A1/joint_summary.json" ]]; do
  echo "[joint_worker_gpu0] Waiting for A1 inference & summary at $(date -Is)..."
  sleep 10
done

echo "=========================================================="
echo "[joint_worker_gpu0] A1 successfully completed! Marking A1.done..."
echo "=========================================================="
QUEUE_DIR="$ROOT_DIR/results/moment_detr_trm_gmr_joint/queue"
mkdir -p "$QUEUE_DIR"
touch "$QUEUE_DIR/A1.done"

echo "[joint_worker_gpu0] Starting queue worker on GPU 0 for next splits..."
exec bash scripts/queue_worker_joint.sh 0 2>&1 | tee "$QUEUE_DIR/worker_gpu0.log"
