#!/usr/bin/env bash
# Waits for A1 pipeline to complete on GPU 0, marks A1.done, then joins the queue worker on GPU 0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE_DIR="$ROOT_DIR/results/moment_detr_trm_gmr_joint/queue"
mkdir -p "$QUEUE_DIR"

echo "Waiting for Split A1 pipeline to complete on GPU 0..."
while true; do
  if [[ -f "$ROOT_DIR/results/moment_detr_trm_gmr_joint/A1/joint_summary.json" ]]; then
    echo "A1 pipeline confirmed complete at $(date -Is)! Marking A1.done..."
    touch "$QUEUE_DIR/A1.done"
    break
  fi
  sleep 10
done

echo "Starting Joint Worker on GPU 0 for remaining splits..."
bash scripts/queue_worker_joint.sh 0 2>&1 | tee "$QUEUE_DIR/worker_gpu0.log"
