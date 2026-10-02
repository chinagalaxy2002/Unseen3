#!/usr/bin/env bash
# Dynamic queue worker for multi-split Moment-DETR-TRM-GMR-Joint training & evaluation
# Strictly allocates 1 task per GPU concurrently
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${1:-1}"
QUEUE_DIR="$ROOT_DIR/results/moment_detr_trm_gmr_joint/queue"
mkdir -p "$QUEUE_DIR"
LOCK_DIR="$QUEUE_DIR/lock"

get_next_task() {
  local max_attempts=120
  local attempt=0
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 0.5
    ((attempt++))
    if [[ $attempt -ge $max_attempts ]]; then
      rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
  done
  
  local task=""
  for split in A1 A2_alt A3 C1 C2_alt; do
    if [[ ! -f "$QUEUE_DIR/${split}.claimed" && ! -f "$QUEUE_DIR/${split}.done" && ! -f "$QUEUE_DIR/${split}.failed" ]]; then
      touch "$QUEUE_DIR/${split}.claimed"
      echo "$GPU" > "$QUEUE_DIR/${split}.gpu"
      printf 'claimed=%s\ngpu=%s\n' "$(date -Is)" "$GPU" >> "$QUEUE_DIR/${split}.claimed"
      task="$split"
      break
    fi
  done
  
  rmdir "$LOCK_DIR" 2>/dev/null || true
  echo "$task"
}

echo "[Joint Worker GPU $GPU] Started at $(date -Is)"
while true; do
  split="$(get_next_task)"
  if [[ -z "$split" ]]; then
    echo "[Joint Worker GPU $GPU] No more unassigned tasks in queue. Exiting at $(date -Is)."
    break
  fi
  
  echo "[Joint Worker GPU $GPU] ========================================="
  echo "[Joint Worker GPU $GPU] Claimed split: $split at $(date -Is)"
  echo "[Joint Worker GPU $GPU] ========================================="
  
  LOG_DIR="results/moment_detr_trm_gmr_joint/${split}"
  mkdir -p "$LOG_DIR"
  
  # 1. Training (Formal 100 Epochs with seed=3407, max_v_l=200, use_exist_head=True)
  echo "[Joint Worker GPU $GPU] [1/2] Launching joint training for $split on GPU $GPU..."
  if CUDA_VISIBLE_DEVICES="$GPU" bash scripts/train_trm_gmr_joint.sh "$split" 2>&1 | tee "$LOG_DIR/train_runner.log"; then
    echo "[Joint Worker GPU $GPU] Joint training for $split succeeded at $(date -Is)"
  else
    echo "[Joint Worker GPU $GPU] ERROR: Joint training for $split failed" >&2
    touch "$QUEUE_DIR/${split}.failed"
    continue
  fi
  
  # 2. Inference & Evaluation (Full 4-quadrant test sets S+, S-, U+, U-)
  echo "[Joint Worker GPU $GPU] [2/2] Launching joint inference & evaluation for $split on GPU $GPU..."
  if CUDA_VISIBLE_DEVICES="$GPU" bash scripts/infer_trm_gmr_joint.sh "$split" 2>&1 | tee "$LOG_DIR/infer_runner.log"; then
    echo "[Joint Worker GPU $GPU] Joint inference for $split succeeded at $(date -Is)"
    touch "$QUEUE_DIR/${split}.done"
  else
    echo "[Joint Worker GPU $GPU] ERROR: Joint inference for $split failed" >&2
    touch "$QUEUE_DIR/${split}.failed"
  fi
  
  echo "[Joint Worker GPU $GPU] Finished pipeline for $split at $(date -Is)"
done
