#!/usr/bin/env bash
set -euo pipefail

# Multi-split training and evaluation runner for Moment-DETR-TRM
# Benchmark splits: A1, A2_alt, A3, C1, C2_alt
SPLITS=("A1" "A2_alt" "A3" "C1" "C2_alt")

echo "=================================================="
echo "Starting Moment-DETR-TRM Multi-Split Execution"
echo "Splits: ${SPLITS[*]}"
echo "=================================================="

for split in "${SPLITS[@]}"; do
  echo ">>> [1/2] Training on split: ${split} <<<"
  bash scripts/train_trm.sh "${split}"

  echo ">>> [2/2] Evaluating on split: ${split} <<<"
  bash scripts/infer_trm.sh "${split}"
done

echo "=================================================="
echo "All 5 splits completed successfully!"
echo "=================================================="
