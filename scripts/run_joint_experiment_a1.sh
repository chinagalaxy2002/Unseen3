#!/usr/bin/env bash
set -euo pipefail

# Master runner for Moment-DETR-TRM-GMR-Joint Split A1
SPLIT="A1"
LOG_DIR="results/moment_detr_trm_gmr_joint/${SPLIT}"
mkdir -p "${LOG_DIR}"

echo "=================================================="
echo "Starting End-to-End Joint Experiment on Split ${SPLIT}"
echo "Step 1: Training"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 bash scripts/train_trm_gmr_joint.sh "${SPLIT}" 2>&1 | tee "${LOG_DIR}/train_runner.log"

echo "=================================================="
echo "Step 2: Inference & Evaluation"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 bash scripts/infer_trm_gmr_joint.sh "${SPLIT}" 2>&1 | tee "${LOG_DIR}/infer_runner.log"

echo "=================================================="
echo "Experiment for Split ${SPLIT} completed successfully!"
echo "=================================================="
