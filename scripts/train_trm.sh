#!/usr/bin/env bash
set -euo pipefail

# Training Runner for Moment-DETR-TRM (Phase 1: Pure Localization, S+ only)
SPLIT="${1:-A1}"
shift || true
PYTHON="/home/guoxiangyu/miniconda3/envs/univtg/bin/python"
RESULTS_DIR="results/moment_detr_trm/${SPLIT}"

echo "=================================================="
echo "Starting Moment-DETR-TRM Training on Split: ${SPLIT}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=================================================="

${PYTHON} training/moment_detr_trm/train_trm.py \
  --model moment_detr_trm \
  --dataset charades_sta_semantic_novelty \
  --feature clip_slowfast \
  --seed 3407 \
  --lr 0.0001 \
  --lr_drop 400 \
  --n_epoch 100 \
  --max_es_cnt -1 \
  --bsz 16 \
  --eval_bsz 16 \
  --max_v_l 200 \
  --max_ts_val 200 \
  --train_path "data/release/semantic_existence_v2/${SPLIT}/train.jsonl" \
  --eval_path "data/release/semantic_existence_v2/${SPLIT}/val.jsonl" \
  --t_feat_dir "features/semantic_existence_v2/shared_clip_text" \
  --phrase_feat_dir "features/phrase_data/clip_phrase" \
  --v_feat_dirs "features/charades_video/vid_slowfast" "features/charades_video/vid_clip" \
  --results_dir "${RESULTS_DIR}" \
  --device cuda \
  --lambda_refine 1.0 \
  --lambda_con 1.0 \
  --lambda_neg 0.5 \
  --lambda_exc 1.0 \
  --iou_thresh 0.1 \
  --overwrite \
  "$@"

echo "=================================================="
echo "Training finished for Split: ${SPLIT}"
echo "=================================================="
