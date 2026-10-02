#!/usr/bin/env bash
set -euo pipefail

# Inference Runner for Moment-DETR-TRM-GMR-Joint (End-to-End Joint v1)
SPLIT="${1:-A1}"
CKPT_PATH="${2:-results/moment_detr_trm_gmr_joint/${SPLIT}/best.ckpt}"
RESULTS_DIR="${3:-results/moment_detr_trm_gmr_joint/${SPLIT}}"
PYTHON="/home/guoxiangyu/miniconda3/envs/univtg/bin/python"

echo "=================================================="
echo "Evaluating Moment-DETR-TRM-GMR-Joint on Split: ${SPLIT}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=================================================="

${PYTHON} training/moment_detr_trm_gmr_joint/infer.py \
  --model moment_detr_trm_gmr_joint \
  --dataset charades_sta_semantic_novelty \
  --feature clip_slowfast \
  --model_path "${CKPT_PATH}" \
  --release_dir "data/release/semantic_existence_v2/${SPLIT}" \
  --results_dir "${RESULTS_DIR}" \
  --device cuda \
  --eval_bsz 16 \
  --max_v_l 200 \
  --max_ts_val 200.0 \
  --t_feat_dir "features/semantic_existence_v2/shared_clip_text" \
  --phrase_feat_dir "features/phrase_data/clip_phrase" \
  --v_feat_dirs "features/charades_video/vid_slowfast" "features/charades_video/vid_clip"

echo "=================================================="
echo "Joint Inference finished for Split: ${SPLIT}"
echo "=================================================="
