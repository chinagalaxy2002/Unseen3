#!/usr/bin/env bash
set -euo pipefail
SPLIT="${1:?split required}"; GPU="${2:-0}"
ROOT="/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3"; cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" /home/guoxiangyu/miniconda3/envs/univtg/bin/python \
  training/moment_detr_trm_gmr_joint_v2/infer.py --model moment_detr_trm_gmr_joint_v2 \
  --model_path "results/moment_detr_trm_gmr_joint_v2/${SPLIT}/best.ckpt" \
  --release_dir "data/release/semantic_existence_v2/${SPLIT}" \
  --results_dir "results/moment_detr_trm_gmr_joint_v2/${SPLIT}" --device cuda --eval_bsz 16
