#!/usr/bin/env bash
set -euo pipefail
SPLIT="${1:?split required}"; GPU="${2:-0}"
ROOT="/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3"
cd "$ROOT"
PYTHON="/home/guoxiangyu/miniconda3/envs/univtg/bin/python"
RESULTS="results/moment_detr_trm_gmr_joint_v2/${SPLIT}"
"$PYTHON" scripts/fit_background_pca_joint_v2.py "$SPLIT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" training/moment_detr_trm_gmr_joint_v2/train.py \
  --model moment_detr_trm_gmr_joint_v2 --dataset charades_sta_semantic_novelty --feature clip_slowfast \
  --seed 3407 --lr 0.0001 --lr_drop 400 --n_epoch 100 --max_es_cnt -1 --bsz 16 --eval_bsz 16 \
  --max_v_l 200 --max_ts_val 200 --train_path "data/release/semantic_existence_v2/${SPLIT}/train.jsonl" \
  --eval_path "data/release/semantic_existence_v2/${SPLIT}/val.jsonl" \
  --t_feat_dir features/semantic_existence_v2/shared_clip_text --phrase_feat_dir features/phrase_data/clip_phrase \
  --v_feat_dirs features/charades_video/vid_slowfast features/charades_video/vid_clip \
  --results_dir "$RESULTS" --device cuda --lambda_refine 1 --phrase_scale 10 --lambda_con 1 --lambda_neg 0.5 \
  --lambda_exc 1 --exist_loss_coef 1 --lambda_auc 0.5 --auc_margin 1
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" training/moment_detr_trm_gmr_joint_v2/infer.py --model moment_detr_trm_gmr_joint_v2 \
  --model_path "$RESULTS/best.ckpt" --release_dir "data/release/semantic_existence_v2/${SPLIT}" \
  --results_dir "$RESULTS" --device cuda --eval_bsz 16
