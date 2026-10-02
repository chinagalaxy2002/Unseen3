#!/usr/bin/env bash
set -euo pipefail

# Inference Runner for Moment-DETR-TRM (Phase 1: Seen S+ and Unseen U+ Test Evaluation)
SPLIT="${1:-A1}"
CKPT_PATH="${2:-results/moment_detr_trm/${SPLIT}/best.ckpt}"
PYTHON="/home/guoxiangyu/miniconda3/envs/univtg/bin/python"
RESULTS_DIR="results/moment_detr_trm/${SPLIT}/eval_output"

mkdir -p "${RESULTS_DIR}"

echo "=================================================="
echo "Evaluating Moment-DETR-TRM on Split: ${SPLIT}"
echo "Checkpoint: ${CKPT_PATH}"
echo "=================================================="

# 1. Evaluate Seen Positive Test (S+)
echo "--- Evaluating Seen Positive (S+) ---"
${PYTHON} training/moment_detr_trm/eval_cli_trm.py \
  --model moment_detr_trm \
  --dataset charades_sta_semantic_novelty \
  --feature clip_slowfast \
  --model_path "${CKPT_PATH}" \
  --split "test_seen_positive" \
  --partition_filter "S+" \
  --eval_path "data/release/semantic_existence_v2/${SPLIT}/test.jsonl" \
  --t_feat_dir "features/semantic_existence_v2/shared_clip_text" \
  --phrase_feat_dir "features/phrase_data/clip_phrase" \
  --v_feat_dirs "features/charades_video/vid_slowfast" "features/charades_video/vid_clip" \
  --results_dir "${RESULTS_DIR}" \
  --device cuda \
  --max_v_l 200 \
  --max_ts_val 200 \
  --eval_bsz 16

# 2. Evaluate Unseen Positive Test (U+)
echo "--- Evaluating Unseen Positive (U+) ---"
${PYTHON} training/moment_detr_trm/eval_cli_trm.py \
  --model moment_detr_trm \
  --dataset charades_sta_semantic_novelty \
  --feature clip_slowfast \
  --model_path "${CKPT_PATH}" \
  --split "test_unseen_positive" \
  --partition_filter "U+" \
  --eval_path "data/release/semantic_existence_v2/${SPLIT}/test.jsonl" \
  --t_feat_dir "features/semantic_existence_v2/shared_clip_text" \
  --phrase_feat_dir "features/phrase_data/clip_phrase" \
  --v_feat_dirs "features/charades_video/vid_slowfast" "features/charades_video/vid_clip" \
  --results_dir "${RESULTS_DIR}" \
  --device cuda \
  --max_v_l 200 \
  --max_ts_val 200 \
  --eval_bsz 16

echo "=================================================="
echo "Inference finished. Evaluating summary metrics:"
echo "=================================================="

${PYTHON} -c "
import json
import numpy as np

def compute_metrics(pred_path, gt_path, partition):
    gt = {}
    with open(gt_path) as f:
        for line in f:
            d = json.loads(line)
            if partition and d.get('partition') != partition:
                continue
            if d.get('relevant_windows'):
                gt[str(d['qid'])] = d['relevant_windows']

    pred = {}
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            pred[str(d['qid'])] = d['pred_relevant_windows'][0][:2]

    ious = []
    for qid, gt_windows in gt.items():
        if qid not in pred:
            continue
        p = pred[qid]
        cur_max = 0.0
        for w in gt_windows:
            inter = max(0.0, min(p[1], w[1]) - max(p[0], w[0]))
            union = (p[1] - p[0]) + (w[1] - w[0]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > cur_max:
                cur_max = iou
        ious.append(cur_max)

    ious = np.array(ious)
    r1_03 = float(np.mean(ious >= 0.3) * 100)
    r1_05 = float(np.mean(ious >= 0.5) * 100)
    r1_07 = float(np.mean(ious >= 0.7) * 100)
    miou = float(np.mean(ious) * 100)
    return len(ious), r1_03, r1_05, r1_07, miou

seen_pred = '${RESULTS_DIR}/test_seen_positive_preds.jsonl'
unseen_pred = '${RESULTS_DIR}/test_unseen_positive_preds.jsonl'
gt_path = 'data/release/semantic_existence_v2/${SPLIT}/test.jsonl'

n_s, s03, s05, s07, smiou = compute_metrics(seen_pred, gt_path, 'S+')
n_u, u03, u05, u07, umiou = compute_metrics(unseen_pred, gt_path, 'U+')

gap_r1_05 = s05 - u05
gap_miou = smiou - umiou

summary = {
    'split': '${SPLIT}',
    'checkpoint': '${CKPT_PATH}',
    'seen_positive': {'count': n_s, 'R1@0.3': s03, 'R1@0.5': s05, 'R1@0.7': s07, 'mIoU': smiou},
    'unseen_positive': {'count': n_u, 'R1@0.3': u03, 'R1@0.5': u05, 'R1@0.7': u07, 'mIoU': umiou},
    'generalization_gap': {'gap_R1@0.5': gap_r1_05, 'gap_mIoU': gap_miou},
}

summary_path = '${RESULTS_DIR}/generalization_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Seen (S+) [N={n_s}]:   R@1@0.3={s03:.2f}, R@1@0.5={s05:.2f}, R@1@0.7={s07:.2f}, mIoU={smiou:.2f}')
print(f'Unseen (U+) [N={n_u}]: R@1@0.3={u03:.2f}, R@1@0.5={u05:.2f}, R@1@0.7={u07:.2f}, mIoU={umiou:.2f}')
print(f'Generalization Gap:   Gap R1@0.5={gap_r1_05:.2f}, Gap mIoU={gap_miou:.2f}')
print(f'Summary saved to: {summary_path}')
"
