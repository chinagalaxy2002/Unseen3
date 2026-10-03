#!/usr/bin/env bash
set -euo pipefail
split="${1:?usage: run_gmr_auc_v4_split.sh SPLIT DEVICE}"
device="${2:-cuda:0}"
[[ -f experiments/moment_detr_gmr_auc_v4/EXPERIMENT_FREEZE.json ]] || { echo 'Missing frozen protocol' >&2; exit 1; }
if [[ ! -f "results/moment_detr_gmr_auc_v4/$split/feature_bank_train.npz" ]]; then
  python -m training.moment_detr_gmr_auc_v4.extract_feature_bank --split "$split" --part train --device "$device"
fi
if [[ ! -f "results/moment_detr_gmr_auc_v4/$split/feature_bank_val.npz" ]]; then
  python -m training.moment_detr_gmr_auc_v4.extract_feature_bank --split "$split" --part val --device "$device"
fi
python -m training.moment_detr_gmr_auc_v4.train_adapter --split "$split" --device "$device"
python -m training.moment_detr_gmr_auc_v4.infer --split "$split" --device "$device"
