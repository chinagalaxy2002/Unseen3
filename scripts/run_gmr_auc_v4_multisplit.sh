#!/usr/bin/env bash
set -euo pipefail
[[ -f experiments/moment_detr_gmr_auc_v4/EXPERIMENT_FREEZE.json ]] || { echo 'Missing frozen protocol' >&2; exit 1; }
run_queue() { local device="$1"; shift; for split in "$@"; do bash scripts/run_gmr_auc_v4_split.sh "$split" "$device"; done; }
run_queue cuda:0 A1 A3 C2_alt > results/moment_detr_gmr_auc_v4/gpu0_queue.log 2>&1 & p0=$!
run_queue cuda:1 A2_alt C1 > results/moment_detr_gmr_auc_v4/gpu1_queue.log 2>&1 & p1=$!
wait "$p0"; wait "$p1"
python scripts/aggregate_gmr_auc_v4.py
