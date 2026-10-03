#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3"; cd "$ROOT"
(bash scripts/train_trm_gmr_joint_v2.sh A1 0 && bash scripts/train_trm_gmr_joint_v2.sh A3 0 && bash scripts/train_trm_gmr_joint_v2.sh C2_alt 0) & p0=$!
(bash scripts/train_trm_gmr_joint_v2.sh A2_alt 1 && bash scripts/train_trm_gmr_joint_v2.sh C1 1) & p1=$!
wait "$p0"; wait "$p1"
"/home/guoxiangyu/miniconda3/envs/univtg/bin/python" scripts/aggregate_joint_v2.py
