#!/usr/bin/env bash
set -euo pipefail
cd /home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3
PYTHON=/home/guoxiangyu/miniconda3/envs/univtg/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if (( GPU_COUNT >= 2 )); then
 "$PYTHON" scripts/queue_worker_joint_v3.py 0 A1 A3 C2_alt & JOINT_V3_P0=$!
 "$PYTHON" scripts/queue_worker_joint_v3.py 1 A2_alt C1 & JOINT_V3_P1=$!
 wait "$JOINT_V3_P0"
 wait "$JOINT_V3_P1"
else
 "$PYTHON" scripts/queue_worker_joint_v3.py 0 A1 A2_alt A3 C1 C2_alt
fi
"$PYTHON" scripts/aggregate_joint_v3.py
