"""Read-only progress report; usable while persistent queues are running."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, RESULTS, write_json


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    progress = []
    for path in sorted(EXPERIMENT.glob("baseline_queue_gpu*.json")):
        queue = json.loads(path.read_text())
        for fold, job in queue["jobs"].items():
            directory = RESULTS / "inner_baselines" / fold
            tr = directory / "train.log"
            va = directory / "val.log"
            epochs = re.findall(r"\[Epoch\]\s+(\d+)", tr.read_text()) if tr.exists() else []
            vals = re.findall(r'"MR-full-mAP":\s*([\d.]+)', va.read_text()) if va.exists() else []
            status = job["status"]
            if status == "running" and not alive(job.get("pid")):
                status = "process_missing"
            progress.append({"fold": fold, "gpu": queue["gpu"], "status": status,
                             "epochs_completed": max(map(int, epochs), default=0),
                             "epochs_planned": 100, "validation_records": len(vals),
                             "best_seen_val_mAP": max(map(float, vals), default=None),
                             "pid": job.get("pid"), "log": str(directory / "job.log")})
    result = {"stage": "inner_baselines", "jobs": progress}
    if args.save:
        write_json(EXPERIMENT / "progress_snapshot.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
