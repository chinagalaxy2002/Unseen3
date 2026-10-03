"""Persistent per-device inner-baseline queue with explicit state records."""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, RESULTS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    args = parser.parse_args()
    states = {fold: {"status": "queued"} for fold in args.folds}
    status_path = EXPERIMENT / f"baseline_queue_gpu{args.gpu}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    write_json(status_path, {"gpu": args.gpu, "pid": os.getpid(), "jobs": states})
    for fold in args.folds:
        output = RESULTS / "inner_baselines" / fold
        output.mkdir(parents=True, exist_ok=True)
        log = output / "job.log"
        states[fold] = {"status": "running", "log": str(log), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(status_path, {"gpu": args.gpu, "pid": os.getpid(), "jobs": states})
        with log.open("a") as stream:
            child = subprocess.Popen([sys.executable, "-m", "training.moment_detr_gmr_evidence_v5.train_inner_baseline", "--fold", fold],
                                     cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            states[fold]["pid"] = child.pid
            write_json(status_path, {"gpu": args.gpu, "pid": os.getpid(), "jobs": states})
            code = child.wait()
        states[fold].update(status="completed" if code == 0 else "failed", exit_code=code)
        write_json(status_path, {"gpu": args.gpu, "pid": os.getpid(), "jobs": states})
        if code:
            # Stop this dependent queue; leave remaining jobs explicitly queued.
            raise SystemExit(code)


if __name__ == "__main__":
    main()
