"""Persistent staged cache/train/eval queue; all variants are frozen beforehand."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, RESULTS, digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    args = parser.parse_args()
    protocol = EXPERIMENT / "READOUT_FREEZE.json"
    frozen = json.loads(protocol.read_text())
    name = f"readout_queue_gpu{args.gpu}"
    state = {"gpu": args.gpu, "pid": os.getpid(), "stage": "P3_readouts",
             "freeze_hash": digest(protocol), "status": "running", "tasks": []}
    status_path = EXPERIMENT / (name + ".json")
    queue_started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    logdir = RESULTS / "queues"
    logdir.mkdir(parents=True, exist_ok=True)

    def execute(module, arguments, task_name):
        log = logdir / (task_name + ".log")
        task = {"task": task_name, "status": "running", "log": str(log), "started_unix": time.time()}
        state["tasks"].append(task)
        write_json(status_path, state)
        with log.open("a") as stream:
            child = subprocess.Popen([sys.executable, "-m", module, *arguments], cwd=ROOT,
                                     env=env, stdout=stream, stderr=subprocess.STDOUT)
            task["pid"] = child.pid
            write_json(status_path, state)
            remaining = frozen["per_device_wall_budget_seconds"] - (time.time() - queue_started)
            try:
                code = child.wait(timeout=max(1, remaining))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                task.update(status="budget_exhausted", exit_code=None)
                state["status"] = "budget_exhausted"
                write_json(status_path, state)
                raise SystemExit("Frozen per-device readout budget exhausted")
        task.update(status="completed" if code == 0 else "failed", exit_code=code,
                    elapsed_seconds=time.time() - task["started_unix"])
        if code:
            state["status"] = "failed"
        write_json(status_path, state)
        if code:
            raise SystemExit(code)

    # Finish cache identities for both folds before training any variants.
    for fold in args.folds:
        execute("training.moment_detr_gmr_evidence_v5.feature_bank", ["--fold", fold], fold + "_extract")
    for variant in frozen["variants"]:
        for fold in args.folds:
            stem = f"{fold}_{variant}_seed3407"
            result = RESULTS / "readouts" / fold / f"{variant}_P3_bce_seed3407" / "result.json"
            if result.exists():
                raise RuntimeError("Existing result; refuse implicit reuse")
            execute("training.moment_detr_gmr_evidence_v5.train_readout", ["--fold", fold, "--variant", variant], stem + "_train")
            execute("training.moment_detr_gmr_evidence_v5.infer_readout", ["--fold", fold, "--variant", variant], stem + "_eval")
            # A partial table is safe; it never controls the frozen queue settings.
            with (logdir / "aggregate.log").open("a") as stream:
                subprocess.run([sys.executable, "scripts/aggregate_evidence_v5_readouts.py"], cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
    state["status"] = "completed"
    write_json(status_path, state)


if __name__ == "__main__":
    main()
