"""Certify frozen input/code provenance and actual completion before readouts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, RESULTS, digest, write_json


def main():
    freeze = json.loads((EXPERIMENT / "INNER_BASELINE_FREEZE.json").read_text())
    for name, expected in freeze["code_hashes"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Changed frozen baseline code: {name}")
    complete = []
    for fold, checksum in freeze["fold_manifest_hashes"].items():
        if digest(EXPERIMENT / "inner_folds" / fold / "manifest.json") != checksum:
            raise RuntimeError("Changed baseline fold")
        directory = RESULTS / "inner_baselines" / fold
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "completed" or status["epochs_trained"] != 100:
            raise RuntimeError("Baseline not fully complete")
        checkpoint = directory / "best.ckpt"
        if digest(checkpoint) != status["checkpoint_hash"]:
            raise RuntimeError("Baseline checkpoint changed")
        train_epochs = re.findall(r"\[Epoch\]\s+(\d+)", (directory / "train.log").read_text())
        validation = (directory / "val.log").read_text().splitlines()
        if list(map(int, train_epochs)) != list(range(1, 101)) or len(validation) != 100:
            raise RuntimeError("Incomplete or repeated epoch records")
        c = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = [json.loads(line.split("[Metrics] ", 1)[1])["brief"]["MR-full-mAP"] for line in validation]
        selected = int(c["epoch"])
        if selected != metrics.index(max(metrics)):
            raise RuntimeError("Checkpoint does not match earliest best Seen MR-mAP")
        complete.append({"fold": fold, "epochs_trained": 100, "validation_records": 100,
                         "best_epoch_1_indexed": selected + 1, "best_seen_mAP": max(metrics),
                         "checkpoint_hash": digest(checkpoint), "elapsed_seconds": status["elapsed_seconds"]})
    write_json(EXPERIMENT / "INNER_BASELINE_COMPLETION.json", {"passed": True, "baselines": complete,
               "baseline_code_unchanged": True, "novel_dev_not_used_for_selection": True})
    print(complete)


if __name__ == "__main__":
    main()
