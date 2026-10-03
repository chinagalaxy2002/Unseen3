"""Evaluate novel dev only after Seen-only model selection and calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.moment_detr_gmr_evidence_v5.readouts import make_readout
from training.moment_detr_gmr_evidence_v5.feature_bank import Bank
from training.moment_detr_gmr_evidence_v5.metrics import diagnostic, cluster_delta_interval
from training.moment_detr_gmr_evidence_v5.protocol import RESULTS, EXPERIMENT, digest, write_json, write_rows
from training.moment_detr_gmr_evidence_v5.train_readout import predict


def threshold(labels, scores):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    candidates = np.concatenate([np.unique(scores), [np.nextafter(scores.max(), np.inf)]])
    best, value = -1.0, None
    for t in candidates:
        decisions = scores >= t
        score = .5 * (decisions[labels == 1].mean() + (~decisions[labels == 0]).mean())
        if score > best:
            best, value = float(score), float(t)
    return {"logit_threshold": value, "seen_balanced_accuracy": best}


def evaluate(fold, variant, device="cuda:0", seed=3407, pair_weight=0.0, bootstrap=0):
    stage = "S1_pair" if pair_weight else "P3_bce"
    output = RESULTS / "readouts" / fold / f"{variant}_{stage}_seed{seed}"
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status["status"] not in ("selected", "evaluated") or status.get("epochs_completed") != 50:
        raise RuntimeError("Refusing to evaluate an unselected or incomplete run")
    checkpoint = output / "best.ckpt"
    if digest(checkpoint) != status["checkpoint_hash"]:
        raise RuntimeError("Selected checkpoint changed")
    c = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if c["freeze_hash"] != digest(Path(c["freeze_path"])) or c["pair_weight"] != pair_weight:
        raise RuntimeError("Protocol/pair-weight mismatch")
    model, keys = make_readout(variant)
    model.load_state_dict(c["model"], strict=True)
    model.to(device).eval()
    seen = Bank(fold, "inner_seen_val")
    if seen.metadata["identity"] != c["bank_val_identity"]:
        raise RuntimeError("Seen bank changed after model selection")
    seen_scores = predict(model, seen, keys, device)
    baseline_seen = np.asarray(seen.arrays["base_logit"], dtype=np.float64)
    calibrations = {"learned": threshold(seen.labels, seen_scores), "baseline": threshold(seen.labels, baseline_seen),
                    "checkpoint_hash": digest(checkpoint), "role": "inner_seen_val"}
    write_json(output / "threshold_frozen.json", calibrations)
    # No novel-development inputs are opened until checkpoint/threshold are fixed.
    novel = Bank(fold, "inner_novel_dev")
    novel_scores = predict(model, novel, keys, device)
    baseline_novel = np.asarray(novel.arrays["base_logit"], dtype=np.float64)
    result = {"fold": fold, "variant": variant, "seed": seed, "pair_weight": pair_weight,
              "best_epoch": c["best_epoch"], "deployment": c["deployment"],
              "baseline": {"seen": diagnostic(seen.rows, baseline_seen), "novel": diagnostic(novel.rows, baseline_novel)},
              "learned": {"seen": diagnostic(seen.rows, seen_scores), "novel": diagnostic(novel.rows, novel_scores)},
              "checkpoint_hash": digest(checkpoint), "trainable_parameters": status["trainable_parameters"],
              "localization_preserved": "offline readout cannot change frozen localization tensors",
              "novel_dev_used_for_epoch_selection": False, "formal_u_read": False,
              "threshold_role": "inner_seen_val", "elapsed_training_seconds": status["elapsed_seconds"]}
    result["delta_novel_auroc"] = result["learned"]["novel"]["pooled_auroc"] - result["baseline"]["novel"]["pooled_auroc"]
    result["delta_seen_auroc"] = result["learned"]["seen"]["pooled_auroc"] - result["baseline"]["seen"]["pooled_auroc"]
    result["policy_novel_auroc"] = result["learned" if c["deployment"] == "learned" else "baseline"]["novel"]["pooled_auroc"]
    if bootstrap:
        result["novel_cluster_bootstrap"] = cluster_delta_interval(novel.rows, baseline_novel, novel_scores, bootstrap)
    predictions = []
    for role, bank, base, scores in (("inner_seen_val", seen, baseline_seen, seen_scores),
                                     ("inner_novel_dev", novel, baseline_novel, novel_scores)):
        for row, s0, score in zip(bank.rows, base, scores):
            predictions.append({"qid": row["qid"], "vid": row["vid"], "role": role,
                                "exist_label": row["exist_label"], "source_qid": row.get("source_qid"),
                                "construction_type": row.get("construction_type"),
                                "base_logit": float(s0), "learned_logit": float(score)})
    write_rows(output / "predictions.jsonl", predictions)
    write_json(output / "result.json", result)
    status.update(status="evaluated", result_hash=digest(output / "result.json"))
    write_json(status_path, status)
    print(f"{fold} {variant}: novel baseline={result['baseline']['novel']['pooled_auroc']:.6f} learned={result['learned']['novel']['pooled_auroc']:.6f} delta={result['delta_novel_auroc']:+.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pair-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.fold, args.variant, args.device, args.seed, args.pair_weight, args.bootstrap)
