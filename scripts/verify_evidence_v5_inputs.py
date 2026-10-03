"""Meaningful protocol and ranking checks before baseline experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, RELEASE, digest, read_rows, semantic, write_json
from training.moment_detr_gmr_auc_v4.metrics import raw_localization_metrics
from training.moment_detr_gmr_evidence_v5.train_inner_baseline import run


def main():
    # foreground logit prefers slot 0; foreground probability must prefer slot 1.
    row = {"qid": "counterexample", "exist_label": 1, "relevant_windows": [[0, 1]]}
    predictions = {"counterexample": {"raw_class_logits": [[5, 6], [3, 0]],
                                        "raw_spans_seconds": [[2, 3], [0, 1]]}}
    metrics = raw_localization_metrics([row], predictions)
    assert metrics["r1_iou05"] == 1.0 and metrics["raw_miou"] == 1.0
    rng = np.random.default_rng(3407)
    logits = torch.tensor(rng.normal(size=(100, 10, 2)), dtype=torch.float64)
    assert torch.equal(logits.softmax(-1)[..., 0].argmax(-1), (logits[..., 0] - logits[..., 1]).argmax(-1))
    index = json.loads((EXPERIMENT / "inner_fold_index.json").read_text())
    verified = []
    for fold in index["folds"]:
        run(fold["fold"], verify_only=True)
        train = read_rows(fold["files"]["inner_train"]["path"], seen_only=True)
        source = {str(r["qid"]): r for r in read_rows(RELEASE / fold["parent_split"] / "train.jsonl", seen_only=True)}
        heldout = set(fold["heldout"])
        assert all(semantic(r, fold["level"]) not in heldout for r in train)
        assert all(semantic(source[str(r["source_qid"])], fold["level"]) not in heldout for r in train if not r["exist_label"])
        verified.append(fold["fold"])
    write_json(EXPERIMENT / "input_verification.json", {"passed": True, "ranking_counterexample_passed": True,
               "softmax_margin_agreement_passed": True, "verified_folds": verified,
               "local_time_mapping_status": "nominal_1s_grid; exact extraction timestamps unavailable; local ROI gate pending",
               "scope": "baseline inputs/ranking; not verifier implementation validation"})
    print("PASS", verified)


if __name__ == "__main__":
    main()
