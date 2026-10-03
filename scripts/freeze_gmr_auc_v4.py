from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.moment_detr_gmr_auc_v4.common import SPLITS, write_json


def main():
    audit = json.loads((ROOT / "experiments/moment_detr_gmr_auc_v4/baseline_audit.json").read_text())
    if not audit["canonical_baseline_available"]: raise SystemExit("Cannot freeze without five canonical baseline checkpoints")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    exp = json.loads((ROOT / "configs/moment_detr_gmr_auc_v4/experiment.json").read_text())
    obj = {
        "experiment": exp["name"], "short_name": exp["short_name"], "status": exp["exploratory_status"],
        "git_sha": sha, "frozen_before_any_v4_test_u_inference": True,
        "u_policy": {"training": False, "adapter_loss": False, "pair_construction": False, "epoch_selection": False,
                     "hyperparameter_selection": False, "threshold_calibration": False, "baseline_checkpoint_selection": False},
        "shared_architecture_and_hyperparameters_across_splits": True,
        "splits": {s: {"baseline_checkpoint": audit["splits"][s]["checkpoint"],
                        "baseline_checkpoint_sha256": audit["splits"][s]["checkpoint_sha256"],
                        "baseline_resolved_config": audit["splits"][s],
                        "feature_order": audit["splits"][s]["feature_order"],
                        "base_existence_pooling": audit["splits"][s]["exist_pool"]} for s in SPLITS},
        "adapter_architecture": "LayerNorm(D+1) -> Linear(D+1,64) -> GELU -> Dropout(0.1) -> Linear(64,1); final layer weight/bias zero",
        "residual_bound": 2.0, "loss_coefficients": exp["loss_coefficients"], "auc_margin": exp["auc_margin"],
        "optimizer": exp["optimizer"], "epochs": exp["epochs"], "batch_sizes": {"bce": 256, "global_positive": 128, "global_negative": 128, "same_semantic_pairs": 256},
        "sampling_protocol": {"bce": "uniform shuffle, one complete traversal per epoch, empirical class distribution",
                              "global_auc": "uniform S+ and S- draws independent of semantics; no replacement unless fewer than 128",
                              "same_semantic_priority": ["identical normalized query", "identical composition", "identical action"],
                              "same_semantic_replacement": "paired anchors sampled uniformly; replacement only if paired-anchor count < 256"},
        "checkpoint_selection_rule": exp["checkpoint_selection"],
        "canonical_auroc_definition": exp["canonical_auroc"],
        "exploratory_status": "GMR-AUC-v4 is an exploratory follow-up and is not an untouched unseen evaluation.",
    }
    write_json(ROOT / "experiments/moment_detr_gmr_auc_v4/EXPERIMENT_FREEZE.json", obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
