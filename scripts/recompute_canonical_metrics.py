"""Revise V4 raw localization without modifying historical predictions/results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, RELEASE, SPLITS, digest, read_rows, write_json
from training.moment_detr_gmr_auc_v4.metrics import raw_localization_metrics


def main():
    summary = {}
    for split in SPLITS:
        path = ROOT / "results/moment_detr_gmr_auc_v4" / split / "test_predictions.jsonl"
        predictions = read_rows(path)
        rows = read_rows(RELEASE / split / "test.jsonl")
        if {r["qid"] for r in rows} != {r["qid"] for r in predictions}:
            raise ValueError("Prediction qid mismatch")
        loc = {r["qid"]: {"raw_class_logits": r["raw_class_logits_base"],
                           "raw_spans_seconds": r["raw_spans_seconds_base"]} for r in predictions}
        metrics = {p: raw_localization_metrics([r for r in rows if r["partition"] == p], loc) for p in ("S+", "U+")}
        metrics.update({"evaluator_version": "raw_margin_revision_01", "units": "fraction",
                        "postprocessing": "none; raw seconds unchanged",
                        "prediction_hash": digest(path), "existence_metrics_changed": False})
        write_json(path.parent / "metrics_revision_01.json", metrics)
        summary[split] = metrics
    write_json(EXPERIMENT / "localization_revision_01.json", summary)
    print({s: {p: m[p]["r1_iou05"] for p in ("S+", "U+")} for s, m in summary.items()})


if __name__ == "__main__":
    main()
