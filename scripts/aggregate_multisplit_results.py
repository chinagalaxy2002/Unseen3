#!/usr/bin/env python3
"""
Aggregate Moment-DETR-TRM Multi-Split Evaluation Results.
Reads generalization_summary.json for all splits (A1, A2_alt, A3, C1, C2_alt),
computes macro-averaged metrics across splits, and updates TASK.md and generalization_summary.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["A1", "A2_alt", "A3", "C1", "C2_alt"]
RESULTS_BASE = REPO_ROOT / "results" / "moment_detr_trm"
OUTPUT_SUMMARY = REPO_ROOT / "experiments" / "trm_momentdetr_generalization" / "generalization_summary.json"

# Reference Moment-DETR Baseline numbers for Charades-STA Semantic Novelty (positive-only)
BASELINE_METRICS = {
    # Reference values on Charades semantic_existence_v2 positive-only Moment-DETR
    # (Provided in baseline audit)
    "seen_positive": {"R1@0.3": 58.20, "R1@0.5": 41.50, "R1@0.7": 22.10, "mIoU": 41.80},
    "unseen_positive": {"R1@0.3": 49.30, "R1@0.5": 31.80, "R1@0.7": 15.60, "mIoU": 34.20},
    "generalization_gap": {"gap_R1@0.5": 9.70, "gap_mIoU": 7.60},
}

def main():
    split_summaries = {}
    completed_splits = []

    for split in SPLITS:
        summary_file = RESULTS_BASE / split / "eval_output" / "generalization_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            split_summaries[split] = data
            completed_splits.append(split)
        else:
            print(f"[-] Split {split}: not yet completed (missing {summary_file})")

    print(f"Completed splits: {len(completed_splits)}/{len(SPLITS)} -> {completed_splits}")
    if not completed_splits:
        print("No completed splits found.")
        return

    # Calculate average across available completed splits
    avg_s03 = sum(split_summaries[s]["seen_positive"]["R1@0.3"] for s in completed_splits) / len(completed_splits)
    avg_s05 = sum(split_summaries[s]["seen_positive"]["R1@0.5"] for s in completed_splits) / len(completed_splits)
    avg_s07 = sum(split_summaries[s]["seen_positive"]["R1@0.7"] for s in completed_splits) / len(completed_splits)
    avg_smiou = sum(split_summaries[s]["seen_positive"]["mIoU"] for s in completed_splits) / len(completed_splits)

    avg_u03 = sum(split_summaries[s]["unseen_positive"]["R1@0.3"] for s in completed_splits) / len(completed_splits)
    avg_u05 = sum(split_summaries[s]["unseen_positive"]["R1@0.5"] for s in completed_splits) / len(completed_splits)
    avg_u07 = sum(split_summaries[s]["unseen_positive"]["R1@0.7"] for s in completed_splits) / len(completed_splits)
    avg_umiou = sum(split_summaries[s]["unseen_positive"]["mIoU"] for s in completed_splits) / len(completed_splits)

    avg_gap_r1_05 = avg_s05 - avg_u05
    avg_gap_miou = avg_smiou - avg_umiou

    delta_seen_r1_05 = avg_s05 - BASELINE_METRICS["seen_positive"]["R1@0.5"]
    delta_unseen_r1_05 = avg_u05 - BASELINE_METRICS["unseen_positive"]["R1@0.5"]
    delta_seen_miou = avg_smiou - BASELINE_METRICS["seen_positive"]["mIoU"]
    delta_unseen_miou = avg_umiou - BASELINE_METRICS["unseen_positive"]["mIoU"]

    macro_summary = {
        "benchmark": "Charades-STA Semantic Novelty (semantic_existence_v2)",
        "model": "Moment-DETR-TRM",
        "protocol": "Phase 1: Pure Localization (S+ training only, no U access, max_v_l=200)",
        "completed_splits": completed_splits,
        "total_splits": len(SPLITS),
        "macro_average": {
            "seen_positive": {
                "R1@0.3": round(avg_s03, 2),
                "R1@0.5": round(avg_s05, 2),
                "R1@0.7": round(avg_s07, 2),
                "mIoU": round(avg_smiou, 2),
            },
            "unseen_positive": {
                "R1@0.3": round(avg_u03, 2),
                "R1@0.5": round(avg_u05, 2),
                "R1@0.7": round(avg_u07, 2),
                "mIoU": round(avg_umiou, 2),
            },
            "generalization_gap": {
                "gap_R1@0.5": round(avg_gap_r1_05, 2),
                "gap_mIoU": round(avg_gap_miou, 2),
            },
            "delta_vs_baseline": {
                "delta_seen_R1@0.5": round(delta_seen_r1_05, 2),
                "delta_unseen_R1@0.5": round(delta_unseen_r1_05, 2),
                "delta_seen_mIoU": round(delta_seen_miou, 2),
                "delta_unseen_mIoU": round(delta_unseen_miou, 2),
            },
        },
        "per_split_details": split_summaries,
    }

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(macro_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("MACRO AVERAGE RESULTS (MOMENT-DETR-TRM):")
    print(f"Seen (S+):   R@1@0.3={avg_s03:.2f}%, R@1@0.5={avg_s05:.2f}%, R@1@0.7={avg_s07:.2f}%, mIoU={avg_smiou:.2f}%")
    print(f"Unseen (U+): R@1@0.3={avg_u03:.2f}%, R@1@0.5={avg_u05:.2f}%, R@1@0.7={avg_u07:.2f}%, mIoU={avg_umiou:.2f}%")
    print(f"Gap (S->U):  Gap R1@0.5={avg_gap_r1_05:.2f}%, Gap mIoU={avg_gap_miou:.2f}%")
    print(f"Delta vs BL: Seen R1@0.5={delta_seen_r1_05:+.2f}%, Unseen R1@0.5={delta_unseen_r1_05:+.2f}%")
    print(f"Saved to: {OUTPUT_SUMMARY}")
    print("=" * 60)

if __name__ == "__main__":
    main()
