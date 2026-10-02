#!/usr/bin/env python3
"""
Aggregate Moment-DETR-TRM-GMR-Joint Multi-Split Evaluation Results.
Reads joint_summary.json for all splits (A1, A2_alt, A3, C1, C2_alt),
computes macro-averaged metrics across splits, and updates comparison table vs Moment-DETR-GMR.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["A1", "A2_alt", "A3", "C1", "C2_alt"]
RESULTS_BASE = REPO_ROOT / "results" / "moment_detr_trm_gmr_joint"
OUTPUT_SUMMARY = REPO_ROOT / "experiments" / "trm_gmr_joint_v1" / "multi_split_summary.json"
OUTPUT_MD = REPO_ROOT / "experiments" / "trm_gmr_joint_v1" / "MULTI_SPLIT_RESULT.md"

# Official Moment-DETR-GMR Baseline from benchmark documentation
BASELINE_GMR = {
    "A1": {"seen_auroc": 0.8044, "unseen_auroc": 0.4973, "gap": 0.3071},
    "A2_alt": {"seen_auroc": 0.7690, "unseen_auroc": 0.5511, "gap": 0.2180},
    "A3": {"seen_auroc": 0.7488, "unseen_auroc": 0.5643, "gap": 0.1845},
    "C1": {"seen_auroc": 0.7610, "unseen_auroc": 0.5621, "gap": 0.1989},
    "C2_alt": {"seen_auroc": 0.6759, "unseen_auroc": 0.4687, "gap": 0.2073},
}

# Baseline Moment-DETR-TRM Localization (S+ R1@0.5 -> U+ R1@0.5)
BASELINE_TRM_LOCALIZATION = {
    "A1": {"seen_r1_05": 43.87, "unseen_r1_05": 23.01, "gap": 20.86},
    "A2_alt": {"seen_r1_05": 36.38, "unseen_r1_05": 25.60, "gap": 10.79},
    "A3": {"seen_r1_05": 36.85, "unseen_r1_05": 51.56, "gap": -14.71},
    "C1": {"seen_r1_05": 37.01, "unseen_r1_05": 47.53, "gap": -10.52},
    "C2_alt": {"seen_r1_05": 37.95, "unseen_r1_05": 36.21, "gap": 1.74},
}


def main():
    split_summaries = {}
    completed_splits = []

    for split in SPLITS:
        summary_file = RESULTS_BASE / split / "joint_summary.json"
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

    # Build Markdown Comparison
    lines = [
        "# Moment-DETR-TRM-GMR-Joint vs Baseline Comparison Across Splits",
        "",
        f"**Completed splits**: {len(completed_splits)}/{len(SPLITS)} ({', '.join(completed_splits)})",
        "",
        "## 1. Existence AUROC Comparison vs Moment-DETR-GMR",
        "",
        "| Split | Baseline Seen AUROC | Baseline Unseen AUROC | Baseline Gap | Joint Seen AUROC | Joint Unseen AUROC | Joint Gap | Gap Change |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    base_seen_list = []
    base_unseen_list = []
    base_gap_list = []
    joint_seen_list = []
    joint_unseen_list = []
    joint_gap_list = []

    for split in completed_splits:
        d = split_summaries[split]
        b = BASELINE_GMR[split]
        j_seen = d.get("seen_auroc", 0.0)
        j_unseen = d.get("unseen_auroc", 0.0)
        j_gap = d.get("seen_unseen_auroc_gap", j_seen - j_unseen)
        gap_diff = j_gap - b["gap"]
        diff_str = f"{gap_diff:+.4f}" if gap_diff != 0 else "0.0000"

        base_seen_list.append(b["seen_auroc"])
        base_unseen_list.append(b["unseen_auroc"])
        base_gap_list.append(b["gap"])
        joint_seen_list.append(j_seen)
        joint_unseen_list.append(j_unseen)
        joint_gap_list.append(j_gap)

        lines.append(
            f"| **{split}** | {b['seen_auroc']:.4f} | {b['unseen_auroc']:.4f} | {b['gap']:.4f} | "
            f"{j_seen:.4f} | {j_unseen:.4f} | **{j_gap:.4f}** | **{diff_str}** |"
        )

    if len(completed_splits) == len(SPLITS):
        avg_b_seen = sum(base_seen_list) / len(base_seen_list)
        avg_b_unseen = sum(base_unseen_list) / len(base_unseen_list)
        avg_b_gap = sum(base_gap_list) / len(base_gap_list)
        avg_j_seen = sum(joint_seen_list) / len(joint_seen_list)
        avg_j_unseen = sum(joint_unseen_list) / len(joint_unseen_list)
        avg_j_gap = sum(joint_gap_list) / len(joint_gap_list)
        avg_diff = avg_j_gap - avg_b_gap

        lines.append(
            f"| **Macro Avg** | **{avg_b_seen:.4f}** | **{avg_b_unseen:.4f}** | **{avg_b_gap:.4f}** | "
            f"**{avg_j_seen:.4f}** | **{avg_j_unseen:.4f}** | **{avg_j_gap:.4f}** | **{avg_diff:+.4f}** |"
        )

    lines.extend([
        "",
        "## 2. Localization Generalization & Existence Robustness",
        "",
        "| Split | S+ Raw R1@0.5 | U+ Raw R1@0.5 | Raw Loc Gap | U+ Soft-Gated R1@0.5 | U+ Hard-Gated R1@0.5 | U+ FRR (误拒率) | U- RR (拒识率) | Matched Pair Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for split in completed_splits:
        d = split_summaries[split]
        s_r1 = float(d.get("S+_raw_R1@0.5", d.get("raw_S+_R1@0.5", 0.0)))
        u_r1 = float(d.get("U+_raw_R1@0.5", d.get("raw_U+_R1@0.5", 0.0)))
        loc_gap = s_r1 - u_r1
        soft_u_r1 = float(d.get("U+_official_gated_R1@0.5", d.get("official_gated_U+_R1@0.5", 0.0)))
        hard_u_r1 = float(d.get("diagnostic_hard_gated_U+_R1@0.5", 0.0))
        u_frr = float(d.get("U+_FRR", 0.0)) * 100
        u_rr = float(d.get("U-_RR", 0.0)) * 100
        pair_acc = d.get("matched_pair_acc", 0.0)
        pair_str = f"{pair_acc * 100:.2f}%" if pair_acc is not None else "N/A"

        lines.append(
            f"| **{split}** | {s_r1:.2f}% | {u_r1:.2f}% | {loc_gap:.2f}% | "
            f"{soft_u_r1:.2f}% | {hard_u_r1:.2f}% | {u_frr:.2f}% | {u_rr:.2f}% | {pair_str} |"
        )

    md_content = "\n".join(lines) + "\n"
    OUTPUT_MD.write_text(md_content, encoding="utf-8")
    print(f"\n[+] Saved multi-split Markdown report to: {OUTPUT_MD}")
    print(md_content)

    # Save JSON summary as well
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(split_summaries, f, indent=2)
    print(f"[+] Saved JSON summary to: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
