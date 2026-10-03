from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("A1", "A2_alt", "A3", "C1", "C2_alt")


def main():
    rows = []
    for split in SPLITS:
        p = ROOT / "results/moment_detr_gmr_auc_v4" / split / "v4_summary.json"
        if p.is_file(): rows.append(json.loads(p.read_text()))
    if not rows: raise SystemExit("No completed split summaries found")
    means = {}
    for key in ("baseline_seen_auroc", "v4_seen_auroc", "baseline_unseen_auroc", "v4_unseen_auroc", "delta_seen_auroc", "delta_unseen_auroc"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        means[key] = sum(vals) / len(vals) if vals else None
    means["improved_split_count"] = sum(r["delta_unseen_auroc"] > 0 for r in rows)
    means["degraded_split_count"] = sum(r["delta_unseen_auroc"] < 0 for r in rows)
    means["completed_split_count"] = len(rows)
    obj = {"experiment": "Moment-DETR-GMR-Residual-AUC-v4", "status": "exploratory follow-up and not an untouched unseen evaluation",
           "macro": means, "splits": rows}
    out_json = ROOT / "experiments/moment_detr_gmr_auc_v4/multi_split_summary.json"
    out_json.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    lines = ["# GMR-AUC-v4 five-split result", "", "Exploratory follow-up; not an untouched unseen evaluation. U was excluded from training, pair construction, validation selection and threshold calibration.", "",
             "| Split | Baseline Seen AUROC | V4 Seen AUROC | ΔSeen | Baseline Unseen AUROC | V4 Unseen AUROC | ΔUnseen (paired bootstrap 95% CI) | Baseline gap | V4 gap | Baseline PairAcc | V4 PairAcc | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ci = r.get("delta_unseen_auroc_bootstrap_ci95", [None,None])
        fmt=lambda x: "NA" if x is None else f"{x:.5f}"
        lines.append(f"| {r['split']} | {fmt(r.get('baseline_seen_auroc'))} | {fmt(r.get('v4_seen_auroc'))} | {fmt(r.get('delta_seen_auroc'))} | {fmt(r.get('baseline_unseen_auroc'))} | {fmt(r.get('v4_unseen_auroc'))} | {fmt(r.get('delta_unseen_auroc'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(r.get('baseline_gap'))} | {fmt(r.get('v4_gap'))} | {fmt(r.get('baseline_matched_pair_acc'))} | {fmt(r.get('v4_matched_pair_acc'))} | {r.get('best_adapter_epoch')} |")
    lines += ["", f"Macro baseline unseen AUROC: **{means['baseline_unseen_auroc']:.5f}**", f"Macro V4 unseen AUROC: **{means['v4_unseen_auroc']:.5f}**", f"Macro Δ unseen AUROC: **{means['delta_unseen_auroc']:.5f}**", f"Macro baseline seen AUROC: **{means['baseline_seen_auroc']:.5f}**", f"Macro V4 seen AUROC: **{means['v4_seen_auroc']:.5f}**", f"Improved / degraded splits: **{means['improved_split_count']} / {means['degraded_split_count']}**", "", "See per-split `v4_summary.json`, `test_predictions.jsonl`, and bootstrap fields for full-precision detail.", ""]
    (ROOT / "experiments/moment_detr_gmr_auc_v4/MULTI_SPLIT_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
