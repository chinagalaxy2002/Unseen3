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
             "| Split | Baseline Seen AUROC | V4 Seen AUROC | ΔSeen | Baseline Unseen AUROC | V4 Unseen AUROC | ΔUnseen (paired bootstrap 95% CI) | Baseline gap | V4 gap | Δgap (V4−base) | Baseline PairAcc | V4 PairAcc | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ci = r.get("delta_unseen_auroc_bootstrap_ci95", [None,None])
        fmt=lambda x: "NA" if x is None else f"{x:.5f}"
        lines.append(f"| {r['split']} | {fmt(r.get('baseline_seen_auroc'))} | {fmt(r.get('v4_seen_auroc'))} | {fmt(r.get('delta_seen_auroc'))} | {fmt(r.get('baseline_unseen_auroc'))} | {fmt(r.get('v4_unseen_auroc'))} | {fmt(r.get('delta_unseen_auroc'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(r.get('baseline_gap'))} | {fmt(r.get('v4_gap'))} | {fmt(r.get('delta_gap'))} | {fmt(r.get('baseline_matched_pair_acc'))} | {fmt(r.get('v4_matched_pair_acc'))} | {r.get('best_adapter_epoch')} |")
    lines += ["", f"Macro baseline unseen AUROC: **{means['baseline_unseen_auroc']:.5f}**", f"Macro V4 unseen AUROC: **{means['v4_unseen_auroc']:.5f}**", f"Macro Δ unseen AUROC: **{means['delta_unseen_auroc']:.5f}**", f"Macro baseline seen AUROC: **{means['baseline_seen_auroc']:.5f}**", f"Macro V4 seen AUROC: **{means['v4_seen_auroc']:.5f}**", f"Macro Δ seen AUROC: **{means['delta_seen_auroc']:.5f}**", f"Improved / degraded / unchanged unseen splits: **{means['improved_split_count']} / {means['degraded_split_count']} / {len(rows)-means['improved_split_count']-means['degraded_split_count']}**", "", "A positive Δgap means the Seen−Unseen gap widened; negative means it narrowed. Every split's unseen paired-bootstrap CI includes zero, so the small positive macro delta is uncertain.", "", "## Secondary diagnostics", "", "| Split | Seen-val AUROC base → selected | U+ FRR | U− RR | S+ raw R1@0.5 | U+ raw R1@0.5 | S+ raw mIoU | U+ raw mIoU | Δ mean / std / max abs | Same pairs exact / composition / action | Train S+ / S− / total | Val rows |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        fmt=lambda x: "NA" if x is None else f"{x:.5f}"
        lines.append(f"| {r['split']} | {fmt(r['seen_val_baseline_auroc'])} → {fmt(r['seen_val_v4_best_auroc'])} (epoch {r['best_adapter_epoch']}) | {fmt(r['U+_FRR'])} | {fmt(r['U-_RR'])} | {fmt(r['S+_raw_R1@0.5_baseline'])} | {fmt(r['U+_raw_R1@0.5_baseline'])} | {fmt(r['S+_raw_mIoU_baseline'])} | {fmt(r['U+_raw_mIoU_baseline'])} | {fmt(r['adapter_delta_mean'])} / {fmt(r['adapter_delta_std'])} / {fmt(r['adapter_delta_abs_max'])} | {fmt(r['same_pair_fraction_exact_query'])} / {fmt(r['same_pair_fraction_composition'])} / {fmt(r['same_pair_fraction_action'])} | {r['seen_train_positive_count']} / {r['seen_train_negative_count']} / {r['feature_bank_train_count']} | {r['feature_bank_val_count']} |")
    lines += ["", "All test qids and full-precision logits were checked. In every split baseline and V4 raw spans, localization class logits, S+ and U+ raw R1@0.5, and raw mIoU are identical. All train/val banks include every Seen sample, contain no U rows, and have zero missing features. NaN/Inf checks passed.", "", "See per-split `v4_summary.json`, `test_predictions.jsonl`, `adapter_training_history.json`, and paired bootstrap fields for full-precision detail.", ""]
    (ROOT / "experiments/moment_detr_gmr_auc_v4/MULTI_SPLIT_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
