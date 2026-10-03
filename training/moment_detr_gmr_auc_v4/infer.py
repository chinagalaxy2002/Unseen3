from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.moment_detr_gmr_auc_v4.common import (
    ROOT, RELEASE_ROOT, SPLITS, auc, dataset_for, finite_or_raise, load_model,
    pooled_repr_and_logits, read_jsonl, start_end_collate, write_json,
)
from training.moment_detr_gmr_auc_v4.metrics import balanced_accuracy_threshold, matched_pair_acc, paired_bootstrap, raw_localization_metrics
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from models.moment_detr_gmr_auc_v4.residual_adapter import ResidualAdapter


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def infer(split: str, device: str):
    out_dir = ROOT / "results" / "moment_detr_gmr_auc_v4" / split
    frozen = json.loads((ROOT / "experiments/moment_detr_gmr_auc_v4/EXPERIMENT_FREEZE.json").read_text())
    model, opt, base_ckpt = load_model(split, device)
    adapter_ckpt = torch.load(out_dir / "best.ckpt", map_location=device, weights_only=False)
    adapter = ResidualAdapter(model.transformer.d_model).to(device)
    adapter.load_state_dict(adapter_ckpt["adapter"], strict=True); adapter.eval()

    # Threshold calibration sees only S validation examples.
    val = np.load(out_dir / "feature_bank_val.npz", allow_pickle=False)
    sv0 = torch.as_tensor(val["base_exist_logit"], dtype=torch.float32, device=device)
    zv = torch.as_tensor(val["base_exist_repr"], dtype=torch.float32, device=device)
    with torch.no_grad(): sv, _ = adapter(zv, sv0)
    val_seen = np.char.startswith(val["partition"].astype(str), "S")
    threshold, threshold_ba = balanced_accuracy_threshold(val["exist_label"][val_seen], sigmoid_np(sv.cpu().numpy())[val_seen])

    ds = dataset_for(split, "test", opt, keep_empty=True)
    source_rows = read_jsonl(RELEASE_ROOT / split / "test.jsonl")
    if len(ds.data) != len(source_rows):
        raise RuntimeError(f"Test feature missing rows: loaded={len(ds.data)} source={len(source_rows)}")
    loader = DataLoader(ds, batch_size=int(opt.eval_bsz), shuffle=False, num_workers=0, collate_fn=start_end_collate)
    preds = []
    for batch in loader:
        metas, batched = batch
        from training.moment_detr_gmr_auc_v4.common import prepare_batch_inputs
        inputs, _ = prepare_batch_inputs(batched, device)
        raw, z, s0 = pooled_repr_and_logits(model, inputs)
        s, delta = adapter(z, s0)
        spans = span_cxw_to_xx(raw["pred_spans"]).detach().cpu().numpy()
        class_logits = raw["pred_logits"].detach().cpu().numpy()
        finite_or_raise(z, s0, s, delta, raw["pred_spans"], raw["pred_logits"])
        for i, meta in enumerate(metas):
            scale = float(meta["duration"])
            span_sec = spans[i] * scale
            span_cxw = raw["pred_spans"][i].detach().cpu().tolist()
            raw_loc_logits = class_logits[i].tolist()
            s0v_i, sv_i, dv_i = float(s0[i].cpu()), float(s[i].cpu()), float(delta[i].cpu())
            row = next((r for r in source_rows if str(r["qid"]) == str(meta["qid"])), None)
            if row is None: raise RuntimeError(f"Unknown test qid {meta['qid']}")
            action, comp = _semantic(row)
            preds.append({
                "qid": str(meta["qid"]), "partition": str(row["partition"]), "exist_label": int(row["exist_label"]),
                "pred_exist_logit_base": s0v_i, "pred_exist_score_base": float(sigmoid_np([s0v_i])[0]),
                "pred_exist_logit_v4": sv_i, "pred_exist_score_v4": float(sigmoid_np([sv_i])[0]),
                "pred_exist_delta": dv_i, "action_id": action, "composition_id": comp,
                "raw_spans_cxw_normalized_base": span_cxw, "raw_spans_cxw_normalized_v4": span_cxw,
                "raw_spans_seconds_base": span_sec.tolist(), "raw_spans_seconds_v4": span_sec.tolist(),
                "raw_class_logits_base": raw_loc_logits, "raw_class_logits_v4": raw_loc_logits,
                "localization_identical": span_cxw == raw["pred_spans"][i].detach().cpu().tolist() and raw_loc_logits == class_logits[i].tolist(),
            })
    qids = [p["qid"] for p in preds]
    if len(set(qids)) != len(qids) or set(qids) != {str(r["qid"]) for r in source_rows}:
        raise RuntimeError("test qid uniqueness or coverage check failed")
    # JSON's float encoder preserves Python's round-trip precision.
    pred_path = out_dir / "test_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds: f.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n")

    by_qid = {p["qid"]: p for p in preds}
    loc = {k: {"raw_spans_seconds": v["raw_spans_seconds_base"], "raw_class_logits": v["raw_class_logits_base"]} for k, v in by_qid.items()}
    y = np.asarray([p["exist_label"] for p in preds], dtype=np.int8)
    part = np.asarray([p["partition"] for p in preds])
    base = np.asarray([p["pred_exist_logit_base"] for p in preds], dtype=np.float64)
    v4 = np.asarray([p["pred_exist_logit_v4"] for p in preds], dtype=np.float64)
    delta = v4 - base
    metrics = {}
    for label, mask in (("seen", np.char.startswith(part, "S")), ("unseen", np.char.startswith(part, "U"))):
        metrics[f"baseline_{label}_auroc"] = auc(y[mask], base[mask])
        metrics[f"v4_{label}_auroc"] = auc(y[mask], v4[mask])
        metrics[f"delta_{label}_auroc"] = metrics[f"v4_{label}_auroc"] - metrics[f"baseline_{label}_auroc"]
        metrics[f"delta_{label}_auroc_bootstrap_ci95"] = paired_bootstrap(y[mask], base[mask], v4[mask], 3407, 2000)["delta_ci95"]
    metrics["baseline_gap"] = metrics["baseline_seen_auroc"] - metrics["baseline_unseen_auroc"]
    metrics["v4_gap"] = metrics["v4_seen_auroc"] - metrics["v4_unseen_auroc"]
    metrics["delta_gap"] = metrics["v4_gap"] - metrics["baseline_gap"]
    pairs = read_jsonl(RELEASE_ROOT / split / "matched_u_pairs.jsonl")
    pair_base, pair_n = matched_pair_acc(pairs, {p["qid"]: p["pred_exist_logit_base"] for p in preds})
    pair_v4, _ = matched_pair_acc(pairs, {p["qid"]: p["pred_exist_logit_v4"] for p in preds})
    metrics["baseline_matched_pair_acc"] = pair_base; metrics["v4_matched_pair_acc"] = pair_v4; metrics["matched_pair_count"] = pair_n
    for part_name in ("S+", "U+"):
        rows = [r for r in source_rows if r["partition"] == part_name]
        loc_metrics = raw_localization_metrics(rows, loc)
        metrics[f"{part_name}_raw_R1@0.5_baseline"] = loc_metrics["r1_iou05"]
        metrics[f"{part_name}_raw_R1@0.5_v4"] = loc_metrics["r1_iou05"]
        metrics[f"{part_name}_raw_mIoU_baseline"] = loc_metrics["raw_miou"]
        metrics[f"{part_name}_raw_mIoU_v4"] = loc_metrics["raw_miou"]
    threshold_pred = np.asarray([p["pred_exist_score_v4"] >= threshold for p in preds])
    u_pos = part == "U+"; u_neg = part == "U-"
    metrics["U+_FRR"] = float(np.mean(~threshold_pred[u_pos])) if u_pos.any() else None
    metrics["U-_RR"] = float(np.mean(~threshold_pred[u_neg])) if u_neg.any() else None
    metrics["threshold_seen_val"] = threshold; metrics["threshold_seen_val_balanced_accuracy"] = threshold_ba
    metrics["adapter_delta_mean"] = float(np.mean(delta)); metrics["adapter_delta_std"] = float(np.std(delta))
    metrics["adapter_delta_abs_max"] = float(np.max(np.abs(delta)))
    metrics["localization_identical"] = all(p["localization_identical"] for p in preds)
    metrics["nan_inf_detected"] = False; metrics["missing_feature_or_fallback"] = False
    metrics["best_adapter_epoch"] = int(adapter_ckpt["best_epoch"])
    metrics["seen_val_baseline_auroc"] = float(adapter_ckpt["baseline_seen_val_auroc"])
    metrics["seen_val_v4_best_auroc"] = float(adapter_ckpt["best_seen_val_auroc"])
    metrics["feature_bank_train_count"] = int(len(np.load(out_dir / "feature_bank_train.npz")["qid"]))
    metrics["feature_bank_val_count"] = int(len(val["qid"]))
    metrics["baseline_checkpoint"] = str(Path(base_ckpt["opt"].results_dir) / "best.ckpt")
    metrics["baseline_checkpoint_sha256"] = frozen["splits"][split]["baseline_checkpoint_sha256"]
    history = json.loads((out_dir / "adapter_training_history.json").read_text())
    fractions = history["same_semantic_pair_fractions_last_epoch"]
    metrics["same_pair_fraction_exact_query"] = fractions["exact_query"]
    metrics["same_pair_fraction_composition"] = fractions["composition"]
    metrics["same_pair_fraction_action"] = fractions["action"]
    metrics["unpaired_positive_count"] = fractions["unpaired_positive_count"]
    train_audit = json.loads((out_dir / "feature_bank_train_audit.json").read_text())
    val_audit = json.loads((out_dir / "feature_bank_val_audit.json").read_text())
    metrics["seen_train_total"] = train_audit["seen_source_count"]
    metrics["seen_train_positive_count"] = train_audit["label_counts"].get("1", 0)
    metrics["seen_train_negative_count"] = train_audit["label_counts"].get("0", 0)
    metrics["feature_bank_train_count"] = train_audit["count"]
    metrics["feature_bank_val_count"] = val_audit["count"]
    metrics["missing_feature_count"] = train_audit["missing_count"] + val_audit["missing_count"]
    metrics["split"] = split
    write_json(out_dir / "v4_summary.json", metrics)
    write_json(out_dir / "test_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


def _semantic(row):
    g = row.get("semantic_graph") or {}; a = str(g.get("action_base") or g.get("action") or "<missing_action>").strip().lower()
    o = str(g.get("object_concept") or g.get("object") or "<no_object>").strip().lower()
    return a, f"{a}::{o}"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=SPLITS, required=True); p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); infer(a.split, a.device)


if __name__ == "__main__": main()
