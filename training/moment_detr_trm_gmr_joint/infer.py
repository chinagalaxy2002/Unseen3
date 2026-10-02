"""
Inference, threshold calibration, and full 4-quadrant evaluation script for Moment-DETR-TRM-GMR-Joint-v1.
Strictly implements Section 9, 10, 11, 13, 16 of the joint-v1 protocol.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
gmr_training_path = os.path.join(REPO_ROOT, "training", "moment_detr_gmr")
if gmr_training_path not in sys.path:
    sys.path.insert(0, gmr_training_path)

from training.moment_detr_trm_gmr_joint.config import BaseOptionsJoint
from training.moment_detr_trm_gmr_joint.dataset import (
    StartEndDatasetJoint,
    start_end_collate_joint,
)
from training.moment_detr_trm_gmr_joint.evaluate import (
    compute_mr_results_joint,
    setup_model_joint,
)
from models.moment_detr_gmr.utils.basic_utils import load_jsonl, save_jsonl, save_json
from scripts.analyze_semantic_existence import choose_threshold, top1_hit, iou

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def compute_iou_metrics(gt_rows, pred_dict, key="pred_relevant_windows"):
    ious = []
    for r in gt_rows:
        qid = str(r["qid"])
        if qid not in pred_dict:
            continue
        windows = pred_dict[qid].get(key, [])
        if not windows:
            ious.append(0.0)
            continue
        p = windows[0][:2]
        gt_windows = r.get("relevant_windows", [])
        if not gt_windows:
            continue
        cur_max = 0.0
        for w in gt_windows:
            inter = max(0.0, min(p[1], w[1]) - max(p[0], w[0]))
            union = (p[1] - p[0]) + (w[1] - w[0]) - inter
            iou_val = inter / union if union > 0 else 0.0
            if iou_val > cur_max:
                cur_max = iou_val
        ious.append(cur_max)

    if not ious:
        return {"count": 0, "R1@0.3": 0.0, "R1@0.5": 0.0, "R1@0.7": 0.0, "mIoU": 0.0}

    arr = np.array(ious)
    return {
        "count": len(arr),
        "R1@0.3": float(np.mean(arr >= 0.3) * 100),
        "R1@0.5": float(np.mean(arr >= 0.5) * 100),
        "R1@0.7": float(np.mean(arr >= 0.7) * 100),
        "mIoU": float(np.mean(arr) * 100),
    }


def run_full_inference(opt, model_path: str, release_dir: str):
    release = Path(release_dir)
    results_dir = Path(opt.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    val_gt = load_jsonl(str(release / "val.jsonl"))
    test_gt = load_jsonl(str(release / "test.jsonl"))
    pairs = load_jsonl(str(release / "matched_u_pairs.jsonl")) if (release / "matched_u_pairs.jsonl").exists() else []

    # 1. Step 1: Threshold Calibration on Seen Validation (S+ and S-)
    val_preds_path = results_dir / f"best_{opt.dset_name}_val_preds.jsonl"
    if not val_preds_path.exists():
        # Fallback to latest if best not renamed
        val_preds_path = results_dir / f"latest_{opt.dset_name}_val_preds.jsonl"

    if val_preds_path.exists():
        val_pred_list = load_jsonl(str(val_preds_path))
        val_pred_dict = {str(p["qid"]): p for p in val_pred_list}
        val_seen = [r for r in val_gt if r.get("partition") in ("S+", "S-")]
        frozen_threshold = choose_threshold(val_seen, val_pred_dict)
    else:
        logger.warning("No val_preds found at %s. Defaulting threshold to 0.5.", val_preds_path)
        frozen_threshold = 0.5

    training_meta_path = results_dir / "training_meta.json"
    training_meta = json.loads(training_meta_path.read_text()) if training_meta_path.exists() else {}

    threshold_info = {
        "threshold": float(frozen_threshold),
        "selection_metric": "MR-full-mAP on Seen Validation",
        "best_epoch": training_meta.get("best_epoch", -1),
        "best_seen_val_mAP": training_meta.get("best_seen_val_mAP", 0.0),
        "val_seen_n": len([r for r in val_gt if r.get("partition") in ("S+", "S-")]),
    }
    save_json(threshold_info, str(results_dir / "threshold_frozen.json"), save_pretty=True)
    logger.info("Calibrated and frozen threshold on Seen Validation: %.4f", frozen_threshold)

    # 2. Step 2: Full Test Inference
    logger.info("Loading checkpoint from %s...", model_path)
    opt.model_path = model_path
    opt.eval_path = str(release / "test.jsonl")
    opt.exist_gate_thd = frozen_threshold
    opt.keep_empty_gt = True
    opt.partition_filter = None  # Full test set: S+, S-, U+, U-

    model = setup_model_joint(opt)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_dataset = StartEndDatasetJoint(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        phrase_feat_dir=getattr(opt, "phrase_feat_dir", None),
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,
        keep_empty_gt=True,
        partition_filter=None,
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=start_end_collate_joint,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    logger.info("Computing predictions on test set (%d queries)...", len(test_dataset))
    predictions, _ = compute_mr_results_joint(0, model, test_loader, opt, criterion=None)

    # Save test_predictions.jsonl and joint_evidence_diagnostics.jsonl
    test_preds_path = results_dir / "test_predictions.jsonl"
    save_jsonl(predictions, str(test_preds_path))

    # Diagnostics file with candidate arrays
    evidence_diagnostics = []
    for p in predictions:
        evidence_diagnostics.append({
            "qid": p["qid"],
            "partition": p.get("partition"),
            "pred_exist_logit": p.get("pred_exist_logit"),
            "pred_exist_score": p.get("pred_exist_score"),
            "top1_span": p.get("top1_span"),
            "top1_raw_score": p.get("top1_raw_score"),
            "top1_official_gated_score": p.get("top1_official_gated_score"),
            "top1_base_margin": p.get("top1_base_margin"),
            "top1_refined_margin": p.get("top1_refined_margin"),
            "top1_candidate_phrase_support": p.get("top1_candidate_phrase_support"),
            "top1_candidate_phrase_gate": p.get("top1_candidate_phrase_gate"),
            "candidate_phrase_support": p.get("candidate_phrase_support"),
            "candidate_phrase_gate": p.get("candidate_phrase_gate"),
            "base_margin": p.get("base_margin"),
            "refined_margin": p.get("refined_margin"),
        })
    diagnostics_path = results_dir / "joint_evidence_diagnostics.jsonl"
    save_jsonl(evidence_diagnostics, str(diagnostics_path))
    logger.info("Saved test predictions and evidence diagnostics.")

    # 3. Step 3: Compute Official GMR and 4-Quadrant Metrics
    test_pred_dict = {str(p["qid"]): p for p in predictions}

    by_part = {}
    for part in ("S+", "S-", "U+", "U-"):
        subset = [r for r in test_gt if r.get("partition") == part]
        scores = np.array([test_pred_dict[str(r["qid"])]["pred_exist_score"] for r in subset], dtype=float)
        by_part[part] = {
            "n": len(subset),
            "mean_exist_score": float(scores.mean()) if len(scores) > 0 else 0.0,
        }
        if part.endswith("+"):
            raw_hits = np.array([top1_hit(r, test_pred_dict[str(r["qid"])], raw=True) for r in subset])
            accepted = scores >= frozen_threshold
            gated_hits = raw_hits & accepted
            by_part[part].update({
                "false_refusal": float((~accepted).mean()),
                "raw_R1_iou05": float(raw_hits.mean()),
                "gated_R1_iou05": float(gated_hits.mean()),
                "raw_correct_count": int(raw_hits.sum()),
                "gated_correct_count": int(gated_hits.sum()),
                "raw_correct_rejected": int((raw_hits & ~accepted).sum()),
            })
            if raw_hits.sum() > 0:
                by_part[part]["raw_correct_rejection_rate"] = float((raw_hits & ~accepted).sum() / raw_hits.sum())
            else:
                by_part[part]["raw_correct_rejection_rate"] = 0.0
        else:
            by_part[part]["rejection_rate"] = float((scores < frozen_threshold).mean())

    # AUROC
    auc = {}
    for prefix, pos, neg in (("seen", "S+", "S-"), ("unseen", "U+", "U-")):
        subset = [r for r in test_gt if r.get("partition") in (pos, neg)]
        labels = [r["partition"] == pos for r in subset]
        scores = [test_pred_dict[str(r["qid"])]["pred_exist_score"] for r in subset]
        auc[prefix] = float(roc_auc_score(labels, scores))

    # Matched Pair Accuracy
    pair_scores = []
    for pair in pairs:
        pos_qid = str(pair["positive_qid"])
        neg_qid = str(pair["negative_qid"])
        if pos_qid in test_pred_dict and neg_qid in test_pred_dict:
            pos_score = test_pred_dict[pos_qid]["pred_exist_score"]
            neg_score = test_pred_dict[neg_qid]["pred_exist_score"]
            pair_scores.append((pos_score > neg_score) + 0.5 * (pos_score == neg_score))

    matched_pair_acc = float(np.mean(pair_scores)) if pair_scores else None

    # Full Localization metrics (Raw & Gated)
    s_pos_rows = [r for r in test_gt if r.get("partition") == "S+"]
    u_pos_rows = [r for r in test_gt if r.get("partition") == "U+"]

    s_raw_metrics = compute_iou_metrics(s_pos_rows, test_pred_dict, key="pred_relevant_windows_pre_exist")
    u_raw_metrics = compute_iou_metrics(u_pos_rows, test_pred_dict, key="pred_relevant_windows_pre_exist")
    s_gated_metrics = compute_iou_metrics(s_pos_rows, test_pred_dict, key="pred_relevant_windows")
    u_gated_metrics = compute_iou_metrics(u_pos_rows, test_pred_dict, key="pred_relevant_windows")

    # Diagnostic Hard Gating on U+ (if exist_score < threshold, window score set to 0.0)
    u_hard_gated_hits = []
    for r in u_pos_rows:
        qid = str(r["qid"])
        item = test_pred_dict[qid]
        score = item["pred_exist_score"]
        if score >= frozen_threshold:
            hit = top1_hit(r, item, raw=True)
        else:
            hit = False
        u_hard_gated_hits.append(hit)
    diagnostic_hard_gated_u_r1_05 = float(np.mean(u_hard_gated_hits) * 100) if u_hard_gated_hits else 0.0

    diagnostics_result = {
        "threshold_source": "val_seen balanced accuracy",
        "threshold": frozen_threshold,
        "quadrants": by_part,
        "AUROC": auc,
        "AUROC_gap": auc["seen"] - auc["unseen"],
        "over_refusal_gap": by_part["U+"]["false_refusal"] - by_part["S+"]["false_refusal"],
        "matched_pair_accuracy": matched_pair_acc,
        "matched_pair_n": len(pair_scores),
        "raw_localization": {
            "S+": s_raw_metrics,
            "U+": u_raw_metrics,
        },
        "official_gated_localization": {
            "S+": s_gated_metrics,
            "U+": u_gated_metrics,
        },
        "diagnostic_hard_gate": {
            "U+_gated_R1@0.5": diagnostic_hard_gated_u_r1_05,
        }
    }
    save_json(diagnostics_result, str(results_dir / "diagnostics.json"), save_pretty=True)

    # 4. Step 4: Machine-readable Joint Summary
    joint_summary = {
        "split": "A1",
        "model": "Moment-DETR-TRM-GMR-Joint-v1",
        "seen_auroc": auc["seen"],
        "unseen_auroc": auc["unseen"],
        "auroc_gap": auc["seen"] - auc["unseen"],
        "S+_raw_R1@0.3": s_raw_metrics["R1@0.3"],
        "S+_raw_R1@0.5": s_raw_metrics["R1@0.5"],
        "S+_raw_R1@0.7": s_raw_metrics["R1@0.7"],
        "S+_raw_mIoU": s_raw_metrics["mIoU"],
        "U+_raw_R1@0.3": u_raw_metrics["R1@0.3"],
        "U+_raw_R1@0.5": u_raw_metrics["R1@0.5"],
        "U+_raw_R1@0.7": u_raw_metrics["R1@0.7"],
        "U+_raw_mIoU": u_raw_metrics["mIoU"],
        "S+_official_gated_R1@0.5": s_gated_metrics["R1@0.5"],
        "U+_official_gated_R1@0.5": u_gated_metrics["R1@0.5"],
        "U+_FRR": by_part["U+"]["false_refusal"],
        "U-_RR": by_part["U-"]["rejection_rate"],
        "raw_correct_U+_rejection_rate": by_part["U+"]["raw_correct_rejection_rate"],
        "diagnostic_hard_gated_U+_R1@0.5": diagnostic_hard_gated_u_r1_05,
        "matched_pair_acc": matched_pair_acc,
        "best_epoch": threshold_info["best_epoch"],
        "seen_val_selection_metric": threshold_info["selection_metric"],
        "seen_val_mAP": threshold_info["best_seen_val_mAP"],
        "existence_threshold": frozen_threshold,
        "counts": {
            "S+": len(s_pos_rows),
            "S-": len([r for r in test_gt if r.get("partition") == "S-"]),
            "U+": len(u_pos_rows),
            "U-": len([r for r in test_gt if r.get("partition") == "U-"]),
        }
    }
    save_json(joint_summary, str(results_dir / "joint_summary.json"), save_pretty=True)

    # 5. Run Official GMR Evaluation via eval_main.py
    official_metrics_path = results_dir / "official_test_metrics.json"
    os.system(
        f"{sys.executable} eval/eval_main.py "
        f"--submission_path {test_preds_path} "
        f"--gt_path {release / 'test.jsonl'} "
        f"--save_path {official_metrics_path} "
        f"> /dev/null 2>&1"
    )

    logger.info("Evaluation completed. Diagnostics summary:")
    print(json.dumps(diagnostics_result, indent=2))
    return diagnostics_result, joint_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Moment-DETR-TRM-GMR-Joint.")
    parser.add_argument("--model", "-m", default="moment_detr_trm_gmr_joint")
    parser.add_argument("--dataset", "-d", default="charades_sta_semantic_novelty")
    parser.add_argument("--feature", "-f", default="clip_slowfast")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--release_dir", type=str, default="data/release/semantic_existence_v2/A1")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_v_l", type=int, default=200)
    parser.add_argument("--max_ts_val", type=float, default=200.0)
    parser.add_argument("--t_feat_dir", type=str, default="features/semantic_existence_v2/shared_clip_text")
    parser.add_argument("--phrase_feat_dir", type=str, default="features/phrase_data/clip_phrase")
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=[
        "features/charades_video/vid_slowfast", "features/charades_video/vid_clip"
    ])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    option_manager = BaseOptionsJoint(args.model, args.dataset, args.feature)
    option_manager.parse()
    opt = option_manager.option

    opt.model_path = args.model_path
    opt.results_dir = args.results_dir
    opt.device = args.device
    opt.eval_bsz = args.eval_bsz
    opt.num_workers = args.num_workers
    opt.max_v_l = args.max_v_l
    opt.max_ts_val = args.max_ts_val
    opt.t_feat_dir = args.t_feat_dir
    opt.phrase_feat_dir = args.phrase_feat_dir
    opt.v_feat_dirs = args.v_feat_dirs
    opt.use_phrase = True
    opt.drop_phrase = False
    opt.lambda_refine = 1.0
    opt.phrase_scale = 10.0
    opt.mr_only = True
    opt.lw_saliency = 0

    run_full_inference(opt, args.model_path, args.release_dir)
