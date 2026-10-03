"""
Evaluation and inference procedures for Moment-DETR-TRM-GMR-Joint-v2.
Computes raw and official soft-gated moments, pred_exist_score, and detailed candidate diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pprint
import sys
from collections import defaultdict
from os.path import basename, dirname, join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
gmr_training_path = os.path.join(REPO_ROOT, "training", "moment_detr_gmr")
if gmr_training_path not in sys.path:
    sys.path.insert(0, gmr_training_path)

from models.moment_detr_gmr.utils.basic_utils import AverageMeter, load_jsonl, save_jsonl, save_json
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from models.moment_detr_gmr.gmr_adapter import apply_existence_gate
from training.moment_detr_gmr.postprocessing import PostProcessorDETR
from training.moment_detr_gmr.standalone_eval.eval import eval_submission as eval_predictions
from training.moment_detr_trm_gmr_joint_v2.dataset import (
    StartEndDatasetJoint,
    prepare_batch_inputs_joint,
    start_end_collate_joint,
)
from models.moment_detr_trm_gmr_joint_v2.moment_detr_trm_gmr_joint import build_moment_detr_trm_gmr_joint

logger = logging.getLogger(__name__)


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    logger.info("Saving/evaluating submission")
    submission_path = join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    metrics = eval_predictions(submission, gt_data, verbose=getattr(opt, "verbose", False))
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]
    return metrics, latest_file_paths


@torch.no_grad()
def compute_mr_results_joint(epoch_i, model, eval_loader, opt, criterion=None):
    del epoch_i
    loss_meters = defaultdict(AverageMeter)
    mr_res = []

    for batch in tqdm(eval_loader, desc="compute moment scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs_joint(batch[1], opt.device)
        outputs = model(**model_inputs)

        pred_spans = outputs["pred_spans"].cpu()
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0].cpu()  # Refined foreground probability
        raw_scores = scores.clone()

        pred_exist_scores = None
        pred_exist_logits = None
        pred_exist_logits_semantic = None
        pred_exist_scores_semantic = None
        if "pred_exist_logits" in outputs:
            pred_exist_logits = outputs["pred_exist_logits"].detach().cpu()
            pred_exist_scores = torch.sigmoid(pred_exist_logits.double())
            pred_exist_logits_semantic = outputs["pred_exist_logits_semantic"].detach().cpu()
            pred_exist_scores_semantic = torch.sigmoid(pred_exist_logits_semantic.double())
            threshold = float(getattr(opt, "exist_gate_thd", 0.5))
            # Official soft gate: scales candidate scores by exist_score if below threshold
            scores = apply_existence_gate(
                scores,
                pred_exist_scores,
                threshold,
                hard=getattr(opt, "hard_exist_gate", False),
            )

        cand_support = outputs["candidate_phrase_support"].detach().cpu()
        cand_gate = outputs["candidate_phrase_gate"].detach().cpu()
        base_margin = outputs["base_margin"].detach().cpu()
        refined_margin = outputs["refined_margin"].detach().cpu()
        cand_sem = outputs["candidate_exist_evidence_semantic"].detach().cpu()
        cand_visual_res = outputs["candidate_visual_residual"].detach().cpu()
        cand_visual_support = outputs["candidate_visual_support"].detach().cpu()
        cand_fused = outputs["candidate_exist_evidence_fused"].detach().cpu()
        alpha_visual = float(outputs["alpha_visual"].detach().cpu())

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):
            duration = meta["duration"]
            spans_sec = span_cxw_to_xx(spans) * duration

            # Official soft-gated ranked predictions
            cur_ranked_preds = torch.cat([spans_sec, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            # Raw un-gated ranked predictions
            raw_ranked_preds = torch.cat([spans_sec, raw_scores[idx, :, None]], dim=1).tolist()
            raw_ranked_preds = sorted(raw_ranked_preds, key=lambda x: x[2], reverse=True)
            raw_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in raw_ranked_preds]

            top1_raw = raw_ranked_preds[0]
            top1_gated = cur_ranked_preds[0]

            # Find slot index of top-1 candidate (from raw score)
            top1_slot_idx = int(torch.argmax(raw_scores[idx]).item())

            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "partition": meta.get("partition", None),
                "pred_relevant_windows": cur_ranked_preds,
                "pred_relevant_windows_pre_exist": raw_ranked_preds,
                # Candidate-level diagnostic arrays [Q]
                "candidate_phrase_support": [float(v) for v in cand_support[idx].tolist()],
                "candidate_phrase_gate": [float(v) for v in cand_gate[idx].tolist()],
                "base_margin": [float(v) for v in base_margin[idx].tolist()],
                "refined_margin": [float(v) for v in refined_margin[idx].tolist()],
                "candidate_exist_evidence_semantic": [float(v) for v in cand_sem[idx].tolist()],
                "candidate_visual_residual": [float(v) for v in cand_visual_res[idx].tolist()],
                "candidate_visual_support": [float(v) for v in cand_visual_support[idx].tolist()],
                "candidate_exist_evidence_fused": [float(v) for v in cand_fused[idx].tolist()],
                "alpha_visual": alpha_visual,
                # Top-1 candidate diagnostics
                "top1_span": [top1_raw[0], top1_raw[1]],
                "top1_raw_score": top1_raw[2],
                "top1_official_gated_score": top1_gated[2],
                "top1_base_margin": float(f"{base_margin[idx, top1_slot_idx].item():.4f}"),
                "top1_refined_margin": float(f"{refined_margin[idx, top1_slot_idx].item():.4f}"),
                "top1_candidate_phrase_support": float(f"{cand_support[idx, top1_slot_idx].item():.4f}"),
                "top1_candidate_phrase_gate": float(f"{cand_gate[idx, top1_slot_idx].item():.4f}"),
            }

            if pred_exist_scores is not None:
                cur_query_pred["pred_exist_score"] = float(pred_exist_scores[idx].item())
                cur_query_pred["pred_exist_logit"] = float(pred_exist_logits[idx].item())
                cur_query_pred["pred_exist_score_semantic"] = float(pred_exist_scores_semantic[idx].item())
                cur_query_pred["pred_exist_logit_semantic"] = float(pred_exist_logits_semantic[idx].item())

            mr_res.append(cur_query_pred)

        if criterion is not None and targets is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length,
        min_ts_val=0,
        max_ts_val=float(getattr(opt, "max_ts_val", 200)),
        min_w_l=1,
        max_w_l=float(getattr(opt, "max_ts_val", 200)),
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )
    return post_processor(mr_res), loss_meters


def eval_epoch_joint(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate_joint,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results_joint(epoch_i, model, eval_loader, opt, criterion)
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission,
        opt,
        eval_dataset.data,
        save_submission_filename,
    )
    return metrics, eval_loss_meters, latest_file_paths


def setup_model_joint(opt):
    logger.info("Setup Moment-DETR-TRM-GMR-Joint-v2 model...")
    model = build_moment_detr_trm_gmr_joint(opt)
    pca_path = getattr(opt, "pca_path", None)
    if pca_path:
        import numpy as np
        with np.load(pca_path) as pca:
            model.set_visual_pca({k: pca[k] for k in ["mu_bg", "components_bg", "residual_mean_bg", "residual_std_bg"]})
    device = torch.device(opt.device)
    model.to(device)
    return model
