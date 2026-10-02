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
from training.moment_detr_gmr.postprocessing import PostProcessorDETR
from training.moment_detr_gmr.standalone_eval.eval import eval_submission as eval_predictions
from training.moment_detr_trm.dataset_trm import (
    StartEndDatasetTRM,
    prepare_batch_inputs_trm,
    start_end_collate_trm,
)
from models.moment_detr_trm.moment_detr_trm import build_moment_detr_trm

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
def compute_mr_results_trm(epoch_i, model, eval_loader, opt, criterion=None):
    del epoch_i
    loss_meters = defaultdict(AverageMeter)
    mr_res = []

    for batch in tqdm(eval_loader, desc="compute moment scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs_trm(batch[1], opt.device)
        outputs = model(**model_inputs)

        pred_spans = outputs["pred_spans"].cpu()
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0].cpu()  # Foreground class is index 0
        raw_scores = scores.clone()

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):
            spans = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": cur_ranked_preds,
            }
            raw_ranked_preds = torch.cat([spans, raw_scores[idx, :, None]], dim=1).tolist()
            cur_query_pred["pred_relevant_windows_pre_exist"] = [
                [float(f"{e:.4f}") for e in row]
                for row in sorted(raw_ranked_preds, key=lambda x: x[2], reverse=True)
            ]
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


def eval_epoch_trm(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate_trm,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results_trm(epoch_i, model, eval_loader, opt, criterion)
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission,
        opt,
        eval_dataset.data,
        save_submission_filename,
    )
    return metrics, eval_loss_meters, latest_file_paths


def setup_model_trm(opt):
    logger.info("Setup model...")
    model = build_moment_detr_trm(opt)
    device = torch.device(opt.device)
    model.to(device)
    return model
