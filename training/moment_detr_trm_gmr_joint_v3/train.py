"""
Training script for Moment-DETR-TRM-GMR-Joint-v3.
End-to-End training on S+ and S- with Seen validation checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import json
import logging
import os
import pprint
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
gmr_training_path = os.path.join(REPO_ROOT, "training", "moment_detr_gmr")
if gmr_training_path not in sys.path:
    sys.path.insert(0, gmr_training_path)

from training.moment_detr_trm_gmr_joint_v3.config import BaseOptionsJoint
from training.moment_detr_trm_gmr_joint_v3.dataset import (
    StartEndDatasetJoint,
    prepare_batch_inputs_joint,
    start_end_collate_joint,
)
from training.moment_detr_trm_gmr_joint_v3.evaluate import eval_epoch_joint
from training.moment_detr_trm_gmr_joint_v3.semantic_groups import SemanticGroupBatchSampler, group_auroc
from models.moment_detr_trm_gmr_joint_v3.moment_detr_trm_gmr_joint import build_moment_detr_trm_gmr_joint
from models.moment_detr_trm_gmr_joint_v3.joint_loss import build_criterion_joint
from models.moment_detr_gmr.utils.basic_utils import (
    AverageMeter,
    write_log,
)
from models.moment_detr_gmr.utils.basic_utils import load_jsonl
from models.moment_detr_gmr.utils.model_utils import count_parameters, ModelEMA

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seed(seed: int, use_cuda: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def setup_model_and_criterion(opt):
    logger.info("Setup Moment-DETR-TRM-GMR-Joint-v3 model and criterion...")
    model = build_moment_detr_trm_gmr_joint(opt)
    criterion = build_criterion_joint(opt)

    device = torch.device(opt.device)
    model.to(device)
    criterion.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_drop, gamma=0.1)

    return model, criterion, optimizer, lr_scheduler


def train_epoch_joint(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info("[Epoch %d]", epoch_i + 1)
    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)
    auc_valid_batches = 0
    total_batches = 0
    ranking_stats = defaultdict(int)

    for batch in tqdm(train_loader, desc="Training Iteration"):
        model_inputs, targets = prepare_batch_inputs_joint(batch[1], opt.device)
        outputs = model(**model_inputs)
        labels = targets["exist_label"]
        auc_valid_batches += int(bool((labels > 0.5).any() and (labels <= 0.5).any()))
        total_batches += 1
        loss_dict = criterion(outputs, targets)
        for level in ("action", "composition"):
            ranking_stats[f"auc_valid_{level}_batches"] += int(loss_dict[f"auc_valid_{level}_groups"] > 0)
        for level in ("exact_query", "composition", "action"):
            ranking_stats[f"matched_{level}_pairs"] += int(loss_dict[f"matched_{level}_pairs"])
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )
        if not torch.isfinite(losses):
            raise FloatingPointError(f"NaN/Inf total loss at epoch {epoch_i + 1}")

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip, error_if_nonfinite=True)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            weight = criterion.weight_dict.get(k, 1.0)
            loss_meters[k].update(float(v) * weight)

    loss_meters["auc_valid_batch_fraction"].update(auc_valid_batches / max(total_batches, 1))
    write_log(opt, epoch_i, loss_meters)
    return auc_valid_batches, total_batches, ranking_stats


def save_checkpoint_bundle(model, optimizer, scheduler, epoch_i, opt, latest_paths, prefix, selection):
    """Persist matching weights, predictions and selection record, even below floor."""
    checkpoint_opt = copy.copy(opt)
    target = Path(opt.results_dir, prefix + ".ckpt")
    checkpoint_opt.ckpt_filepath = str(target)
    temporary = target.with_suffix(".ckpt.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(), "epoch": epoch_i,
                "opt": checkpoint_opt}, temporary)
    temporary.replace(target)
    for path in latest_paths:
        source = Path(path)
        shutil.copy2(source, source.with_name(source.name.replace("latest_", prefix + "_", 1)))
    destination = Path(opt.results_dir, prefix + "_selection.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)


def train_joint(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    sampler = SemanticGroupBatchSampler(train_dataset.data, seed=opt.seed)
    train_loader = DataLoader(train_dataset, collate_fn=start_end_collate_joint,
                              batch_sampler=sampler, num_workers=opt.num_workers)
    manifest = json.loads(Path(opt.semantic_manifest).read_text())
    if sampler.audit() != manifest["sampler_audit"]:
        raise ValueError("Feature-backed train grouping differs from frozen audit")
    localization_floor = manifest["localization_floor_mAP"]

    model_ema = None
    if getattr(opt, "model_ema", False):
        logger.info("Using model EMA")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = float("-inf")
    best_map = float("-inf")
    selected_map = None
    selected_group_metrics = None
    selected_worst_auc = None
    eligible_seen = False
    highest_map_epoch = -1
    selection_mode = None
    best_epoch = -1
    es_cnt = 0
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"
    total_auc_valid_batches = 0
    total_train_batches = 0
    total_ranking_stats = defaultdict(int)

    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        sampler.set_epoch(epoch_i)
        valid_batches, epoch_batches, ranking_stats = train_epoch_joint(model, criterion, train_loader, optimizer, opt, epoch_i)
        total_auc_valid_batches += valid_batches
        total_train_batches += epoch_batches
        for key, value in ranking_stats.items():
            total_ranking_stats[key] += value
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval != 0:
            continue

        with torch.no_grad():
            eval_model = model_ema.module if model_ema is not None else model
            metrics, eval_loss_meters, latest_file_paths = eval_epoch_joint(
                epoch_i,
                eval_model,
                val_dataset,
                opt,
                save_submission_filename,
                criterion,
            )

        val_predictions = load_jsonl(latest_file_paths[0])
        group_metrics = group_auroc(val_dataset.data, val_predictions, manifest["validation_selection_groups"])
        current_map = float(metrics["brief"]["MR-full-mAP"])
        worst_auc = float(group_metrics["worst_semantic_auroc"])
        feasible = current_map >= localization_floor
        metrics["semantic_groups"] = group_metrics
        metrics["selection"] = {"localization_floor_mAP": localization_floor,
                                 "localization_constraint_satisfied": feasible,
                                 "worst_semantic_auroc": worst_auc}
        write_log(opt, epoch_i + 1, eval_loss_meters, metrics=metrics, mode="val")
        logger.info("Seen validation: mAP=%.4f floor=%.4f worst-semantic-AUROC=%.6f eligible=%s",
                    current_map, localization_floor, worst_auc, feasible)

        # Stale mAP epochs are diagnostic only; every formal run completes 50 epochs.
        improved_map = current_map > best_map
        if improved_map:
            best_map = current_map
            highest_map_epoch = epoch_i + 1
            es_cnt = 0
            map_selection = dict(metrics["selection"], best_epoch=epoch_i + 1,
                                 best_seen_val_mAP=current_map, selection_mode="seen_mAP")
            save_checkpoint_bundle(model, optimizer, lr_scheduler, epoch_i, opt,
                                   latest_file_paths, "best_mAP", map_selection)
        else:
            es_cnt += 1

        constrained_improved = feasible and (not eligible_seen or worst_auc > prev_best_score or
                                   (worst_auc == prev_best_score and (selected_map is None or current_map > selected_map)))
        fallback_improved = not feasible and not eligible_seen and improved_map
        if constrained_improved or fallback_improved:
            if constrained_improved:
                eligible_seen = True
                prev_best_score = worst_auc
                selection_mode = "constrained_worst_semantic_auroc"
            else:
                selection_mode = "fallback_seen_mAP"
            selected_map = current_map
            selected_group_metrics = group_metrics
            selected_worst_auc = worst_auc
            best_epoch = epoch_i + 1
            selected_info = dict(metrics["selection"], best_epoch=best_epoch,
                                 best_seen_val_mAP=current_map, selection_mode=selection_mode)
            save_checkpoint_bundle(model, optimizer, lr_scheduler, epoch_i, opt,
                                   latest_file_paths, "best", selected_info)
            logger.info("Updated %s checkpoint at epoch %d, mAP %.4f, worst-AUROC %.6f",
                        selection_mode, best_epoch, current_map, worst_auc)

        meta = {
            "best_epoch": best_epoch, "best_seen_val_mAP": selected_map,
            "best_seen_worst_semantic_auroc": selected_worst_auc,
            "best_seen_group_metrics": selected_group_metrics,
            "highest_seen_val_mAP": best_map,
            "highest_seen_val_mAP_epoch": highest_map_epoch,
            "selection_mode": selection_mode,
            "localization_constraint_satisfied": eligible_seen,
            "fallback_checkpoint_used": not eligible_seen,
            "always_saved_mAP_checkpoint": "best_mAP.ckpt",
            "localization_reference_mAP": manifest["localization_reference_mAP"],
            "localization_floor_mAP": localization_floor,
            "selection_metric": "Seen-only constrained worst-semantic AUROC; Seen-mAP fallback if no eligible epoch",
            "epochs_trained": epoch_i + 1, "max_epochs": opt.n_epoch,
            "max_es_cnt": opt.max_es_cnt, "mAP_stale_epochs": es_cnt,
            "training_status": "completed" if epoch_i+1==opt.n_epoch else "running",
            "early_stopping_enabled": False,
            "selection_status": "eligible_checkpoint_available" if eligible_seen else "fallback_checkpoint_available",
            "auc_valid_batch_fraction": total_auc_valid_batches / max(total_train_batches, 1),
            "nan_inf_detected": False, "train_dataset_queries": len(train_dataset),
            "val_dataset_queries": len(val_dataset), "missing_feature_or_fallback": False,
            "auc_valid_action_batch_fraction": total_ranking_stats["auc_valid_action_batches"] / max(total_train_batches,1),
            "auc_valid_composition_batch_fraction": total_ranking_stats["auc_valid_composition_batches"] / max(total_train_batches,1),
            "matched_pair_counts": {k:v for k,v in total_ranking_stats.items() if k.startswith("matched_")},
            "sampler": "4 semantic groups x (2 positive + 2 negative); alternating action/composition",
        }
        path = Path(opt.results_dir, "training_meta.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    if best_epoch < 0:
        raise RuntimeError("No checkpoint saved despite completed training")
    if not eligible_seen:
        logger.warning("Localization floor unmet; using explicitly labelled Seen-mAP best fallback for evaluation")


def build_dataset_config_joint(
    opt,
    data_path: str,
    load_labels: bool = True,
    keep_empty_gt: bool = True,
    partition_filter: str | list[str] | None = None,
):
    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        phrase_feat_dir=getattr(opt, "phrase_feat_dir", None),
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
        mr_only=bool(getattr(opt, "mr_only", True)),
        keep_empty_gt=keep_empty_gt,
        partition_filter=partition_filter,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Moment-DETR-TRM-GMR-Joint-v3.")
    parser.add_argument("--model", "-m", default="moment_detr_trm_gmr_joint_v3")
    parser.add_argument("--dataset", "-d", default="charades_sta_semantic_novelty")
    parser.add_argument("--feature", "-f", default="clip_slowfast")
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--max_es_cnt", type=int, default=-1)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--phrase_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr_drop", type=int, default=400)
    parser.add_argument("--max_v_l", type=int, default=200)
    parser.add_argument("--max_ts_val", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    # Loss coefficients (frozen per protocol)
    parser.add_argument("--lambda_refine", type=float, default=1.0)
    parser.add_argument("--phrase_scale", type=float, default=10.0)
    parser.add_argument("--lambda_con", type=float, default=1.0)
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--lambda_exc", type=float, default=1.0)
    parser.add_argument("--exist_loss_coef", type=float, default=1.0)
    parser.add_argument("--lambda_robust", type=float, default=1.0)
    parser.add_argument("--lambda_matched", type=float, default=1.0)
    parser.add_argument("--robust_tau", type=float, default=0.1)
    parser.add_argument("--semantic_manifest", type=str, required=True)
    parser.add_argument("--auc_margin", type=float, default=1.0)
    parser.add_argument("--iou_thresh", type=float, default=0.1)
    parser.add_argument("--exist_gate_thd", type=float, default=0.5)
    parser.add_argument("--no_drop_phrase", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    option_manager = BaseOptionsJoint(args.model, args.dataset, args.feature, args.resume)
    option_manager.parse()
    opt = option_manager.option

    for name in [
        "lr", "seed", "n_epoch", "bsz", "eval_bsz", "max_es_cnt", "lr_drop",
        "max_v_l", "max_ts_val", "train_path", "eval_path", "t_feat_dir",
        "results_dir", "device"
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)

    if opt.n_epoch != 50 or opt.max_es_cnt != -1:
        raise ValueError("Formal Joint-v3 restart requires exactly 50 epochs with early stopping disabled")

    # TRM & Joint Hyperparameters
    opt.use_phrase = True
    opt.drop_phrase = not args.no_drop_phrase
    opt.lambda_refine = args.lambda_refine
    opt.phrase_scale = args.phrase_scale
    opt.lambda_con = args.lambda_con
    opt.lambda_neg = args.lambda_neg
    opt.lambda_exc = args.lambda_exc
    opt.exist_loss_coef = args.exist_loss_coef
    opt.lambda_robust = args.lambda_robust
    opt.lambda_matched = args.lambda_matched
    opt.robust_tau = args.robust_tau
    opt.semantic_manifest = args.semantic_manifest
    opt.auc_margin = args.auc_margin
    opt.iou_thresh = args.iou_thresh
    opt.exist_gate_thd = 0.0  # raw localization evaluation during Seen-only selection
    opt.phrase_feat_dir = args.phrase_feat_dir or "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/clip_phrase"
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.keep_empty_gt = True

    if Path(opt.results_dir, "train.log").exists():
        raise FileExistsError("Existing Joint-v3 training log: refusing to overwrite a run")
    option_manager.clean_and_makedirs(overwrite=args.overwrite)

    # Save resolved configuration
    resolved_config_path = os.path.join(opt.results_dir, "resolved_config.json")
    with open(resolved_config_path, "w") as f:
        json.dump(dict(opt), f, indent=2)

    logger.info("Setup config, data and model...")
    set_seed(opt.seed, use_cuda=opt.device == "cuda")

    # Training strictly on S+ and S- (Seen data only)
    train_dataset = StartEndDatasetJoint(**build_dataset_config_joint(
        opt,
        opt.train_path,
        load_labels=True,
        keep_empty_gt=True,
        partition_filter=["S+", "S-"],
    ))
    # Validation strictly on S+ and S- (Seen validation only)
    val_dataset = StartEndDatasetJoint(**build_dataset_config_joint(
        opt,
        opt.eval_path,
        load_labels=True,
        keep_empty_gt=True,
        partition_filter=["S+", "S-"],
    ))
    expected_train = sum(r.get("partition") in ("S+", "S-") for r in load_jsonl(opt.train_path))
    expected_val = sum(r.get("partition") in ("S+", "S-") for r in load_jsonl(opt.eval_path))
    if len(train_dataset) != expected_train or len(val_dataset) != expected_val:
        raise FileNotFoundError(
            f"Feature-backed query count mismatch: train {len(train_dataset)}/{expected_train}, "
            f"val {len(val_dataset)}/{expected_val}; missing visual feature fallback is forbidden"
        )

    model, criterion, optimizer, lr_scheduler = setup_model_and_criterion(opt)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model checkpoint: %s", args.resume)

    count_parameters(model)
    logger.info("Start training Moment-DETR-TRM-GMR-Joint-v3...")
    train_joint(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)


if __name__ == "__main__":
    main()
