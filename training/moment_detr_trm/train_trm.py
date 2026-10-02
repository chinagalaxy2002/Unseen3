from __future__ import annotations

import argparse
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

from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_trm.config import BaseOptionsTRM
from training.moment_detr_trm.dataset_trm import (
    StartEndDatasetTRM,
    prepare_batch_inputs_trm,
    start_end_collate_trm,
)
from training.moment_detr_trm.evaluate_trm import eval_epoch_trm
from models.moment_detr_trm.moment_detr_trm import build_moment_detr_trm
from models.moment_detr_trm.trm_loss import build_criterion_trm
from models.moment_detr_gmr.utils.basic_utils import (
    AverageMeter,
    rename_latest_to_best,
    save_checkpoint,
    write_log,
)
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
    logger.info("Setup Moment-DETR-TRM model and criterion...")
    model = build_moment_detr_trm(opt)
    criterion = build_criterion_trm(opt)

    device = torch.device(opt.device)
    model.to(device)
    criterion.to(device)

    # In Moment-DETR, input projections and transformer are trained with opt.lr
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_drop, gamma=0.1)

    return model, criterion, optimizer, lr_scheduler

def train_epoch_trm(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info("[Epoch %d]", epoch_i + 1)
    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)

    for batch in tqdm(train_loader, desc="Training Iteration"):
        model_inputs, targets = prepare_batch_inputs_trm(batch[1], opt.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            weight = criterion.weight_dict.get(k, 1.0)
            loss_meters[k].update(float(v) * weight)

    write_log(opt, epoch_i, loss_meters)

def train_trm(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate_trm,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    model_ema = None
    if getattr(opt, "model_ema", False):
        logger.info("Using model EMA")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    es_cnt = 0
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"

    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch_trm(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval != 0:
            continue

        with torch.no_grad():
            eval_model = model_ema.module if model_ema is not None else model
            metrics, eval_loss_meters, latest_file_paths = eval_epoch_trm(
                epoch_i,
                eval_model,
                val_dataset,
                opt,
                save_submission_filename,
                criterion,
            )

        write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
        logger.info("metrics %s", pprint.pformat(metrics["brief"], indent=4))

        # Checkpoint selection strictly based on Seen Validation MR-full-mAP
        stop_score = metrics["brief"].get("MR-full-mAP", 0)

        if stop_score > prev_best_score:
            prev_best_score = stop_score
            save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
            rename_latest_to_best(latest_file_paths)
            es_cnt = 0
            logger.info("Updated best checkpoint with score %.4f", prev_best_score)
        else:
            es_cnt += 1
            logger.info("Early stop counter: %d/%d", es_cnt, opt.max_es_cnt)
            if int(opt.max_es_cnt) >= 0 and es_cnt >= int(opt.max_es_cnt):
                logger.info("Early stopping at epoch %d. Best score %.4f", epoch_i + 1, prev_best_score)
                break

def build_dataset_config_trm(opt, data_path: str, load_labels: bool = True, keep_empty_gt: bool = False, partition_filter: str | list[str] | None = None):
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
    parser = argparse.ArgumentParser(description="Train Moment-DETR-TRM.")
    parser.add_argument("--model", "-m", default="moment_detr_trm", choices=["moment_detr", "moment_detr_trm"])
    parser.add_argument("--dataset", "-d", default="charades_sta_semantic_novelty")
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_epoch", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=None)
    parser.add_argument("--eval_bsz", type=int, default=None)
    parser.add_argument("--max_es_cnt", type=int, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--phrase_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    # TRM hyperparameters
    parser.add_argument("--lambda_refine", type=float, default=1.0)
    parser.add_argument("--lambda_con", type=float, default=1.0)
    parser.add_argument("--lambda_neg", type=float, default=0.5)
    parser.add_argument("--lambda_exc", type=float, default=1.0)
    parser.add_argument("--iou_thresh", type=float, default=0.1)
    parser.add_argument("--no_drop_phrase", action="store_true")
    parser.add_argument("--no_exist_head", action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model == "moment_detr_trm" or args.dataset == "charades_sta_semantic_novelty":
        option_manager = BaseOptionsTRM(args.model, args.dataset, args.feature, args.resume)
    else:
        option_manager = BaseOptions(args.model, args.dataset, args.feature, args.resume)
    option_manager.parse()
    opt = option_manager.option

    for name in ["lr", "seed", "n_epoch", "bsz", "eval_bsz", "max_es_cnt", "train_path", "eval_path", "t_feat_dir", "results_dir", "device"]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)

    # TRM Hyperparameters
    opt.use_phrase = True
    opt.drop_phrase = not args.no_drop_phrase
    opt.lambda_refine = args.lambda_refine
    opt.lambda_con = args.lambda_con
    opt.lambda_neg = args.lambda_neg
    opt.lambda_exc = args.lambda_exc
    opt.iou_thresh = args.iou_thresh
    opt.phrase_feat_dir = args.phrase_feat_dir or "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/clip_phrase"
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_exist_head = False

    option_manager.clean_and_makedirs(overwrite=args.overwrite)

    logger.info("Setup config, data and model...")
    set_seed(opt.seed, use_cuda=opt.device == "cuda")

    train_dataset = StartEndDatasetTRM(**build_dataset_config_trm(
        opt,
        opt.train_path,
        load_labels=True,
        keep_empty_gt=False,
        partition_filter="S+",
    ))
    val_dataset = StartEndDatasetTRM(**build_dataset_config_trm(
        opt,
        opt.eval_path,
        load_labels=True,
        keep_empty_gt=False,
        partition_filter="S+",
    ))

    model, criterion, optimizer, lr_scheduler = setup_model_and_criterion(opt)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model checkpoint: %s", args.resume)

    count_parameters(model)
    logger.info("Start training Moment-DETR-TRM...")
    train_trm(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)

if __name__ == "__main__":
    main()
