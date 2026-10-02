from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys
from os.path import basename, dirname, join

import torch
from torch.utils.data import DataLoader

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
from training.moment_detr_trm.evaluate_trm import (
    compute_mr_results_trm,
    eval_epoch_post_processing,
    setup_model_trm,
)
from models.moment_detr_trm.trm_loss import build_criterion_trm
from training.moment_detr_trm.train_trm import build_dataset_config_trm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def start_inference(opt):
    logger.info("Setup config, data and model...")
    eval_dataset = StartEndDatasetTRM(**build_dataset_config_trm(
        opt,
        opt.eval_path,
        load_labels=True,
        keep_empty_gt=False,
        partition_filter=getattr(opt, "partition_filter", None),
    ))

    model = setup_model_trm(opt)
    logger.info("Load model checkpoint from: %s", opt.model_path)
    checkpoint = torch.load(opt.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    criterion = None
    if getattr(opt, "loss_eval", False):
        criterion = build_criterion_trm(opt)
        criterion.to(opt.device)

    save_submission_filename = f"{opt.split}_preds.jsonl"
    logger.info("Starting inference on %s...", opt.eval_path)
    model.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate_trm,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results_trm(0, model, eval_loader, opt, criterion)
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission,
        opt,
        eval_dataset.data,
        save_submission_filename,
    )

    logger.info("Inference completed. Results saved to: %s", latest_file_paths)
    logger.info("Brief metrics: %s", pprint.pformat(metrics["brief"], indent=4))
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Moment-DETR-TRM checkpoint.")
    parser.add_argument("--model", "-m", default="moment_detr_trm", choices=["moment_detr", "moment_detr_trm"])
    parser.add_argument("--dataset", "-d", default="charades_sta_semantic_novelty")
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to best.ckpt")
    parser.add_argument("--split", type=str, default="test", help="Split tag for filename")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to test jsonl")
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--phrase_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--partition_filter", type=str, default=None, help="Filter partition: e.g. S+ or U+")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.model == "moment_detr_trm" or args.dataset == "charades_sta_semantic_novelty":
        option_manager = BaseOptionsTRM(args.model, args.dataset, args.feature)
    else:
        option_manager = BaseOptions(args.model, args.dataset, args.feature)
    option_manager.parse()
    opt = option_manager.option

    opt.model_path = args.model_path
    opt.split = args.split
    opt.eval_path = args.eval_path
    opt.results_dir = args.results_dir
    opt.device = args.device
    opt.eval_bsz = args.eval_bsz
    opt.num_workers = args.num_workers
    opt.partition_filter = args.partition_filter

    if args.t_feat_dir is not None:
        opt.t_feat_dir = args.t_feat_dir
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    opt.phrase_feat_dir = args.phrase_feat_dir or "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/clip_phrase"

    opt.use_phrase = True
    opt.drop_phrase = False
    opt.lambda_refine = 1.0
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_exist_head = False

    os.makedirs(opt.results_dir, exist_ok=True)
    start_inference(opt)
