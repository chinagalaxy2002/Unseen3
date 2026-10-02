"""
Dataset loader for Moment-DETR-TRM-PT.
Extends StartEndDatasetTRM to load phrase pseudo-temporal annotations for S+ training.
"""
from __future__ import annotations

import os
from os.path import exists, join
from typing import Dict, List, Any

import numpy as np
import torch

from models.moment_detr_gmr.utils.tensor_utils import pad_sequences_1d
from training.moment_detr_trm.dataset_trm import StartEndDatasetTRM, DEFAULT_PHRASE_DIR

DEFAULT_PT_DIR = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/pt_pseudo_labels"

class StartEndDatasetTRM_PT(StartEndDatasetTRM):
    def __init__(
        self,
        dset_name: str,
        domain: str | None,
        data_path: str,
        v_feat_dirs: list[str],
        a_feat_dirs: list[str] | None = None,
        q_feat_dir: str | None = None,
        phrase_feat_dir: str | None = None,
        pt_pseudo_dir: str | None = None,
        q_feat_type: str = "last_hidden_state",
        v_feat_types: str = "slowfast_clip",
        a_feat_types: str | None = None,
        max_q_l: int = 32,
        max_v_l: int = 75,
        max_a_l: int = 75,
        ctx_mode: str = "video_tef",
        clip_len: float = 2.0,
        max_windows: int = 8,
        span_loss_type: str = "l1",
        load_labels: bool = True,
        mr_only: bool = True,
        keep_empty_gt: bool = False,
        partition_filter: str | list[str] | None = None,
    ):
        super().__init__(
            dset_name=dset_name,
            domain=domain,
            data_path=data_path,
            v_feat_dirs=v_feat_dirs,
            a_feat_dirs=a_feat_dirs,
            q_feat_dir=q_feat_dir,
            phrase_feat_dir=phrase_feat_dir,
            q_feat_type=q_feat_type,
            v_feat_types=v_feat_types,
            a_feat_types=a_feat_types,
            max_q_l=max_q_l,
            max_v_l=max_v_l,
            max_a_l=max_a_l,
            ctx_mode=ctx_mode,
            clip_len=clip_len,
            max_windows=max_windows,
            span_loss_type=span_loss_type,
            load_labels=load_labels,
            mr_only=mr_only,
            keep_empty_gt=keep_empty_gt,
            partition_filter=partition_filter,
        )
        self.pt_pseudo_dir = pt_pseudo_dir or DEFAULT_PT_DIR

    def _get_pseudo_label_by_qid(self, qid: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pt_path = join(self.pt_pseudo_dir, f"qid{qid}.npz")
        if exists(pt_path):
            data = np.load(pt_path)
            pseudo_spans = torch.from_numpy(data["pseudo_spans"].astype(np.float32))
            pseudo_conf = torch.from_numpy(data["pseudo_conf"].astype(np.float32))
        else:
            pseudo_spans = torch.zeros((10, 2), dtype=torch.float32)
            pseudo_conf = torch.zeros((10,), dtype=torch.float32)
        return pseudo_spans, pseudo_conf

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        meta = item["meta"]
        qid = meta["qid"]

        pseudo_spans, pseudo_conf = self._get_pseudo_label_by_qid(qid)
        item["model_inputs"]["phrase_pseudo_spans"] = pseudo_spans
        item["model_inputs"]["phrase_pseudo_conf"] = pseudo_conf
        return item

def start_end_collate_trm_pt(batch: list[dict[str, Any]]):
    batch_meta, batched_data = StartEndDatasetTRM.start_end_collate_fn(batch) if hasattr(StartEndDatasetTRM, "start_end_collate_fn") else None, None
    from training.moment_detr_trm.dataset_trm import start_end_collate_trm
    batch_meta, batched_data = start_end_collate_trm(batch)
    if "phrase_pseudo_spans" in batch[0]["model_inputs"]:
        batched_data["phrase_pseudo_spans"] = torch.stack([e["model_inputs"]["phrase_pseudo_spans"] for e in batch], dim=0)
        batched_data["phrase_pseudo_conf"] = torch.stack([e["model_inputs"]["phrase_pseudo_conf"] for e in batch], dim=0)
    return batch_meta, batched_data

def prepare_batch_inputs_trm_pt(batched_model_inputs, device, non_blocking: bool = False):
    from training.moment_detr_trm.dataset_trm import prepare_batch_inputs_trm
    model_inputs, targets = prepare_batch_inputs_trm(batched_model_inputs, device, non_blocking=non_blocking)
    if targets is None:
        targets = {}
    if "phrase_pseudo_spans" in batched_model_inputs:
        targets["phrase_pseudo_spans"] = batched_model_inputs["phrase_pseudo_spans"].to(device, non_blocking=non_blocking)
        targets["phrase_pseudo_conf"] = batched_model_inputs["phrase_pseudo_conf"].to(device, non_blocking=non_blocking)
    return model_inputs, targets
