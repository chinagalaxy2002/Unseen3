from __future__ import annotations

import json
import logging
import os
from os.path import exists, join
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.moment_detr_gmr.utils.basic_utils import load_jsonl, l2_normalize_np_array
from models.moment_detr_gmr.utils.span_utils import span_xx_to_cxw
from models.moment_detr_gmr.utils.tensor_utils import pad_sequences_1d
from training.moment_detr_gmr.dataset import StartEndDataset, video_id_to_feature_stem

logger = logging.getLogger(__name__)

DEFAULT_PHRASE_DIR = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/clip_phrase"

class StartEndDatasetTRM(StartEndDataset):
    """
    Dataset loader for Moment-DETR-TRM.
    Extends StartEndDataset to load both full query token features and phrase token features.
    """
    def __init__(
        self,
        dset_name: str,
        domain: str | None,
        data_path: str,
        v_feat_dirs: list[str],
        a_feat_dirs: list[str] | None = None,
        q_feat_dir: str | None = None,
        phrase_feat_dir: str | None = None,
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
        self.partition_filter = partition_filter
        base_dset_name = "charades_semantic_existence" if "charades" in dset_name else dset_name
        super().__init__(
            dset_name=base_dset_name,
            domain=domain,
            data_path=data_path,
            v_feat_dirs=v_feat_dirs,
            a_feat_dirs=a_feat_dirs,
            q_feat_dir=q_feat_dir,
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
        )
        self.phrase_feat_dir = phrase_feat_dir or DEFAULT_PHRASE_DIR

    def load_data(self):
        datalist = super().load_data()
        if getattr(self, "partition_filter", None) is not None:
            allowed = [self.partition_filter] if isinstance(self.partition_filter, str) else set(self.partition_filter)
            datalist = [d for d in datalist if d.get("partition") in allowed]
        return datalist

    def _get_phrase_feat_by_qid(self, qid: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        p_path = join(self.phrase_feat_dir, f"qid{qid}.npz")
        if exists(p_path):
            data = np.load(p_path)
            phrase_features = torch.from_numpy(data["phrase_features"].astype(np.float32))
            phrase_tokens_mask = torch.from_numpy(data["phrase_tokens_mask"].astype(np.float32))
            phrase_mask = torch.from_numpy(data["phrase_mask"].astype(np.float32))
            phrase_count = int(data.get("phrase_count", int(phrase_mask.sum())))
        else:
            # Fallback zero tensor
            phrase_features = torch.zeros((10, 16, 512), dtype=torch.float32)
            phrase_tokens_mask = torch.zeros((10, 16), dtype=torch.float32)
            phrase_mask = torch.zeros((10,), dtype=torch.float32)
            phrase_mask[0] = 1.0
            phrase_tokens_mask[0, 0] = 1.0
            phrase_count = 1

        return phrase_features, phrase_tokens_mask, phrase_mask, phrase_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        meta = item["meta"]
        qid = meta["qid"]

        p_feat, p_tok_mask, p_mask, p_cnt = self._get_phrase_feat_by_qid(qid)
        item["model_inputs"]["phrase_features"] = p_feat
        item["model_inputs"]["phrase_tokens_mask"] = p_tok_mask
        item["model_inputs"]["phrase_mask"] = p_mask
        item["model_inputs"]["phrase_count"] = p_cnt
        return item


def start_end_collate_trm(batch: list[dict[str, Any]]):
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = set(batch[0]["model_inputs"].keys())
    for e in batch[1:]:
        model_inputs_keys &= set(e["model_inputs"].keys())

    batched_data = {}
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
        elif k == "exist_label":
            batched_data[k] = torch.tensor([e["model_inputs"][k] for e in batch], dtype=torch.float32)
        elif k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
        elif k in ["phrase_features", "phrase_tokens_mask", "phrase_mask"]:
            batched_data[k] = torch.stack([e["model_inputs"][k] for e in batch], dim=0)
        elif k == "phrase_count":
            batched_data[k] = torch.tensor([e["model_inputs"][k] for e in batch], dtype=torch.int64)
        else:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch],
                dtype=torch.float32,
                fixed_length=None,
            )
    return batch_meta, batched_data


def prepare_batch_inputs_trm(batched_model_inputs, device, non_blocking: bool = False):
    model_inputs = {
        "src_txt": batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        "src_txt_mask": batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        "src_vid": batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        "src_vid_mask": batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    }

    if "phrase_features" in batched_model_inputs:
        model_inputs["phrase_features"] = batched_model_inputs["phrase_features"].to(device, non_blocking=non_blocking)
        model_inputs["phrase_tokens_mask"] = batched_model_inputs["phrase_tokens_mask"].to(device, non_blocking=non_blocking)
        model_inputs["phrase_mask"] = batched_model_inputs["phrase_mask"].to(device, non_blocking=non_blocking)

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "exist_label" in batched_model_inputs:
        targets["exist_label"] = batched_model_inputs["exist_label"].to(device, non_blocking=non_blocking)
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    return model_inputs, targets or None
