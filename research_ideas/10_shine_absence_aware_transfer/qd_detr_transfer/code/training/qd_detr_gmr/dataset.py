from __future__ import annotations

import json
import logging
import random
from os.path import exists, join
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.qd_detr_gmr.utils.basic_utils import load_jsonl
from models.qd_detr_gmr.utils.span_utils import span_xx_to_cxw
from models.qd_detr_gmr.utils.tensor_utils import pad_sequences_1d

logger = logging.getLogger(__name__)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def l2_normalize_np_array(np_array: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def video_id_to_feature_stem(vid: str) -> str:
    path = Path(vid)
    stem = path.stem
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    return stem


class StartEndDataset(Dataset):
    """Start-End Moment Dataset with GMR existence supervision."""

    def __init__(
        self,
        dset_name: str,
        data_path: str,
        v_feat_dirs: list[str],
        q_feat_dir: str,
        q_feat_type: str = "last_hidden_state",
        max_q_l: int = 32,
        max_v_l: int = 75,
        ctx_mode: str = "video_tef",
        clip_len: int = 2,
        max_windows: int = 8,
        span_loss_type: str = "l1",
        load_labels: bool = True,
        mr_only: bool = True,
        keep_empty_gt: bool = True,
        **kwargs,
    ):
        self.dset_name = dset_name
        self.data_path = data_path
        self.v_feat_dirs = [v_feat_dirs] if isinstance(v_feat_dirs, str) else list(v_feat_dirs)
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels
        self.mr_only = bool(mr_only)
        self.keep_empty_gt = bool(keep_empty_gt)
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)

        if self.load_labels and not self.keep_empty_gt:
            datalist = [
                d for d in datalist
                if isinstance(d.get("relevant_windows", []), list) and len(d.get("relevant_windows", [])) > 0
            ]

        missing_log_path = f"{self.data_path}.missing_features.jsonl"
        kept = []
        missing_items = []
        for d in datalist:
            qid = d.get("qid")
            vid = d.get("vid")
            stem = video_id_to_feature_stem(vid) if vid else None
            missing_paths = []

            if self.use_video:
                for feat_dir in self.v_feat_dirs:
                    if exists(join(feat_dir, f"{stem}.npz")) or exists(join(feat_dir, f"{stem}.pt")):
                        continue
                    missing_paths.append(join(feat_dir, f"{stem}.npz"))

            if qid is not None:
                if not (exists(join(self.q_feat_dir, f"qid{qid}.npz")) or exists(join(self.q_feat_dir, f"qid{qid}.pt"))):
                    missing_paths.append(join(self.q_feat_dir, f"qid{qid}.npz"))

            if missing_paths:
                missing_items.append({
                    "qid": qid,
                    "vid": vid,
                    "data_path": self.data_path,
                    "missing_paths": missing_paths,
                })
                continue
            kept.append(d)

        if missing_items:
            logger.warning(
                "Skip %d/%d examples with missing features. Log: %s",
                len(missing_items),
                len(datalist),
                missing_log_path,
            )
            try:
                with open(missing_log_path, "w", encoding="utf-8") as f:
                    for item in missing_items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("Failed to write missing feature log %s: %s", missing_log_path, exc)

        return kept

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]
        model_inputs = {
            "query_feat": self._get_query_feat_by_qid(meta["qid"]),
        }

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            windows = meta.get("relevant_windows", [])
            has_moment = isinstance(windows, list) and len(windows) > 0
            model_inputs["exist_label"] = 1.0 if has_moment else 0.0
            model_inputs["span_labels"] = self.get_span_labels(windows, ctx_l)

            if not self.mr_only and has_moment:
                pos_clip, neg_clip, score_arr = self.get_saliency_labels_sub_as_query(windows[0], ctx_l)
                model_inputs["saliency_pos_labels"] = pos_clip
                model_inputs["saliency_neg_labels"] = neg_clip
                model_inputs["saliency_all_labels"] = torch.from_numpy(score_arr).float()
            elif not self.mr_only:
                model_inputs["saliency_pos_labels"] = [0, 0]
                model_inputs["saliency_neg_labels"] = [0, 0]
                model_inputs["saliency_all_labels"] = torch.zeros(ctx_l, dtype=torch.float32)

        return {"meta": meta, "model_inputs": model_inputs}

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except ValueError:
            neg_clip_indices = pos_clip_indices

        score_array = np.zeros(ctx_l, dtype=np.float32)
        score_array[gt_st:gt_ed + 1] = 1.0
        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        if windows is None or (isinstance(windows, (list, tuple)) and len(windows) == 0):
            if self.span_loss_type == "l1":
                return torch.zeros((0, 2), dtype=torch.float32)
            if self.span_loss_type == "ce":
                return torch.zeros((0, 2), dtype=torch.long)
            raise NotImplementedError

        windows = list(windows)
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]

        if self.span_loss_type == "l1":
            windows = torch.tensor(windows, dtype=torch.float32) / (ctx_l * self.clip_len)
            windows = span_xx_to_cxw(windows)
        elif self.span_loss_type == "ce":
            windows = torch.tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows
            ], dtype=torch.long)
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        q_feat_path_npz = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat_path_pt = join(self.q_feat_dir, f"qid{qid}.pt")
        if exists(q_feat_path_npz):
            q_feat = np.load(q_feat_path_npz)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            q_feat = l2_normalize_np_array(q_feat)
            return torch.from_numpy(q_feat)
        elif exists(q_feat_path_pt):
            q_feat = safe_torch_load(q_feat_path_pt, map_location="cpu")
            if isinstance(q_feat, dict) and self.q_feat_type in q_feat:
                q_feat = q_feat[self.q_feat_type]
            if isinstance(q_feat, torch.Tensor):
                q_feat = q_feat.numpy()
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            q_feat = l2_normalize_np_array(q_feat.astype(np.float32))
            return torch.from_numpy(q_feat)
        raise FileNotFoundError(f"Feature not found for qid {qid} in {self.q_feat_dir}")

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        vid_for_path = video_id_to_feature_stem(vid)
        for feat_dir in self.v_feat_dirs:
            feat_path_npz = join(feat_dir, f"{vid_for_path}.npz")
            feat_path_pt = join(feat_dir, f"{vid_for_path}.pt")
            if exists(feat_path_npz):
                feat = np.load(feat_path_npz)["features"][:self.max_v_l].astype(np.float32)
            elif exists(feat_path_pt):
                feat = safe_torch_load(feat_path_pt, map_location="cpu")
                if isinstance(feat, dict) and "features" in feat:
                    feat = feat["features"]
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                feat = feat[:self.max_v_l].astype(np.float32)
            else:
                raise FileNotFoundError(f"Feature not found for vid {vid} in {feat_dir}")
            feat = l2_normalize_np_array(feat)
            v_feat_list.append(feat)

        min_len = min(len(e) for e in v_feat_list)
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)


def start_end_collate(batch):
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
        elif k == "saliency_all_labels":
            batched_data[k] = torch.stack([e["model_inputs"][k] for e in batch])
        else:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch],
                dtype=torch.float32,
                fixed_length=None,
            )
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = {
        "src_txt": batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        "src_txt_mask": batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        "src_vid": batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        "src_vid_mask": batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    }

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
    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    return model_inputs, targets or None
