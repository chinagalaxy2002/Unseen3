from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "moment_detr_gmr"))
sys.path.insert(0, str(ROOT))
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate  # noqa: E402
from models.moment_detr_gmr.moment_detr import build_model  # noqa: E402
from models.moment_detr_gmr_auc_v4.residual_adapter import ResidualAdapter  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SPLITS = ("A1", "A2_alt", "A3", "C1", "C2_alt")
OLD_ROOT = Path("/home/guoxiangyu/paper/Openword/generalized-moment-retrieval")
OLD_RESULTS = OLD_ROOT / "results/semantic_existence/multi_split_v2"
RELEASE_ROOT = Path("/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2")
FEATURE_ROOT = OLD_ROOT / "features/semantic_existence_v2"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def normalize_query(query: str) -> str:
    return " ".join(str(query).lower().split())


def semantic_ids(row):
    graph = row.get("semantic_graph") or {}
    action = graph.get("action_base") or graph.get("action") or "<missing_action>"
    obj = graph.get("object_concept") or graph.get("object") or "<no_object>"
    return str(action).strip().lower(), f"{str(action).strip().lower()}::{str(obj).strip().lower()}"


def baseline_path(split: str) -> Path:
    return OLD_RESULTS / split / "moment" / "best.ckpt"


def resolved_baseline_config(checkpoint):
    raw = checkpoint["opt"]
    if isinstance(raw, dict):
        cfg = EasyDict(raw)
    else:
        cfg = EasyDict(dict(raw))
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def load_model(split: str, device: str):
    path = baseline_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical Moment-DETR-GMR checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    opt = resolved_baseline_config(checkpoint)
    opt.device = device
    model, _ = build_model(opt)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, opt, checkpoint


def dataset_for(split: str, part: str, opt, keep_empty=True):
    source = Path(opt.eval_path) if part == "val" else RELEASE_ROOT / split / f"{part}.jsonl"
    # Feature order is recovered from each original checkpoint's resolved opt.
    ds = StartEndDataset(
        dset_name=opt.dset_name,
        domain=None,
        data_path=str(source),
        v_feat_dirs=list(opt.v_feat_dirs),
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        mr_only=True,
        keep_empty_gt=keep_empty,
    )
    return ds


def pooled_repr_and_logits(model, model_inputs):
    captured = []
    def capture(_module, args):
        captured.append(args[0].detach())
    handle = model.exist_head.register_forward_pre_hook(capture)
    out = model(**model_inputs)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one original existence representation, got {len(captured)}")
    decoder_states = captured[0]
    if model.exist_pool == "mean":
        z = decoder_states.mean(dim=1)
    else:
        z = decoder_states.max(dim=1).values
    x = z
    for i, layer in enumerate(model.exist_head.layers):
        x = torch.relu(layer(x)) if i < len(model.exist_head.layers) - 1 else layer(x)
    s0_check = x.squeeze(-1)
    s0 = out["pred_exist_logits"]
    if not torch.allclose(s0, s0_check, atol=1e-7, rtol=0):
        raise RuntimeError(f"Frozen existence head failed representation replay: maxerr={(s0-s0_check).abs().max().item()}")
    return out, z, s0


def finite_or_raise(*tensors):
    for t in tensors:
        if not torch.isfinite(t).all():
            raise FloatingPointError("NaN/Inf detected")


def rows_to_bank(rows, reprs, logits):
    if len(rows) != len(reprs) or len(rows) != len(logits):
        raise ValueError("feature bank row count mismatch")
    return {
        "qid": np.asarray([str(r["qid"]) for r in rows]),
        "partition": np.asarray([str(r["partition"]) for r in rows]),
        "exist_label": np.asarray([int(r["exist_label"]) for r in rows], dtype=np.int8),
        "normalized_query": np.asarray([normalize_query(r["query"]) for r in rows]),
        "action_id": np.asarray([semantic_ids(r)[0] for r in rows]),
        "composition_id": np.asarray([semantic_ids(r)[1] for r in rows]),
        "base_exist_logit": np.asarray(logits, dtype=np.float64),
        "base_exist_repr": np.asarray(reprs, dtype=np.float32),
    }


def auc(y, scores):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def make_adapter(dim: int, device: str):
    return ResidualAdapter(dim).to(device)
