"""Immutable, full-precision inner-fold banks, with input-content identities."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, RESULTS, digest, read_rows, write_json

SCHEMA = "evidence_v5_decoder_bank_01"
ROLES = ("inner_train", "inner_seen_val", "inner_novel_dev")


def load_fold_model(fold, device):
    directory = RESULTS / "inner_baselines" / fold
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") != "completed" or status.get("epochs_trained") != 100:
        raise RuntimeError(f"Incomplete baseline: {fold}")
    path = directory / "best.ckpt"
    if digest(path) != status["checkpoint_hash"]:
        raise RuntimeError("Baseline checkpoint hash mismatch")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    opt = copy.deepcopy(ckpt["opt"])
    opt.device = device
    from models.moment_detr_gmr.moment_detr import build_model
    model, _ = build_model(opt)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, opt, path


def source_identity(rows, opt, cache_path):
    """Hash each unique source; cache hashes only when size and nanosecond mtime agree."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    files = set()
    for row in rows:
        for directory in opt.v_feat_dirs:
            files.add(str(Path(directory) / (str(row["vid"]) + ".npz")))
        files.add(str(Path(opt.t_feat_dir) / ("qid" + str(row["qid"]) + ".npz")))
    identities = {}
    for name in sorted(files):
        path = Path(name)
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        previous = cache.get(name)
        if previous and previous["stat"] == signature:
            checksum = previous["sha256"]
        else:
            checksum = digest(path)
        cache[name] = {"stat": signature, "sha256": checksum}
        identities[name] = checksum
    write_json(cache_path, cache)
    import hashlib
    aggregate = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return aggregate, len(identities)


@torch.no_grad()
def extract(fold, device="cuda:0"):
    manifest_path = EXPERIMENT / "inner_folds" / fold / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model, opt, checkpoint = load_fold_model(fold, device)
    sys.path.insert(0, str(ROOT / "training/moment_detr_gmr"))
    from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
    from training.moment_detr_gmr.train import build_dataset_config
    from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx

    base = RESULTS / "banks" / fold
    base.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = digest(checkpoint)
    extractor_hash = digest(Path(__file__))
    completed = {}
    for role in ROLES:
        entry = manifest["files"][role]
        if digest(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Changed fold annotations")
        rows = read_rows(entry["path"], seen_only=True)
        sources_hash, source_count = source_identity(rows, opt, base / "source_hash_cache.json")
        identity = {"schema": SCHEMA, "fold": fold, "role": role,
                    "checkpoint_hash": checkpoint_hash, "annotation_hash": entry["sha256"],
                    "manifest_hash": digest(manifest_path), "extractor_hash": extractor_hash,
                    "dataset_code_hash": digest(ROOT / "training/moment_detr_gmr/dataset.py"),
                    "feature_sources_hash": sources_hash, "feature_source_count": source_count,
                    "feature_order": list(opt.v_feat_dirs), "query_dir": opt.t_feat_dir,
                    "max_q_l": opt.max_q_l, "max_v_l": opt.max_v_l,
                    "normalization": "original per-modality per-clip/token L2",
                    "dtype": "float32", "local_roi_enabled": False}
        directory = base / role
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("identity") != identity:
                raise RuntimeError(f"Stale bank {directory}; archive it before rebuilding")
            for key, audit in meta["arrays"].items():
                array_path = directory / (key + ".npy")
                if digest(array_path) != audit["sha256"]:
                    raise RuntimeError(f"Corrupt bank array: {array_path}")
            completed[role] = meta
            print(f"{fold}/{role}: verified existing cache", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        ds = StartEndDataset(**build_dataset_config(opt, entry["path"], keep_empty_gt=True))
        if len(ds.data) != len(rows) or [str(r['qid']) for r in ds.data] != [str(r['qid']) for r in rows]:
            raise RuntimeError("Missing features or reordered dataset")
        n, q, d, max_text, text_dim = len(rows), opt.num_queries, opt.hidden_dim, opt.max_q_l, opt.t_feat_dim
        shapes = {"pooled": (n, d), "slots": (n, q, d), "base_logit": (n,),
                  "query_tokens": (n, max_text, text_dim), "query_mask": (n, max_text),
                  "video_mean": (n, opt.v_feat_dim - 2),
                  "spans_seconds": (n, q, 2), "class_logits": (n, q, 2)}
        arrays = {k: np.lib.format.open_memmap(directory / (k + ".npy"), mode="w+", dtype=np.float32, shape=v)
                  for k, v in shapes.items()}
        captured = []
        hook = model.exist_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=start_end_collate)
        offset, replay_maxerr, independent_maxerr = 0, 0.0, 0.0
        started = time.time()
        try:
            for batch_number, (metas, batch) in enumerate(loader):
                inputs, _ = prepare_batch_inputs(batch, device)
                captured.clear()
                out = model(**inputs)
                if len(captured) != 1:
                    raise RuntimeError("Unexpected decoder hook calls")
                slots = captured[0]
                pooled = slots.max(dim=1).values if opt.exist_pool == "max" else slots.mean(dim=1)
                replay = model.exist_head(slots)
                replay_maxerr = max(replay_maxerr, float((replay - out["pred_exist_logits"]).abs().max()))
                if batch_number == 0:
                    # A second independent forward, not copied arrays.
                    check = model(**inputs)
                    independent_maxerr = max(float((check[k] - out[k]).abs().max()) for k in ("pred_spans", "pred_logits"))
                    if independent_maxerr != 0:
                        raise RuntimeError("Frozen baseline independent forward mismatch")
                batch_size = len(metas)
                selection = slice(offset, offset + batch_size)
                values = {"pooled": pooled, "slots": slots,
                          "base_logit": out["pred_exist_logits"], "class_logits": out["pred_logits"]}
                spans = span_cxw_to_xx(out["pred_spans"])
                durations = torch.tensor([m["duration"] for m in metas], device=device)
                values["spans_seconds"] = spans * durations[:, None, None]
                mask = inputs["src_vid_mask"]
                # TEF does not enter the video-only feature baseline.
                values["video_mean"] = (inputs["src_vid"][..., :-2] * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
                for key, value in values.items():
                    if not torch.isfinite(value).all():
                        raise RuntimeError("Nonfinite frozen features")
                    arrays[key][selection] = value.cpu().numpy()
                text = inputs["src_txt"].cpu().numpy()
                text_mask = inputs["src_txt_mask"].cpu().numpy()
                arrays["query_tokens"][selection] = 0
                arrays["query_mask"][selection] = 0
                arrays["query_tokens"][selection, :text.shape[1]] = text
                arrays["query_mask"][selection, :text.shape[1]] = text_mask
                offset += batch_size
        finally:
            hook.remove()
        if offset != n or replay_maxerr > 1e-7:
            raise RuntimeError("Bank coverage or existence replay failure")
        for array in arrays.values():
            array.flush()
        array_audits = {k: {"shape": list(v.shape), "sha256": digest(directory / (k + ".npy"))} for k, v in arrays.items()}
        metadata_rows = [{k: r.get(k) for k in ("qid", "vid", "partition", "exist_label", "source_qid", "construction_type", "query", "duration", "relevant_windows", "semantic_graph")}
                         for r in rows]
        from training.moment_detr_gmr_evidence_v5.protocol import write_rows
        write_rows(directory / "rows.jsonl", metadata_rows)
        meta = {"identity": identity, "count": n, "arrays": array_audits,
                "rows_sha256": digest(directory / "rows.jsonl"),
                "missing_count": 0, "replay_maxerr": replay_maxerr,
                "independent_localization_forward_maxerr": independent_maxerr,
                "elapsed_seconds": time.time() - started,
                "storage_bytes": sum((directory / (k + ".npy")).stat().st_size for k in arrays),
                "label_counts": {str(label): sum(r["exist_label"] == label for r in rows) for label in (0, 1)}}
        write_json(meta_path, meta)
        completed[role] = meta
        print(f"{fold}/{role}: {n} rows, {meta['elapsed_seconds']:.1f}s, replay={replay_maxerr}", flush=True)
    write_json(EXPERIMENT / "bank_audits" / (fold + ".json"), completed)


class Bank:
    """Metadata outside model inputs; array slices loaded into an explicit device."""
    def __init__(self, fold, role):
        if role not in ROLES:
            raise ValueError(role)
        self.directory = RESULTS / "banks" / fold / role
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["identity"]["role"] != role or self.metadata["identity"]["schema"] != SCHEMA:
            raise RuntimeError("Bank role/schema mismatch")
        self.rows = read_rows(self.directory / "rows.jsonl", seen_only=True)
        if digest(self.directory / "rows.jsonl") != self.metadata["rows_sha256"]:
            raise RuntimeError("Bank metadata rows changed")
        self.arrays = {k: np.load(self.directory / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                       for k in self.metadata["arrays"]}
        for key, audit in self.metadata["arrays"].items():
            if digest(self.directory / (key + ".npy")) != audit["sha256"]:
                raise RuntimeError(f"Bank array changed: {key}")
        self.labels = np.asarray([r["exist_label"] for r in self.rows], dtype=np.float32)
        self.qid_to_index = {str(r["qid"]): i for i, r in enumerate(self.rows)}
        self.vid_to_index = {}
        for i, row in enumerate(self.rows):
            self.vid_to_index.setdefault(str(row["vid"]), i)

    def batch(self, indices, device, keys):
        result = {}
        for key in keys:
            selected = indices
            if key == "video_mean":
                selected = np.asarray([self.vid_to_index[str(self.rows[int(i)]["vid"])] for i in indices])
            result[key] = torch.as_tensor(np.array(self.arrays[key][selected], copy=True), device=device)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    extract(args.fold, args.device)
