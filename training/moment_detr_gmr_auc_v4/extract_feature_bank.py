from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.moment_detr_gmr_auc_v4.common import (
    ROOT, SPLITS, dataset_for, finite_or_raise, load_model, pooled_repr_and_logits,
    read_jsonl, rows_to_bank, start_end_collate, write_json, prepare_batch_inputs,
)


@torch.no_grad()
def extract(split: str, part: str, device: str):
    model, opt, ckpt = load_model(split, device)
    ds = dataset_for(split, part, opt, keep_empty=True)
    if any(r.get("partition", "").startswith("U") for r in ds.data):
        raise RuntimeError(f"Refusing to create {part} bank containing U rows")
    loader = DataLoader(ds, batch_size=int(opt.eval_bsz), shuffle=False, num_workers=0, collate_fn=start_end_collate)
    reprs, logits = [], []
    for batch in loader:
        _, batched = batch
        inputs, _ = prepare_batch_inputs(batched, device)
        out, z, s0 = pooled_repr_and_logits(model, inputs)
        finite_or_raise(z, s0, out["pred_spans"], out["pred_logits"])
        reprs.extend(z.cpu().numpy())
        logits.extend(s0.cpu().numpy())
    source_path = Path(opt.eval_path) if part == "val" else Path(opt.train_path)
    source_rows = read_jsonl(source_path)
    seen_rows = [r for r in source_rows if str(r.get("partition", "")).startswith("S")]
    if len(ds.data) != len(reprs) or len(reprs) != len(seen_rows):
        raise RuntimeError(f"Feature bank row count {len(reprs)} != Seen {part} rows {len(ds.data)}")
    bank = rows_to_bank(ds.data, reprs, logits)
    if len(set(bank["qid"].tolist())) != len(bank["qid"]):
        raise RuntimeError(f"Duplicate qids in {split}/{part}")
    out_dir = ROOT / "results" / "moment_detr_gmr_auc_v4" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_bank_{part}.npz"
    np.savez_compressed(out_path, **bank)
    label_counts = {str(k): int(v) for k, v in zip(*np.unique(bank["exist_label"], return_counts=True))}
    part_counts = {str(k): int(v) for k, v in zip(*np.unique(bank["partition"], return_counts=True))}
    meta = {
        "split": split, "part": part, "checkpoint": str(Path(ckpt["opt"].results_dir) / "best.ckpt"),
        "count": len(bank["qid"]), "seen_source_count": len(seen_rows),
        "missing_count": len(seen_rows) - len(bank["qid"]),
        "label_counts": label_counts, "partition_counts": part_counts,
        "feature_order": list(opt.v_feat_dirs), "representation_dim": int(bank["base_exist_repr"].shape[1]),
    }
    write_json(out_dir / f"feature_bank_{part}_audit.json", meta)
    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--part", choices=("train", "val"), required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    extract(a.split, a.part, a.device)


if __name__ == "__main__":
    main()
