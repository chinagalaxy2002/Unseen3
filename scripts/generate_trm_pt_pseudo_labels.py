"""
Generate Phrase-Level Pseudo Temporal Labels for TRM-PT Branch [PAPER-DERIVED-REPRODUCTION].
Empowers phrase generalization using pretrained multimodal CLIP alignments on S+ training queries.
STRICT CONSTRAINT: Strictly processes S+ training instances only. Never touches U+ or test queries.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "features" / "phrase_data" / "pt_pseudo_labels")
DEFAULT_PHRASE_DIR = str(REPO_ROOT / "features" / "phrase_data" / "clip_phrase")
DEFAULT_VID_DIR = str(REPO_ROOT / "features" / "charades_video" / "vid_clip")
DATA_DIR = str(REPO_ROOT / "data" / "release" / "semantic_existence_v2")

def find_best_temporal_window(similarity_curve: np.ndarray, top_ratio: float = 0.3) -> tuple[float, float, float]:
    """
    Given a 1D similarity curve [T] between a phrase and video frames:
    Finds the contiguous segment [st, ed] in [0, 1] maximizing the alignment score.
    Returns (st, ed, confidence).
    """
    T = len(similarity_curve)
    if T == 0:
        return 0.0, 1.0, 0.0
    if T == 1:
        return 0.0, 1.0, float(similarity_curve[0])

    # Dynamic programming: find max sum contiguous subarray of centered similarity
    mean_val = float(np.mean(similarity_curve))
    max_val = float(np.max(similarity_curve))
    thd = mean_val + (max_val - mean_val) * (1.0 - top_ratio)

    active_indices = np.where(similarity_curve >= thd)[0]
    if len(active_indices) == 0:
        peak_idx = int(np.argmax(similarity_curve))
        st = max(0, peak_idx - 1) / T
        ed = min(T, peak_idx + 2) / T
        conf = max_val
    else:
        st = float(active_indices[0]) / T
        ed = float(active_indices[-1] + 1) / T
        conf = float(np.mean(similarity_curve[active_indices]))

    # Ensure valid duration
    if ed <= st:
        ed = min(1.0, st + 1.0 / T)

    return st, ed, conf

def generate_pseudo_labels(
    phrase_dir: str,
    vid_dir: str,
    output_dir: str,
    splits: list[str],
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TRM-PT] Scanning S+ training queries across splits: {splits}...")

    s_plus_records: dict[str, dict[str, Any]] = {}
    for split in splits:
        train_path = os.path.join(DATA_DIR, split, "train.jsonl")
        if not os.path.exists(train_path):
            continue
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                # STRICT CONSTRAINT: Only S+ partition allowed
                if d.get("partition") == "S+":
                    qid = str(d["qid"])
                    if qid not in s_plus_records:
                        s_plus_records[qid] = {
                            "qid": qid,
                            "vid": d["vid"],
                            "query": d["query"],
                            "duration": float(d.get("duration", 0)),
                        }

    print(f"[TRM-PT] Total unique S+ training queries to process: {len(s_plus_records)}")

    success_count = 0
    for qid, info in tqdm(s_plus_records.items(), desc="Generating TRM-PT pseudo labels"):
        out_path = os.path.join(output_dir, f"qid{qid}.npz")
        p_path = os.path.join(phrase_dir, f"qid{qid}.npz")
        v_path = os.path.join(vid_dir, f"{info['vid']}.npz")

        if not os.path.exists(p_path) or not os.path.exists(v_path):
            continue

        p_data = np.load(p_path)
        v_data = np.load(v_path)

        phrase_feat = p_data["phrase_features"]        # [10, 16, 512]
        p_tok_mask = p_data["phrase_tokens_mask"]      # [10, 16]
        phrase_mask = p_data["phrase_mask"]            # [10]
        v_feat = v_data["features"]                    # [T, 512]

        # Compute phrase representation via masked mean over valid tokens
        valid_tok_counts = np.maximum(p_tok_mask.sum(axis=1, keepdims=True), 1.0)
        p_rep = (phrase_feat * p_tok_mask[:, :, None]).sum(axis=1) / valid_tok_counts  # [10, 512]
        p_norm = p_rep / (np.linalg.norm(p_rep, axis=-1, keepdims=True) + 1e-6)        # [10, 512]

        # Video frame normalization
        v_norm = v_feat / (np.linalg.norm(v_feat, axis=-1, keepdims=True) + 1e-6)      # [T, 512]

        # Cosine similarity matrix: [10, T]
        sim_matrix = np.dot(p_norm, v_norm.T)

        pseudo_spans = np.zeros((10, 2), dtype=np.float32)
        pseudo_conf = np.zeros((10,), dtype=np.float32)

        for p_idx in range(10):
            if phrase_mask[p_idx] > 0:
                curve = sim_matrix[p_idx]
                st, ed, conf = find_best_temporal_window(curve)
                pseudo_spans[p_idx] = [st, ed]
                pseudo_conf[p_idx] = conf

        np.savez_compressed(
            out_path,
            pseudo_spans=pseudo_spans,
            pseudo_conf=pseudo_conf,
            phrase_mask=phrase_mask,
            qid=qid,
        )
        success_count += 1

    print(f"[TRM-PT] Completed! Successfully generated {success_count} pseudo-label files in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase_dir", type=str, default=DEFAULT_PHRASE_DIR)
    parser.add_argument("--vid_dir", type=str, default=DEFAULT_VID_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["A1", "A2_alt", "A3", "C1", "C2_alt"])
    args = parser.parse_args()

    generate_pseudo_labels(args.phrase_dir, args.vid_dir, args.output_dir, args.splits)

if __name__ == "__main__":
    main()
