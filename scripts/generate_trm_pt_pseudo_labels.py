"""
Generate Phrase-Level Pseudo Temporal Labels for TRM-PT Branch [PAPER-DERIVED-REPRODUCTION].
Empowers phrase generalization using pretrained multimodal CLIP alignments on S+ training queries.
STRICT PROTOCOL CONSTRAINTS:
1. Strictly processes S+ training instances only; never touches U+ or test queries.
2. Sentence GT Constraint: Phrase pseudo spans are strictly restricted within the sentence's
   ground truth interval [gt_st, gt_ed], enforcing the compositional principle of TRM-PT.
3. Contiguous Connected Component: Identifies the single best contiguous run exceeding the
   adaptive activation threshold, strictly preventing multi-peak bridging across disconnected frames.
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

def find_best_contiguous_component(
    similarity_curve: np.ndarray,
    gt_window: tuple[float, float] | None = None,
    top_ratio: float = 0.3,
) -> tuple[float, float, float]:
    """
    Finds the best contiguous temporal window [st, ed] in [0, 1] for a phrase.

    1. Sentence GT Constraint:
       Restricts search within the sentence ground truth interval [gt_st, gt_ed].
    2. Contiguous Run / Connected Component:
       Finds maximal contiguous segments of frames where similarity >= threshold.
       Selects the single best contiguous component containing the peak activation,
       strictly preventing disconnected region bridging.
    """
    T = len(similarity_curve)
    if T == 0:
        return 0.0, 1.0, 0.0
    if T == 1:
        return 0.0, 1.0, float(similarity_curve[0])

    if gt_window is not None:
        gt_st, gt_ed = gt_window
        gt_st = max(0.0, min(1.0, gt_st))
        gt_ed = max(gt_st + 1e-4, min(1.0, gt_ed))
        f_st = max(0, min(T - 1, int(np.floor(gt_st * T))))
        f_ed = max(f_st + 1, min(T, int(np.ceil(gt_ed * T))))
    else:
        gt_st, gt_ed = 0.0, 1.0
        f_st, f_ed = 0, T

    sub_curve = similarity_curve[f_st:f_ed]
    sub_T = len(sub_curve)
    if sub_T <= 1:
        return gt_st, gt_ed, float(sub_curve[0]) if sub_T == 1 else 0.0

    mean_val = float(np.mean(sub_curve))
    max_val = float(np.max(sub_curve))
    thd = mean_val + (max_val - mean_val) * (1.0 - top_ratio)

    # Find all contiguous runs (connected components) of frames >= thd
    is_active = (sub_curve >= thd)
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0

    for i, active in enumerate(is_active):
        if active and not in_run:
            in_run = True
            run_start = i
        elif not active and in_run:
            in_run = False
            runs.append((run_start, i - 1))
    if in_run:
        runs.append((run_start, sub_T - 1))

    # Fallback if no frame strictly exceeded threshold
    if not runs:
        peak_sub_idx = int(np.argmax(sub_curve))
        runs = [(peak_sub_idx, peak_sub_idx)]

    # Select the optimal contiguous run (connected component)
    peak_global_sub = int(np.argmax(sub_curve))
    best_run = runs[0]
    best_score = -1e9

    for r_st, r_ed in runs:
        run_vals = sub_curve[r_st : r_ed + 1]
        run_sum = float(np.sum(run_vals))
        # Prioritize the component containing the peak activation frame
        contains_peak = 1.0 if (r_st <= peak_global_sub <= r_ed) else 0.0
        score = run_sum + contains_peak * 10.0
        if score > best_score:
            best_score = score
            best_run = (r_st, r_ed)

    best_st, best_ed = best_run
    global_st = f_st + best_st
    global_ed = f_st + best_ed + 1  # exclusive frame index

    # Normalize to [0, 1] and clamp strictly within sentence GT interval
    min_len = min(0.005, 0.2 * (gt_ed - gt_st))
    st_raw = float(global_st) / T
    ed_raw = float(global_ed) / T

    st_norm = max(gt_st, min(gt_ed - min_len, st_raw))
    ed_norm = min(gt_ed, max(gt_st + min_len, ed_raw))
    if ed_norm <= st_norm:
        mid = 0.5 * (gt_st + gt_ed)
        st_norm = max(gt_st, mid - 0.5 * min_len)
        ed_norm = min(gt_ed, mid + 0.5 * min_len)

    conf = float(np.mean(sub_curve[best_st : best_ed + 1]))

    return float(st_norm), float(ed_norm), conf

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
                        dur = float(d.get("duration", 0))
                        rw = d.get("relevant_windows", [])
                        if rw and dur > 0:
                            gt_w = (float(rw[0][0]) / dur, float(rw[0][1]) / dur)
                        else:
                            gt_w = (0.0, 1.0)
                        s_plus_records[qid] = {
                            "qid": qid,
                            "vid": d["vid"],
                            "query": d["query"],
                            "duration": dur,
                            "gt_window": gt_w,
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

        phrase_mask = p_data["phrase_mask"]            # [10]
        v_feat = v_data["features"]                    # [T, 512]

        # Use projected CLIP phrase embeddings living in the shared visual-textual space
        if "projected_phrase_features" in p_data:
            p_rep = p_data["projected_phrase_features"]  # [10, 512]
            p_norm = p_rep / (np.linalg.norm(p_rep, axis=-1, keepdims=True) + 1e-6)
        else:
            phrase_feat = p_data["phrase_features"]
            p_tok_mask = p_data["phrase_tokens_mask"]
            valid_tok_counts = np.maximum(p_tok_mask.sum(axis=1, keepdims=True), 1.0)
            p_rep = (phrase_feat * p_tok_mask[:, :, None]).sum(axis=1) / valid_tok_counts
            p_norm = p_rep / (np.linalg.norm(p_rep, axis=-1, keepdims=True) + 1e-6)

        # Video frame normalization (vid_clip embeddings)
        v_norm = v_feat / (np.linalg.norm(v_feat, axis=-1, keepdims=True) + 1e-6)      # [T, 512]

        # Cosine similarity matrix in shared CLIP space: [10, T]
        sim_matrix = np.dot(p_norm, v_norm.T)

        pseudo_spans = np.zeros((10, 2), dtype=np.float32)
        pseudo_conf = np.zeros((10,), dtype=np.float32)

        gt_w = info.get("gt_window", (0.0, 1.0))

        for p_idx in range(10):
            if phrase_mask[p_idx] > 0:
                curve = sim_matrix[p_idx]
                st, ed, conf = find_best_contiguous_component(curve, gt_window=gt_w)
                pseudo_spans[p_idx] = [st, ed]
                pseudo_conf[p_idx] = conf

        np.savez_compressed(
            out_path,
            pseudo_spans=pseudo_spans,
            pseudo_conf=pseudo_conf,
            phrase_mask=phrase_mask,
            gt_window=np.array(gt_w, dtype=np.float32),
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
