#!/usr/bin/env python3
"""
Extract phrase-level CLIP text features (last_hidden_state) using ViT-B/32.
Preserves identical information budget, tokenizer, and token normalization
as the full query baseline pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

CLIP_CODE_DIR = "/home/guoxiangyu/paper/新建文件夹/MomentofUntruth/UniVTG-NA/run_on_video"
CLIP_WEIGHTS_PATH = "/home/guoxiangyu/Beyond_Caption-Based_Queries_for_Video_Moment_Retrieval/experiments_and_data/evaluations/flash_vtg_subset_consistency_eval/models/ViT-B-32.pt"
METADATA_FILE = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/trm_phrase_metadata.jsonl"
OUTPUT_DIR = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/clip_phrase"

MAX_PHRASES = 10
MAX_PHRASE_L = 16
FEAT_DIM = 512

def l2_normalize_np_array(np_array: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    norm = np.linalg.norm(np_array, axis=-1, keepdims=True)
    return np_array / (norm + eps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=METADATA_FILE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--clip_code", type=str, default=CLIP_CODE_DIR)
    parser.add_argument("--clip_weights", type=str, default=CLIP_WEIGHTS_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clip_code not in sys.path:
        sys.path.insert(0, args.clip_code)
    import clip

    print(f"[CLIP] Loading model from {args.clip_weights} on {args.device}...")
    model, _ = clip.load(args.clip_weights, device=args.device, jit=False)
    model.eval()

    print(f"[Data] Reading metadata from {args.metadata}...")
    records = []
    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[Data] Loaded {len(records)} query records.")

    # Check already processed qids
    todo_records = []
    for r in records:
        qid = r["qid"]
        out_path = os.path.join(args.output_dir, f"qid{qid}.npz")
        if args.overwrite or not os.path.exists(out_path):
            todo_records.append(r)

    print(f"[CLIP] To encode: {len(todo_records)}/{len(records)} records.")
    if not todo_records:
        print("[CLIP] All records already processed.")
        return

    # Process in batches of queries
    # Flatten phrases in query batches to maximize GPU utilization
    bsz = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(todo_records), bsz), desc="Encoding phrases"):
            batch_records = todo_records[i : i + bsz]
            all_phrases = []
            flat_indices = []  # (record_idx, phrase_idx)

            for rec in batch_records:
                phrases = rec.get("phrases", [])[:MAX_PHRASES]
                if not phrases:
                    phrases = [rec["query"].strip()]
                rec["_phrase_feat"] = np.zeros((MAX_PHRASES, MAX_PHRASE_L, FEAT_DIM), dtype=np.float32)
                rec["_phrase_tok_mask"] = np.zeros((MAX_PHRASES, MAX_PHRASE_L), dtype=np.float32)
                rec["_phrase_mask"] = np.zeros((MAX_PHRASES,), dtype=np.float32)
                for p_idx, phrase in enumerate(phrases):
                    all_phrases.append(phrase)
                    flat_indices.append((rec, p_idx))
                    rec["_phrase_mask"][p_idx] = 1.0

            if not all_phrases:
                continue

            # Tokenize all phrases
            tokens = clip.tokenize(all_phrases, context_length=77).to(args.device)
            # Forward pass
            out = model.encode_text(tokens)
            states = out["last_hidden_state"].float().cpu().numpy()  # [N_flat, 77, 512]
            token_lengths = (tokens != 0).sum(1).cpu().numpy()

            # Fill in features from flat results
            for flat_idx, (rec, p_idx) in enumerate(flat_indices):
                length = min(int(token_lengths[flat_idx]), MAX_PHRASE_L)
                state = states[flat_idx][:length]  # [length, 512]
                state_norm = l2_normalize_np_array(state)

                rec["_phrase_feat"][p_idx, :length] = state_norm
                rec["_phrase_tok_mask"][p_idx, :length] = 1.0

            # Save each record in this batch
            for rec in batch_records:
                qid = rec["qid"]
                phrases = rec.get("phrases", [])[:MAX_PHRASES]
                if not phrases:
                    phrases = [rec["query"].strip()]
                k = len(phrases)
                out_path = os.path.join(args.output_dir, f"qid{qid}.npz")

                feat = rec.get("_phrase_feat")
                tok_mask = rec.get("_phrase_tok_mask")
                p_mask = rec.get("_phrase_mask")

                if feat is None:
                    feat = np.zeros((MAX_PHRASES, MAX_PHRASE_L, FEAT_DIM), dtype=np.float32)
                    tok_mask = np.zeros((MAX_PHRASES, MAX_PHRASE_L), dtype=np.float32)
                    p_mask = np.zeros((MAX_PHRASES,), dtype=np.float32)
                    p_mask[0] = 1.0

                np.savez_compressed(
                    out_path,
                    phrase_features=feat,
                    phrase_tokens_mask=tok_mask,
                    phrase_mask=p_mask,
                    phrase_count=np.int32(k),
                )

    print(f"[Done] All phrase CLIP features saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
