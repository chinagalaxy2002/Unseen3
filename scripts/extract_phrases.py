#!/usr/bin/env python3
"""
Extract phrase decomposition for Charades-STA Semantic Novelty benchmark queries.
Unified Parser Protocol:
- Train, val, and test queries ALL use the exact same frozen spaCy constituent parser.
- Eliminates any train/test distribution shift from mixing human annotations with parser outputs.
- Official TRM annotations (charades_train.json) are strictly preserved for audit reference only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

import spacy

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACY_MODEL_PATH = "/home/guoxiangyu/miniconda3/envs/owvtg/lib/python3.10/site-packages/en_core_web_sm/en_core_web_sm-3.8.0"
TRM_TRAIN_JSON = str(REPO_ROOT / "_external" / "TRM" / "dataset" / "Charades-STA" / "charades_train.json")
DATA_RELEASE_DIR = str(REPO_ROOT / "data" / "release" / "semantic_existence_v2")
OUTPUT_METADATA_FILE = str(REPO_ROOT / "features" / "phrase_data" / "trm_phrase_metadata.jsonl")

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def load_trm_audit_dict() -> Dict[str, List[str]]:
    """Loads official TRM train annotations strictly for audit comparison."""
    audit_dict: Dict[str, List[str]] = {}
    if os.path.exists(TRM_TRAIN_JSON):
        with open(TRM_TRAIN_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        for vid, anno in data.items():
            sentences = anno.get("sentences", [])
            phrases_list = anno.get("phrases", [])
            for s, p in zip(sentences, phrases_list):
                norm_s = normalize_text(s)
                valid_p = [ph.strip() for ph in p if ph.strip()]
                if norm_s and valid_p and norm_s not in audit_dict:
                    audit_dict[norm_s] = valid_p
    return audit_dict

def extract_phrases_spacy(doc) -> tuple[List[str], List[str]]:
    """
    Extracts semantic constituent phrases from parsed spaCy doc:
    1. Predicate verb phrases (verb + particles/adverbs/negations)
    2. Noun chunks (subjects, objects)
    3. Prepositional phrases
    """
    phrases: List[str] = []
    roles: List[str] = []

    # 1. Extract verb phrases (predicates) with verb particles and direct adverbs
    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            vp_tokens = [token.text]
            for child in token.children:
                if child.dep_ in ("prt", "advmod", "neg"):
                    vp_tokens.append(child.text)
            vp_text = " ".join(vp_tokens)
            if vp_text not in phrases:
                phrases.append(vp_text)
                roles.append("predicate")

    # 2. Extract noun chunks (subjects, objects)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text and chunk_text not in phrases:
            phrases.append(chunk_text)
            role = "subject" if chunk.root.dep_ in ("nsubj", "nsubjpass") else "object"
            roles.append(role)

    # 3. Extract prepositional phrases
    for token in doc:
        if token.dep_ == "prep":
            prep_tokens = [token.text]
            for child in token.subtree:
                if child != token and child.text not in prep_tokens:
                    prep_tokens.append(child.text)
            prep_text = " ".join(prep_tokens)
            if prep_text and prep_text not in phrases and len(prep_tokens) > 1:
                phrases.append(prep_text)
                roles.append("prepositional")

    return phrases, roles

def main():
    os.makedirs(os.path.dirname(OUTPUT_METADATA_FILE), exist_ok=True)
    trm_audit_dict = load_trm_audit_dict()
    print(f"[Audit] Loaded {len(trm_audit_dict)} TRM train annotations for distribution reference.")

    print(f"[spaCy] Loading unified frozen parser model from {SPACY_MODEL_PATH}...")
    nlp = spacy.load(SPACY_MODEL_PATH, disable=["ner"])

    # Collect all queries across splits
    all_queries: Dict[str, str] = {}
    train_qids: Set[str] = set()
    eval_qids: Set[str] = set()

    for split in ["A1", "A2_alt", "A3", "C1", "C2_alt"]:
        split_dir = Path(DATA_RELEASE_DIR) / split
        for jsonl_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            jsonl_path = split_dir / jsonl_name
            if not jsonl_path.exists():
                continue
            is_eval = (jsonl_name in ["val.jsonl", "test.jsonl"])
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    qid = str(item["qid"])
                    query = item["query"]
                    all_queries[qid] = query
                    if is_eval:
                        eval_qids.add(qid)
                    else:
                        train_qids.add(qid)

    print(f"[Dataset] Total unique qids across 5 splits: {len(all_queries)}")
    print(f"  Train qids: {len(train_qids)}")
    print(f"  Eval (Val/Test) qids: {len(eval_qids)}")

    spacy_matched = 0
    fallback_count = 0
    trm_overlap_count = 0

    with open(OUTPUT_METADATA_FILE, "w", encoding="utf-8") as out_f:
        for qid, query in sorted(all_queries.items()):
            norm_q = normalize_text(query)

            # Check overlap with TRM audit dict for documentation
            if norm_q in trm_audit_dict or norm_q.rstrip(".") in trm_audit_dict:
                trm_overlap_count += 1

            # Unified rule: 100% of all queries (train, val, test) parsed by frozen spaCy parser
            doc = nlp(query)
            phrases, roles = extract_phrases_spacy(doc)
            phrases = phrases[:10]
            fallback_status = False

            if phrases:
                source = "spacy_srl_constituent"
                spacy_matched += 1
            else:
                # Fallback to full sentence if empty
                phrases = [query.strip()]
                roles = ["full_sentence"]
                source = "fallback"
                fallback_status = True
                fallback_count += 1

            if not roles:
                roles = ["constituent"] * len(phrases)

            record = {
                "qid": qid,
                "query": query,
                "phrases": phrases,
                "phrase_roles": roles,
                "phrase_count": len(phrases),
                "is_eval_query": (qid in eval_qids),
                "parser_source": source,
                "parser_version": "spacy-3.8.0-en_core_web_sm",
                "parse_success": not fallback_status,
                "fallback_status": fallback_status,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"[Done] Unified phrase extraction completed -> {OUTPUT_METADATA_FILE}")
    print(f"  Total records: {len(all_queries)}")
    print(f"  Unified spaCy Parser Matches (100% of all splits): {spacy_matched} ({spacy_matched / len(all_queries) * 100:.2f}%)")
    print(f"  Fallback (Full Sentence): {fallback_count} ({fallback_count / len(all_queries) * 100:.2f}%)")
    print(f"  Informational TRM Train Dict Overlap: {trm_overlap_count} ({trm_overlap_count / len(all_queries) * 100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
