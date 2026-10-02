#!/usr/bin/env python3
"""
Extract phrase decomposition for Charades-STA Semantic Novelty benchmark queries.
Combines official TRM released annotations (from minghangz/TRM) with an SRL/constituent
fallback parser using spaCy, ensuring 100% query coverage without network API calls.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

import spacy

SPACY_MODEL_PATH = "/home/guoxiangyu/miniconda3/envs/owvtg/lib/python3.10/site-packages/en_core_web_sm/en_core_web_sm-3.8.0"
TRM_TRAIN_JSON = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/_external/TRM/dataset/Charades-STA/charades_train.json"
TRM_TEST_JSON = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/_external/TRM/dataset/Charades-STA/charades_test.json"
DATA_RELEASE_DIR = "/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2"
OUTPUT_METADATA_FILE = "/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/features/phrase_data/trm_phrase_metadata.jsonl"

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def build_trm_dictionary() -> Dict[str, List[str]]:
    trm_dict: Dict[str, List[str]] = {}
    for json_path in [TRM_TRAIN_JSON, TRM_TEST_JSON]:
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for vid, anno in data.items():
            sentences = anno.get("sentences", [])
            phrases_list = anno.get("phrases", [])
            for s, p in zip(sentences, phrases_list):
                norm_s = normalize_text(s)
                # Keep non-empty phrases
                valid_p = [ph.strip() for ph in p if ph.strip()]
                if norm_s and valid_p and norm_s not in trm_dict:
                    trm_dict[norm_s] = valid_p
                # Also store without trailing period
                norm_s_no_dot = norm_s.rstrip(".")
                if norm_s_no_dot and valid_p and norm_s_no_dot not in trm_dict:
                    trm_dict[norm_s_no_dot] = valid_p
    print(f"[TRM Dict] Loaded {len(trm_dict)} canonical sentence-to-phrase mappings from TRM.")
    return trm_dict

def extract_phrases_spacy(doc) -> tuple[List[str], List[str]]:
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
    trm_dict = build_trm_dictionary()

    print(f"[spaCy] Loading model from {SPACY_MODEL_PATH}...")
    nlp = spacy.load(SPACY_MODEL_PATH, disable=["ner"])

    # Collect all unique queries across splits
    all_queries: Dict[str, str] = {}
    for split in ["A1", "A2_alt", "A3", "C1", "C2_alt"]:
        split_dir = Path(DATA_RELEASE_DIR) / split
        for jsonl_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            jsonl_path = split_dir / jsonl_name
            if not jsonl_path.exists():
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    qid = str(item["qid"])
                    query = item["query"]
                    all_queries[qid] = query

    print(f"[Dataset] Total unique qids across 5 splits: {len(all_queries)}")

    # Process and write metadata
    trm_matched = 0
    spacy_matched = 0
    fallback_count = 0

    with open(OUTPUT_METADATA_FILE, "w", encoding="utf-8") as out_f:
        for qid, query in sorted(all_queries.items()):
            norm_q = normalize_text(query)
            norm_q_no_dot = norm_q.rstrip(".")

            source = None
            phrases = None
            roles = None
            fallback_status = False

            # Priority 1: Match from official TRM annotations
            if norm_q in trm_dict:
                phrases = trm_dict[norm_q][:10]
                source = "trm_official"
                trm_matched += 1
            elif norm_q_no_dot in trm_dict:
                phrases = trm_dict[norm_q_no_dot][:10]
                source = "trm_official"
                trm_matched += 1

            # Priority 2: Use spaCy SRL constituent parser
            if not phrases:
                doc = nlp(query)
                phrases, roles = extract_phrases_spacy(doc)
                phrases = phrases[:10]
                if phrases:
                    source = "spacy_srl_constituent"
                    spacy_matched += 1

            # Priority 3: Fallback to full query
            if not phrases or len(phrases) == 0:
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
                "parser_source": source,
                "parser_version": "spacy-3.8.0-en_core_web_sm" if source == "spacy_srl_constituent" else ("trm-aaai2023" if source == "trm_official" else "identity"),
                "parse_success": not fallback_status,
                "fallback_status": fallback_status,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"[Done] Phrase extraction completed -> {OUTPUT_METADATA_FILE}")
    print(f"  Total records: {len(all_queries)}")
    print(f"  TRM Official Matches: {trm_matched} ({trm_matched / len(all_queries) * 100:.2f}%)")
    print(f"  spaCy Constituent Matches: {spacy_matched} ({spacy_matched / len(all_queries) * 100:.2f}%)")
    print(f"  Fallback (Full Sentence): {fallback_count} ({fallback_count / len(all_queries) * 100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
