from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/moment_detr_gmr_evidence_v5"
RESULTS = ROOT / "results/moment_detr_gmr_evidence_v5"
RELEASE = Path("/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2")
BASELINES = Path("/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2")
SPLITS = ("A1", "A2_alt", "A3", "C1", "C2_alt")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path, seen_only=False):
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len({str(r["qid"]) for r in rows}) != len(rows):
        raise ValueError(f"Duplicate qids: {path}")
    if seen_only and any(not str(r["partition"]).startswith("S") for r in rows):
        raise ValueError(f"Non-Seen rows in {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def semantic(row, level):
    graph = row.get("semantic_graph") or {}
    action = str(graph.get("action_base") or graph.get("action") or "<missing>").lower().strip()
    obj = str(graph.get("object_concept") or graph.get("object") or "<no_object>").lower().strip()
    return action if level == "action" else action + "::" + obj


def pairs_for(rows, split):
    by_qid = {str(r["qid"]): r for r in rows}
    result = []
    for negative in rows:
        if int(negative["exist_label"]) != 0:
            continue
        source = str(negative.get("source_qid", ""))
        positive = by_qid.get(source)
        reason = None
        if positive is None:
            reason = "source_not_in_allowed_rows"
        elif int(positive["exist_label"]) != 1:
            reason = "source_not_positive"
        elif positive["vid"] != negative["vid"]:
            reason = "video_mismatch"
        elif " ".join(positive["query"].lower().split()) == " ".join(negative["query"].lower().split()):
            reason = "identical_query"
        elif negative.get("source_query") and negative["source_query"] != positive["query"]:
            reason = "source_query_mismatch"
        elif not str(negative.get("verification_status", "")).startswith(("user_attested", "human", "verified")):
            reason = "unsupported_verification_status"
        result.append({
            "pair_id": split + ":" + str(negative["qid"]), "split": split,
            "positive_qid": source, "negative_qid": str(negative["qid"]),
            "vid": negative["vid"], "source_qid": source,
            "construction_type": negative.get("construction_type"),
            "positive_query_hash": hashlib.sha256(positive["query"].encode()).hexdigest() if positive else None,
            "negative_query_hash": hashlib.sha256(negative["query"].encode()).hexdigest(),
            "source_windows": positive.get("relevant_windows", []) if positive else [],
            "verification_status": negative.get("verification_status"),
            "pair_eligible": reason is None, "exclusion_reason": reason,
        })
    return result
