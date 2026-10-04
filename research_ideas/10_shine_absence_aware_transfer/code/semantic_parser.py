#!/usr/bin/env python3
"""Build train-only semantic holdouts and reviewable Charades-STA counterfactuals.

The program deliberately does not certify absence from incomplete annotations.
Negative rows go to a candidate file and require video review before benchmark use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from nltk.corpus import wordnet as wn


ROOT = Path(__file__).resolve().parents[1]
SKIP_VERBS = {"be", "start", "begin", "continue", "go", "get", "have", "do", "seem", "try"}
ACTION_ALIASES = {"shut": "close"}
PRONOUNS = {"it", "them", "something", "anything", "one", "some", "this", "that", "him", "her"}
ACTION_NEIGHBORS = {
    "open": ["close"], "close": ["open"],
    "take_off": ["put_on"], "put_on": ["take_off"],
    "take_out": ["put_in"], "put_in": ["take_out"],
    "put_down": ["pick_up"], "pick_up": ["put_down"],
    "hold": ["throw"], "throw": ["hold"],
    "drink": ["pour"], "pour": ["drink"],
    "dress": ["undress"], "undress": ["dress"],
    "enter": ["leave"], "leave": ["enter"],
}
CLOTHING = {"shoe", "sock", "boot", "coat", "jacket", "shirt", "hat", "clothe", "dress", "sweater", "pants"}
OBJECT_NEIGHBORS = {
    "door": ["cabinet", "closet"], "cabinet": ["door", "refrigerator"],
    "refrigerator": ["cabinet", "closet"], "closet": ["cabinet", "door"],
    "cup": ["glass", "bottle"], "glass": ["cup", "bottle"], "bottle": ["cup", "glass"],
    "book": ["paper", "picture"], "picture": ["book", "paper"],
    "shoe": ["sock", "boot"], "bag": ["box", "basket"],
    "phone": ["remote", "camera"], "pillow": ["blanket", "cushion"],
    "chair": ["couch", "bed", "sofa"], "couch": ["chair", "bed", "sofa"],
    "bed": ["chair", "couch", "sofa"], "sofa": ["chair", "couch", "bed"],
}


def stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def vn_index(vn_dir: Path) -> dict[str, list[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    if not vn_dir.exists():
        return {}
    for path in sorted(vn_dir.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
            for member in root.findall("./MEMBERS/MEMBER"):
                index[member.attrib["name"].lower()].add(root.attrib["ID"])
        except (ET.ParseError, KeyError):
            continue
    return {k: sorted(v) for k, v in index.items()}


def wn_id(lemma: str, pos: str) -> str | None:
    try:
        senses = wn.synsets(lemma, pos=pos)
        return senses[0].name() if senses else None
    except LookupError:
        return None


def wordnet_distance(left: str | None, right: str | None) -> float | None:
    if not left or not right:
        return None
    try:
        similarity = wn.synset(left).path_similarity(wn.synset(right))
        return round(1 - similarity, 4) if similarity is not None else None
    except (LookupError, ValueError):
        return None


def content_verb(doc):
    verbs = [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() not in SKIP_VERBS]
    return verbs[0] if verbs else None


def predicate_components(verb):
    if verb is None:
        return None, None, None
    base = ACTION_ALIASES.get(verb.lemma_.lower(), verb.lemma_.lower())
    particle = None
    if base in {"take", "put", "pick"}:
        candidates = [t for t in verb.children if t.dep_ == "prt" and t.lower_ in {"off", "on", "out", "in", "down", "up"}]
        if not candidates:
            candidates = [t for t in verb.children if t.dep_ == "prep" and (
                t.lower_ in {"out", "in"} or (
                    t.lower_ in {"off", "on"} and any(c.lemma_.lower() in CLOTHING for c in t.children if c.dep_ == "pobj")
                ))]
        particle = candidates[0] if candidates else None
    action = base + "_" + particle.lower_ if particle else base
    return action, base, particle


def object_token(verb):
    direct = [t for t in verb.children if t.dep_ in {"dobj", "obj", "attr", "oprd"} and t.pos_ in {"NOUN", "PROPN"}]
    if direct:
        return direct[0]
    for prep in verb.children:
        if prep.dep_ in {"prep", "prt", "agent"}:
            nouns = [t for t in prep.children if t.dep_ == "pobj" and t.pos_ in {"NOUN", "PROPN"}]
            if nouns:
                return nouns[0]
    return None


def canonicalize(doc, vn: dict[str, list[str]]) -> dict:
    verb = content_verb(doc)
    obj = object_token(verb) if verb is not None else None
    action, base_action, particle = predicate_components(verb)
    noun = obj.lemma_.lower() if obj is not None else None
    if noun in PRONOUNS:
        noun = None
    rel = None
    object_role = None
    if verb is not None and obj is not None:
        prep = obj.head if obj.head.dep_ == "prep" else None
        rel = prep.lemma_.lower() if prep is not None else None
        object_role = "theme" if obj.dep_ in {"dobj", "obj", "attr", "oprd"} else ("source" if rel == "from" else "location")
    return {
        "agent": "person" if any(t.lemma_.lower() in {"person", "man", "woman", "someone", "somebody"} for t in doc) else None,
        "action": action, "action_base": base_action, "action_class": action,
        "verbnet_classes": vn.get(base_action or "", []),
        "action_wordnet_sense": wn_id(action, wn.VERB) if action else None,
        "object": noun, "object_concept": noun, "object_wordnet_sense": wn_id(noun, wn.NOUN) if noun else None,
        "relation": rel, "object_role": object_role,
        "location": noun if object_role == "location" else None,
        "temporal_relation": None,
        "canonicalization_confidence": "high" if action and noun else "low",
        "action_span": [verb.idx, verb.idx + len(verb)] if verb is not None else None,
        "particle_span": [particle.idx, particle.idx + len(particle)] if particle is not None else None,
        "object_span": [obj.idx, obj.idx + len(obj)] if obj is not None and noun else None,
    }


def pair(graph: dict) -> tuple[str, str] | None:
    if graph["action"] and graph["object"]:
        return graph["action"], graph["object"]
    return None


def pair_id(p: tuple[str, str]) -> str:
    return "|".join(p)


def load_split_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1 or spec.get("axis") not in {"action", "composition"}:
        raise ValueError(f"Invalid split spec schema or axis: {path}")
    actions = spec.get("held_actions")
    compositions = spec.get("held_compositions")
    if not isinstance(actions, list) or not isinstance(compositions, list):
        raise ValueError("Split spec requires held_actions and held_compositions lists")
    if len(actions) != len(set(actions)) or len(compositions) != len(set(compositions)):
        raise ValueError("Duplicate held semantics in split spec")
    if any(not isinstance(a, str) or not a for a in actions):
        raise ValueError("Invalid held action")
    if any(not isinstance(p, str) or p.count("|") != 1 or not all(p.split("|")) for p in compositions):
        raise ValueError("Invalid held composition")
    if spec["axis"] == "action" and (not actions or compositions):
        raise ValueError("Action split must hold actions only")
    if spec["axis"] == "composition" and (actions or not compositions):
        raise ValueError("Composition split must hold compositions only")
    return spec


def contains_held_semantics(row: dict, held_actions: set[str], held_pair_ids: set[str]) -> bool:
    event_pairs = set(row["_pairs"])
    event_actions = {p.split("|", 1)[0] for p in event_pairs}
    return bool(set(row["_verbs"]) & held_actions or event_actions & held_actions or event_pairs & held_pair_ids)


def read_charades_actions(metadata_dir: Path, nlp, vn) -> dict[str, set[tuple[str, str]]]:
    classes_path = metadata_dir / "Charades_v1_classes.txt"
    classes = {}
    if classes_path.exists():
        for line in classes_path.read_text().splitlines():
            code, _, label = line.partition(" ")
            graph = canonicalize(nlp(label), vn)
            if pair(graph):
                classes[code] = pair(graph)
    video_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for name in ("Charades_v1_train.csv", "Charades_v1_test.csv"):
        path = metadata_dir / name
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                for action in row.get("actions", "").split(";"):
                    code = action.split(" ")[0]
                    if code in classes:
                        video_pairs[row["id"]].add(classes[code])
    return video_pairs


def read_action_genome(path: Path | None, nlp) -> dict[str, set[tuple[str, str]]]:
    evidence: dict[str, set[tuple[str, str]]] = defaultdict(set)
    if path is None or not path.exists():
        return evidence
    relation_to_action = {
        "holding": "hold", "carrying": "carry", "drinking_from": "drink",
        "eating": "eat", "sitting_on": "sit", "standing_on": "stand",
        "wearing": "wear", "touching": "touch", "leaning_on": "lean",
        "lying_on": "lie", "wiping": "wipe", "writing_on": "write",
        "twisting": "twist",
    }
    object_cache: dict[str, list[str]] = {}
    relation_cache: dict[str, str | None] = {}
    with path.open("rb") as f:
        annotation = pickle.load(f, encoding="latin1")
    for frame_key, objects in annotation.items():
        vid = frame_key.split("/")[0].removesuffix(".mp4")
        for obj in objects:
            raw_class = str(obj.get("class", ""))
            if raw_class not in object_cache:
                aliases = []
                for part in raw_class.split("/"):
                    noun_doc = nlp(part.replace("_", " "))
                    noun = next((t.lemma_.lower() for t in reversed(noun_doc) if t.pos_ in {"NOUN", "PROPN"}), None)
                    if noun:
                        aliases.append(noun)
                object_cache[raw_class] = aliases
            aliases = object_cache[raw_class]
            if not aliases:
                continue
            for kind in ("contacting_relationship", "spatial_relationship", "attention_relationship"):
                for relation in obj.get(kind, []) or []:
                    if relation not in relation_cache:
                        rel_verb = relation_to_action.get(relation)
                        if rel_verb is None:
                            rel_doc = nlp(str(relation).replace("_", " "))
                            rel_verb = next((t.lemma_.lower() for t in rel_doc if t.pos_ == "VERB"), None)
                        relation_cache[relation] = ACTION_ALIASES.get(rel_verb, rel_verb) if rel_verb else None
                    action = relation_cache[relation]
                    if action:
                        for noun in aliases:
                            evidence[vid].add((action, noun))
    return evidence


def all_verb_lemmas(doc) -> set[str]:
    return {ACTION_ALIASES.get(t.lemma_.lower(), t.lemma_.lower()) for t in doc if t.pos_ == "VERB"}


def all_event_pairs(doc) -> set[tuple[str, str]]:
    result = set()
    for verb in doc:
        if verb.pos_ != "VERB" or verb.lemma_.lower() in SKIP_VERBS:
            continue
        obj = object_token(verb)
        if obj and obj.lemma_.lower() not in PRONOUNS:
            result.add((predicate_components(verb)[0], obj.lemma_.lower()))
    return result


def source_row(raw: dict, split: str, graph: dict, verbs: set[str], event_pairs: set[tuple[str, str]]) -> dict:
    vid = raw.get("vid", raw.get("video_id"))
    windows = [[float(a), float(b)] for a, b in raw["relevant_windows"]]
    return {
        "qid": str(raw["qid"]), "video_id": vid, "vid": vid, "query": raw["query"],
        "duration": float(raw["duration"]), "relevant_windows": windows,
        "exist_label": 1, "existence_status": "present", "semantic_graph": graph,
        "source_qid": str(raw["qid"]), "source_split": split,
        "annotation_provenance": "Charades-STA positive via GMR pos_only export",
        "verification_status": "human_temporal_annotation", "_verbs": sorted(verbs),
        "_pairs": sorted(pair_id(p) for p in event_pairs),
    }


def select_compositions(train: list[dict], test: list[dict], held_actions: set[str], cap: int) -> set[tuple[str, str]]:
    train_count = Counter(pair(r["semantic_graph"]) for r in train if pair(r["semantic_graph"]) and r["semantic_graph"]["action"] not in held_actions)
    test_count = Counter(pair(r["semantic_graph"]) for r in test if pair(r["semantic_graph"]) and r["semantic_graph"]["action"] not in held_actions)
    candidates = [p for p in train_count if train_count[p] >= 8 and test_count[p] >= 3]
    candidates.sort(key=lambda p: (stable_bucket(pair_id(p)), -test_count[p], pair_id(p)))
    selected: set[tuple[str, str]] = set()
    action_count = Counter(r["semantic_graph"]["action"] for r in train)
    object_count = Counter(r["semantic_graph"]["object"] for r in train)
    for p in candidates:
        if len(selected) >= cap:
            break
        if action_count[p[0]] - train_count[p] < 20 or object_count[p[1]] - train_count[p] < 20:
            continue
        selected.add(p)
        action_count[p[0]] -= train_count[p]
        object_count[p[1]] -= train_count[p]
    return selected


def novelty(graph: dict, inventory: dict, held_actions: set[str], held_pairs: set[tuple[str, str]]) -> tuple[str, str]:
    action = graph["action"]
    p = pair(graph)
    if not action or p is None:
        return "unknown", "uncanonicalized"
    if action in held_actions:
        return "unseen", "unseen_action"
    if p in held_pairs:
        return "unseen", "unseen_composition"
    if p in inventory["pairs"]:
        return "seen", "seen_composition"
    if action in inventory["actions"] and graph["object"] in inventory["objects"]:
        return "unseen", "unseen_composition_unanchored"
    return "unknown", "out_of_inventory"


def replace_span(text: str, span: list[int], value: str) -> str:
    old = text[span[0]:span[1]]
    if old[:1].isupper():
        value = value.capitalize()
    return text[:span[0]] + value + text[span[1]:]


def inflect_like(source_token, replacement: str) -> str:
    tag = source_token.tag_
    irregular = {
        "put": {"VBD": "put", "VBN": "put", "VBG": "putting"},
        "take": {"VBD": "took", "VBN": "taken", "VBG": "taking"},
        "pick": {"VBD": "picked", "VBN": "picked", "VBG": "picking"},
        "hold": {"VBD": "held", "VBN": "held", "VBG": "holding"},
        "throw": {"VBD": "threw", "VBN": "thrown", "VBG": "throwing"},
        "drink": {"VBD": "drank", "VBN": "drunk", "VBG": "drinking"},
        "pour": {"VBD": "poured", "VBN": "poured", "VBG": "pouring"},
        "eat": {"VBD": "ate", "VBN": "eaten", "VBG": "eating"},
        "run": {"VBD": "ran", "VBN": "run", "VBG": "running"},
        "sit": {"VBD": "sat", "VBN": "sat", "VBG": "sitting"},
        "leave": {"VBD": "left", "VBN": "left", "VBG": "leaving"},
        "stand": {"VBD": "stood", "VBN": "stood", "VBG": "standing"},
    }
    if tag in irregular.get(replacement, {}):
        return irregular[replacement][tag]
    if tag == "VBG":
        return replacement[:-1] + "ing" if replacement.endswith("e") else replacement + "ing"
    if tag in {"VBD", "VBN"}:
        return replacement + ("d" if replacement.endswith("e") else "ed")
    if tag == "VBZ":
        return replacement + ("es" if replacement.endswith(("s", "sh", "ch")) else "s")
    return replacement


def inflect_noun_like(token: str, replacement: str) -> str:
    if not token.lower().endswith("s") or token.lower().endswith("ss"):
        return replacement
    if replacement.endswith(("s", "sh", "ch", "x", "z")):
        return replacement + "es"
    if replacement.endswith("y") and replacement[-2:-1] not in "aeiou":
        return replacement[:-1] + "ies"
    return replacement + "s"


def action_edit_allowed(source_doc, graph: dict, target_action: str, allow_general: bool = False) -> bool:
    source = graph["action"]
    if target_action not in ACTION_NEIGHBORS.get(source, []):
        if not allow_general or "_" in source or "_" in target_action:
            return False
        verb = next((t for t in source_doc if graph["action_span"] and t.idx == graph["action_span"][0]), None)
        return verb is not None and not any(t.dep_ == "prt" for t in verb.children)
    if source in {"take_off", "put_on"} and graph["object"] not in CLOTHING:
        return False
    verb = next((t for t in source_doc if graph["action_span"] and t.idx == graph["action_span"][0]), None)
    if verb is None:
        return False
    if source in {"hold", "throw", "dress", "undress", "enter", "leave"}:
        return any(t.dep_ in {"dobj", "obj"} for t in verb.children) and not any(t.dep_ == "prt" for t in verb.children)
    if source in {"pour", "drink"}:
        return not any(t.dep_ == "prep" and t.lower_ in {"into", "onto", "over"} for t in verb.children)
    return True


def positive_evidence(vid: str, candidate: tuple[str, str], query_index, charades_index, ag_index) -> list[str]:
    sources = []
    if candidate in query_index.get(vid, set()):
        sources.append("Charades-STA query")
    if candidate in charades_index.get(vid, set()):
        sources.append("Charades action interval")
    if candidate in ag_index.get(vid, set()):
        sources.append("Action Genome sampled graph")
    return sources


def output_positive(row: dict, inventory: dict, held_actions, held_pairs) -> dict | None:
    status, kind = novelty(row["semantic_graph"], inventory, held_actions, held_pairs)
    if status == "unknown" or kind == "unseen_composition_unanchored":
        return None
    output = {k: v for k, v in row.items() if not k.startswith("_")}
    output.update(semantic_status=status, novelty_type=kind, partition="U+" if status == "unseen" else "S+", construction_type="original_positive")
    return output


def make_candidate(source: dict, target_action: str | None, target_object: str | None, construction: str, inventory, held_actions, held_pairs, evidence, nlp, allow_general_action_edit=False) -> dict | None:
    graph = source["semantic_graph"]
    changed_action = target_action is not None and target_action != graph["action"]
    changed_object = target_object is not None and target_object != graph["object"]
    if changed_action == changed_object:
        return None
    span = graph["action_span"] if changed_action else graph["object_span"]
    if span is None:
        return None
    source_doc = nlp(source["query"])
    if changed_object:
        token = next((t for t in source_doc if t.idx == span[0]), None)
        if token is None or any(t.dep_ in {"compound", "amod", "poss", "nummod"} for t in token.children):
            return None
        if token.i + 1 < len(source_doc) and source_doc[token.i + 1].lower_ == "of":
            return None
        if target_object in {t.lemma_.lower() for t in source_doc}:
            return None
    if changed_action and not action_edit_allowed(source_doc, graph, target_action, allow_general_action_edit):
        return None
    old_text = source["query"][span[0]:span[1]]
    source_verb = next((t for t in source_doc if t.idx == span[0]), None)
    new_text = inflect_like(source_verb, target_action.split("_")[0]) if changed_action else inflect_noun_like(old_text, target_object)
    edits = [(span, new_text)]
    if changed_action and graph["particle_span"] is not None and "_" in target_action:
        edits.append((graph["particle_span"], target_action.split("_", 1)[1]))
    query = source["query"]
    for edit_span, edit_text in sorted(edits, key=lambda e: e[0][0], reverse=True):
        query = replace_span(query, edit_span, edit_text)
    if query.lower() == source["query"].lower():
        return None
    candidate_graph = {k: v for k, v in graph.items() if k not in {"action_span", "object_span", "particle_span"}}
    if changed_action:
        candidate_graph["action"] = target_action
        candidate_graph["action_base"] = target_action.split("_")[0]
        candidate_graph["action_class"] = target_action
        candidate_graph["action_wordnet_sense"] = wn_id(target_action, wn.VERB)
        candidate_graph["verbnet_classes"] = evidence["verbnet"].get(target_action.split("_")[0], [])
    if changed_object:
        candidate_graph["object"] = target_object
        candidate_graph["object_concept"] = target_object
        candidate_graph["object_wordnet_sense"] = wn_id(target_object, wn.NOUN)
    distance = wordnet_distance(
        graph["action_wordnet_sense"] if changed_action else graph["object_wordnet_sense"],
        candidate_graph["action_wordnet_sense"] if changed_action else candidate_graph["object_wordnet_sense"])
    p = pair(candidate_graph)
    signature = (p, candidate_graph["object_role"], candidate_graph["relation"])
    if signature not in evidence["all_human_signatures"]:
        return None
    reparsed = canonicalize(nlp(query), evidence["verbnet"])
    if (pair(reparsed), reparsed["object_role"], reparsed["relation"]) != signature:
        return None
    status, kind = novelty(candidate_graph, inventory, held_actions, held_pairs)
    if status == "unknown" or kind == "unseen_composition_unanchored":
        return None
    conflicts = positive_evidence(source["video_id"], p, evidence["queries"], evidence["charades"], evidence["ag"])
    if conflicts:
        return None
    return {
        "qid": "neg_" + hashlib.sha256((source["qid"] + "|" + query).encode()).hexdigest()[:16],
        "video_id": source["video_id"], "vid": source["video_id"],
        "query": query, "duration": source["duration"],
        "relevant_windows": [], "exist_label": 0, "existence_status": "absent_candidate",
        "semantic_status": status, "partition": "U-" if status == "unseen" else "S-",
        "novelty_type": kind, "construction_type": construction,
        "source_qid": source["qid"], "source_query": source["query"],
        "source_windows": source["relevant_windows"], "semantic_graph": candidate_graph,
        "annotation_provenance": "minimal edit of Charades-STA positive",
        "verification_status": "needs_video_review",
        "absence_evidence": "no matching event in available positive annotations; open-world absence unverified",
        "semantic_edit_distance": 1, "surface_token_edits": len(edits),
        "wordnet_semantic_distance": distance,
        "query_token_delta": len(query.split()) - len(source["query"].split()),
    }


def audit(rows: list[dict], train_inventory: dict, held_actions: set[str], held_pairs: set[tuple[str, str]], train_rows: list[dict]) -> dict:
    partitions = Counter(r["partition"] for r in rows)
    duplicate = Counter((r["video_id"], r["query"].lower()) for r in rows)
    queries_by_partition = defaultdict(list)
    actions_by_partition = defaultdict(Counter)
    objects_by_partition = defaultdict(Counter)
    durations_by_partition = defaultdict(list)
    windows_by_partition = defaultdict(list)
    provenance_by_partition = defaultdict(Counter)
    semantic_distances_by_partition = defaultdict(list)
    for r in rows:
        p = r["partition"]
        queries_by_partition[p].append(len(r["query"].split()))
        actions_by_partition[p][r["semantic_graph"]["action"]] += 1
        objects_by_partition[p][r["semantic_graph"]["object"]] += 1
        durations_by_partition[p].append(r["duration"])
        provenance_by_partition[p][r["annotation_provenance"]] += 1
        if r.get("wordnet_semantic_distance") is not None:
            semantic_distances_by_partition[p].append(r["wordnet_semantic_distance"])
        for a, b in (r["relevant_windows"] or r.get("source_windows", [])):
            windows_by_partition[p].append(round(((a + b) / 2) / r["duration"], 3) if r["duration"] else None)
    def summary(vals):
        vals = [v for v in vals if v is not None]
        return {"n": len(vals), "mean": round(statistics.mean(vals), 3), "median": round(statistics.median(vals), 3)} if vals else {"n": 0}
    held_pair_ids = {pair_id(p) for p in held_pairs}
    train_leak = [r["qid"] for r in train_rows if set(r["_verbs"]) & held_actions or set(r["_pairs"]) & held_pair_ids]
    opposite_semantics = defaultdict(set)
    for r in rows:
        opposite_semantics[pair(r["semantic_graph"])].add(r["semantic_status"])
    word_labels = defaultdict(lambda: Counter())
    for r in rows:
        for word in set(re.findall(r"[a-z]+", r["query"].lower())):
            word_labels[word][r["exist_label"]] += 1
    label_only = {w: dict(c) for w, c in word_labels.items() if sum(c.values()) >= 5 and len(c) == 1}
    return {
        "partition_counts": dict(partitions),
        "query_words": {p: summary(v) for p, v in queries_by_partition.items()},
        "video_duration": {p: summary(v) for p, v in durations_by_partition.items()},
        "temporal_center_fraction": {p: summary(v) for p, v in windows_by_partition.items()},
        "top_actions": {p: c.most_common(20) for p, c in actions_by_partition.items()},
        "top_objects": {p: c.most_common(20) for p, c in objects_by_partition.items()},
        "original_vs_generated": {p: dict(c) for p, c in provenance_by_partition.items()},
        "semantic_edit_distance": {"negatives": 1},
        "wordnet_semantic_distance": {p: summary(v) for p, v in semantic_distances_by_partition.items()},
        "duplicate_video_query": sum(v - 1 for v in duplicate.values() if v > 1),
        "train_heldout_leakage": train_leak,
        "semantic_seen_unseen_conflicts": [pair_id(p) for p, states in opposite_semantics.items() if p and len(states) > 1],
        "words_with_single_existence_label_min5": label_only,
        "warning": "Generated language is correlated with the candidate negative label. Video review and language balancing are required before official evaluation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/charades_sta/charades_pos_train.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/charades_sta/charades_pos_test.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/charades/annotations")
    parser.add_argument("--verbnet", type=Path, default=ROOT / "external/verbnet/verbnet3.4")
    parser.add_argument("--action-genome", type=Path, default=ROOT / "data/raw/action_genome/annotations/object_bbox_and_relationship.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/semantic_existence")
    parser.add_argument("--held-actions", nargs="+", default=None)
    parser.add_argument("--composition-cap", type=int, default=16)
    parser.add_argument("--split-spec", type=Path, help="Frozen v2 split specification; controls all held semantics")
    args = parser.parse_args()
    spec = load_split_spec(args.split_spec) if args.split_spec else None
    if spec and (args.held_actions is not None or args.composition_cap != 16):
        parser.error("--split-spec is the sole semantic selection input; omit --held-actions and --composition-cap")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    vn = vn_index(args.verbnet)
    inputs = [("train", read_jsonl(args.train)), ("test", read_jsonl(args.test))]
    records = []
    for split, raw_rows in inputs:
        docs = nlp.pipe((r["query"] for r in raw_rows), batch_size=256)
        records += [source_row(r, split, canonicalize(doc, vn), all_verb_lemmas(doc), all_event_pairs(doc)) for r, doc in zip(raw_rows, docs)]
    base_train = [r for r in records if r["source_split"] == "train" and stable_bucket(r["video_id"]) >= 10]
    base_val = [r for r in records if r["source_split"] == "train" and stable_bucket(r["video_id"]) < 10]
    base_test = [r for r in records if r["source_split"] == "test"]
    held_actions = set(spec["held_actions"] if spec else (args.held_actions or ["open", "close"]))
    held_pairs = ({tuple(p.split("|", 1)) for p in spec["held_compositions"]} if spec else
                  select_compositions(base_train, base_test, held_actions, args.composition_cap))
    held_pair_ids = {pair_id(p) for p in held_pairs}
    if spec:
        available_actions = {r["semantic_graph"]["action"] for r in base_train + base_test}
        available_pairs = {pair(r["semantic_graph"]) for r in base_train + base_test}
        if held_actions - available_actions or held_pairs - available_pairs:
            raise ValueError("Split spec contains semantics absent from parsed original positives")
    downstream_train, removed_train = [], []
    for r in base_train:
        if contains_held_semantics(r, held_actions, held_pair_ids):
            removed_train.append(r)
        elif pair(r["semantic_graph"]):
            downstream_train.append(r)
    inventory = {
        "actions": {r["semantic_graph"]["action"] for r in downstream_train},
        "objects": {r["semantic_graph"]["object"] for r in downstream_train},
        "pairs": {pair(r["semantic_graph"]) for r in downstream_train},
        "relations": {r["semantic_graph"]["relation"] for r in downstream_train if r["semantic_graph"]["relation"]},
        "relation_compositions": {f"{r['semantic_graph']['action']}|{r['semantic_graph']['object']}|{r['semantic_graph']['relation']}" for r in downstream_train if r["semantic_graph"]["relation"]},
    }
    if spec and spec["axis"] == "composition":
        unanchored = [pair_id(p) for p in held_pairs if p[0] not in inventory["actions"] or
                      p[1] not in inventory["objects"] or p in inventory["pairs"]]
        if unanchored:
            raise ValueError(f"Held compositions are not genuinely compositional: {unanchored}")
    if spec and held_actions & inventory["actions"]:
        raise ValueError("Held action leaked into the downstream train inventory")
    train_pos = [x for r in downstream_train if (x := output_positive(r, inventory, held_actions, held_pairs)) and x["partition"] == "S+"]
    val_pos = [x for r in base_val if (x := output_positive(r, inventory, held_actions, held_pairs))]
    test_pos = [x for r in base_test if (x := output_positive(r, inventory, held_actions, held_pairs))]
    all_source = records
    query_index = defaultdict(set)
    for r in all_source:
        for event_pair in r["_pairs"]:
            action, obj = event_pair.split("|", 1)
            query_index[r["video_id"]].add((action, obj))
    charades_index = read_charades_actions(args.metadata, nlp, vn)
    ag_index = read_action_genome(args.action_genome, nlp)
    all_human_pairs = {pair(r["semantic_graph"]) for r in all_source if pair(r["semantic_graph"])}
    all_human_signatures = {(pair(r["semantic_graph"]), r["semantic_graph"]["object_role"], r["semantic_graph"]["relation"]) for r in all_source if pair(r["semantic_graph"])}
    action_signatures = {(p[0], p[1], role, relation) for p, role, relation in all_human_signatures}
    evidence = {"queries": query_index, "charades": charades_index, "ag": ag_index,
                "verbnet": vn, "all_human_pairs": all_human_pairs,
                "all_human_signatures": all_human_signatures}
    def negatives(positives: list[dict], train_only_seen=False):
        result = []
        seen = set()
        positive_words = {w for r in positives for w in re.findall(r"[a-z]+", r["query"].lower())}
        for source in positives:
            graph = source["semantic_graph"]
            action, obj = graph["action"], graph["object"]
            choices = [(a, None, "action_counterfactual") for a in ACTION_NEIGHBORS.get(action, [])]
            if spec and spec["axis"] == "action":
                choices += [(a, None, "action_counterfactual") for a in sorted(held_actions)
                            if a != action and (a, obj, graph["object_role"], graph["relation"]) in action_signatures]
            observed_pairs = (query_index.get(source["video_id"], set()) |
                              charades_index.get(source["video_id"], set()) |
                              ag_index.get(source["video_id"], set()))
            video_objects = {o for _, o in observed_pairs}
            choices += [(None, o, "composition_counterfactual" if o in video_objects else "object_counterfactual")
                        for o in OBJECT_NEIGHBORS.get(obj, [])]
            # An action primitive observed elsewhere in this video, recombined with this object's concept.
            video_actions = sorted({a for a, _ in query_index.get(source["video_id"], set()) if a != action})
            choices += [(a, None, "composition_counterfactual") for a in video_actions if (a, obj) in held_pairs]
            for a, o, kind in choices:
                candidate = make_candidate(source, a, o, kind, inventory, held_actions, held_pairs, evidence, nlp,
                                           allow_general_action_edit=bool(spec and spec["axis"] == "action" and a in held_actions))
                if not candidate or (train_only_seen and candidate["partition"] != "S-"):
                    continue
                if set(re.findall(r"[a-z]+", candidate["query"].lower())) - positive_words:
                    continue
                key = (candidate["video_id"], candidate["query"].lower())
                if key in seen:
                    continue
                seen.add(key)
                result.append(candidate)
        return result
    val_neg = negatives(val_pos)
    test_neg = negatives(test_pos)
    train_neg = negatives(train_pos, train_only_seen=True)
    # Pair source positive and unseen negative in the same video, using a single changed concept.
    test_uplus = {r["qid"]: r for r in test_pos if r["partition"] == "U+"}
    pairs = []
    used_positives = set()
    ordered_negatives = sorted(test_neg, key=lambda r: (
        0 if r["construction_type"] == "action_counterfactual" else 1,
        abs(r["query_token_delta"]), r["qid"]))
    for neg in ordered_negatives:
        if neg["partition"] == "U-" and neg["source_qid"] in test_uplus:
            pos = test_uplus[neg["source_qid"]]
            if pos["novelty_type"] != neg["novelty_type"]:
                continue
            if pos["qid"] in used_positives:
                continue
            used_positives.add(pos["qid"])
            pairs.append({"pair_id": "pair_" + neg["qid"], "positive_qid": pos["qid"],
                          "negative_qid": neg["qid"], "video_id": pos["video_id"],
                          "shared_source": True, "semantic_edit_distance": 1,
                          "query_token_delta": neg["query_token_delta"],
                          "negative_verification_status": neg["verification_status"]})
    train_clean = train_pos + train_neg
    val_candidates = val_pos + val_neg
    test_candidates = test_pos + test_neg
    args.out.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.out / "train_candidates.jsonl", train_clean)
    dump_jsonl(args.out / "val_candidates.jsonl", val_candidates)
    dump_jsonl(args.out / "test_candidates.jsonl", test_candidates)
    dump_jsonl(args.out / "test_positives.jsonl", test_pos)
    dump_jsonl(args.out / "test_negative_review_queue.jsonl", test_neg)
    dump_jsonl(args.out / "matched_u_pairs.jsonl", pairs)
    dump_jsonl(args.out / "removed_train_holdouts.jsonl", [{k: v for k, v in r.items() if not k.startswith("_")} for r in removed_train])
    dump_json(args.out / "semantic_inventory.json", {
        "actions": sorted(inventory["actions"]), "objects": sorted(inventory["objects"]),
        "action_object_compositions": sorted(pair_id(p) for p in inventory["pairs"]),
        "relations": sorted(inventory["relations"]), "heldout_actions": sorted(held_actions),
        "relation_compositions": sorted(inventory["relation_compositions"]),
        "heldout_compositions": sorted(pair_id(p) for p in held_pairs),
        "inventory_source": "downstream training positives only after holdout removal",
    })
    report = {
        "source_counts": {"train": len(inputs[0][1]), "test": len(inputs[1][1])},
        "split_counts": {"downstream_train_positives": len(train_pos), "val_positives": len(val_pos), "test_positives": len(test_pos),
                         "removed_train_holdouts": len(removed_train), "train_negative_candidates": len(train_neg),
                         "val_negative_candidates": len(val_neg), "test_negative_candidates": len(test_neg),
                         "matched_u_pairs": len(pairs)},
        "external_evidence": {"action_genome_loaded": args.action_genome.exists(), "charades_video_action_metadata_loaded": bool(charades_index)},
        "train": audit(train_clean, inventory, held_actions, held_pairs, downstream_train),
        "val": audit(val_candidates, inventory, held_actions, held_pairs, downstream_train),
        "test": audit(test_candidates, inventory, held_actions, held_pairs, downstream_train),
        "formal_test_status": "positives annotated; all negatives require manual video review before official scoring",
    }
    dump_json(args.out / "audit.json", report)
    if spec:
        dump_json(args.out / "split_spec_provenance.json", {
            "split_spec": spec,
            "split_spec_sha256": hashlib.sha256(args.split_spec.read_bytes()).hexdigest(),
            "split_spec_path": str(args.split_spec.resolve()),
        })
    if report["train"]["train_heldout_leakage"] or report["test"]["semantic_seen_unseen_conflicts"]:
        raise RuntimeError("Semantic consistency failure; inspect audit.json")
    print(json.dumps({"outputs": str(args.out), "split_counts": report["split_counts"],
                      "test_partitions": report["test"]["partition_counts"], "action_genome_loaded": args.action_genome.exists()}, indent=2))


if __name__ == "__main__":
    main()
