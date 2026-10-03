"""Build source-pair manifests and inner folds using Seen metadata only."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import (
    EXPERIMENT, RELEASE, RESULTS, SPLITS, digest, pairs_for, read_rows,
    semantic, write_json, write_rows,
)


def counts(rows):
    return {"rows": len(rows), "videos": len({r["vid"] for r in rows}),
            "labels": dict(collections.Counter(str(r["exist_label"]) for r in rows))}


def make_fold(split, level, heldout, train, val):
    heldout = set(heldout)
    train_index = {str(r["qid"]): r for r in train}
    train_kept = []
    for row in train:
        if semantic(row, level) in heldout:
            continue
        if int(row["exist_label"]) == 0:
            source = train_index.get(str(row.get("source_qid")))
            if source is None or semantic(source, level) in heldout:
                continue
        train_kept.append(row)
    novel = [r for r in val if semantic(r, level) in heldout]
    novel_vids = {r["vid"] for r in novel}
    seen = [r for r in val if r["vid"] not in novel_vids]
    eval_vids = {r["vid"] for r in val}
    train_kept = [r for r in train_kept if r["vid"] not in eval_vids]
    if len(novel) == 0 or min(sum(r["exist_label"] == y for r in novel) for y in (0, 1)) < 50:
        return None
    if len(novel_vids) < 30 or min(sum(r["exist_label"] == y for r in seen) for y in (0, 1)) < 50:
        return None
    if level == "composition":
        actions = {semantic(r, "action") for r in train_kept if r["exist_label"] == 1}
        objects = {semantic(r, "composition").split("::", 1)[1] for r in train_kept if r["exist_label"] == 1}
        if any(k.split("::", 1)[0] not in actions or k.split("::", 1)[1] not in objects for k in heldout):
            return None
    return train_kept, seen, novel


def main():
    fold_specs = []
    audits = {}
    for split in SPLITS:
        tr = read_rows(RELEASE / split / "train.jsonl", seen_only=True)
        va = [r for r in read_rows(RELEASE / split / "val.jsonl") if r["partition"].startswith("S")]
        pairs = pairs_for(tr, split)
        path = EXPERIMENT / "source_pairs" / split
        write_rows(path / "train_source_pairs.jsonl", pairs)
        write_rows(path / "val_source_pairs.jsonl", pairs_for(va, split))
        audits[split] = {"train": counts(tr), "seen_val": counts(va),
                         "eligible_pairs": sum(p["pair_eligible"] for p in pairs),
                         "exclusions": dict(collections.Counter(p["exclusion_reason"] for p in pairs if not p["pair_eligible"])),
                         "train_hash": digest(RELEASE / split / "train.jsonl"),
                         "pair_hash": digest(path / "train_source_pairs.jsonl")}
        if split not in ("A1", "C1"):
            continue
        level = "action" if split == "A1" else "composition"
        buckets = collections.defaultdict(lambda: [0, 0])
        for r in va:
            buckets[semantic(r, level)][int(r["exist_label"])] += 1
        candidates = sorted((k for k, v in buckets.items() if min(v) >= 3),
                            key=lambda k: (-min(buckets[k]), -sum(buckets[k]), k))
        used = set()
        for number in (1, 2):
            selected, generated = [], None
            for candidate in candidates:
                if candidate in used:
                    continue
                selected.append(candidate)
                generated = make_fold(split, level, selected, tr, va)
                if generated is not None:
                    break
            if generated is None:
                raise RuntimeError(f"Cannot build {split}/{level}/{number}: metadata support insufficient")
            used.update(selected)
            name = f"{split}_{level}_{number:02d}"
            directory = EXPERIMENT / "inner_folds" / name
            train, seen, novel = generated
            roles = {"inner_train": train, "inner_seen_val": seen, "inner_novel_dev": novel}
            for role, rows in roles.items():
                write_rows(directory / (role + ".jsonl"), rows)
                write_rows(directory / (role + "_source_pairs.jsonl"), pairs_for(rows, name))
            manifest = {"fold": name, "parent_split": split, "level": level, "heldout": selected,
                        "selection": "Seen metadata only; support-greedy, deterministic; disjoint heldout IDs",
                        "counts": {k: counts(v) for k, v in roles.items()},
                        "files": {k: {"path": str(directory / (k + '.jsonl')),
                                      "sha256": digest(directory / (k + '.jsonl'))} for k in roles},
                        "excluded_train_rows": len(tr) - len(train),
                        "excluded_val_context_rows": len(va) - len(seen) - len(novel),
                        "train_source_semantic_exposure": 0,
                        "train_eval_video_overlap": 0,
                        "initialization": "random task-model initialization; no canonical trained weights",
                        "baseline_epochs": 100, "seed": 3407,
                        "baseline_selection": "inner Seen-val MR-full-mAP"}
            # Verify the claims rather than merely record them.
            if {r['vid'] for r in train} & {r['vid'] for r in seen + novel}:
                raise RuntimeError("Video overlap")
            if any(semantic(r, level) in set(selected) for r in train):
                raise RuntimeError("Heldout semantics leaked")
            write_json(directory / "manifest.json", manifest)
            fold_specs.append(manifest)
    write_json(EXPERIMENT / "source_pair_audit.json", audits)
    write_json(EXPERIMENT / "inner_fold_index.json", {"folds": fold_specs, "status": "metadata_prepared"})
    print({s: a["eligible_pairs"] for s, a in audits.items()})
    print([(f["fold"], f["heldout"], f["counts"]) for f in fold_specs])


if __name__ == "__main__":
    main()
