"""Train a fresh inner-fold baseline with canonical saved configuration."""
from __future__ import annotations

import argparse
import copy
import datetime
import sys
import time
from pathlib import Path

import torch

from training.moment_detr_gmr_evidence_v5.protocol import (
    ROOT, EXPERIMENT, RESULTS, BASELINES, digest, read_rows, semantic, write_json,
)


def run(fold, verify_only=False):
    import json
    directory = EXPERIMENT / "inner_folds" / fold
    manifest = json.loads((directory / "manifest.json").read_text())
    roles = {}
    for role, entry in manifest["files"].items():
        if digest(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Changed fold data: {role}")
        roles[role] = read_rows(entry["path"], seen_only=True)
    heldout = set(manifest["heldout"])
    parent = manifest["parent_split"]
    from training.moment_detr_gmr_evidence_v5.protocol import RELEASE
    source_rows = {str(r["qid"]): r for r in read_rows(RELEASE / parent / "train.jsonl", seen_only=True)}
    for r in roles["inner_train"]:
        if semantic(r, manifest["level"]) in heldout:
            raise RuntimeError("Query semantic exposure")
        if int(r["exist_label"]) == 0:
            source = source_rows.get(str(r.get("source_qid")))
            if source is None or semantic(source, manifest["level"]) in heldout:
                raise RuntimeError("Source semantic exposure")
    video_sets = [{r["vid"] for r in roles[k]} for k in ("inner_train", "inner_seen_val", "inner_novel_dev")]
    if any(video_sets[i] & video_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Inner video role overlap")

    checkpoint = BASELINES / parent / "moment/best.ckpt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    opt = copy.deepcopy(saved["opt"])
    del saved  # Canonical weights are not loaded into this fresh task model.
    output = RESULTS / "inner_baselines" / fold
    output.mkdir(parents=True, exist_ok=True)
    opt.train_path = manifest["files"]["inner_train"]["path"]
    opt.eval_path = manifest["files"]["inner_seen_val"]["path"]
    opt.results_dir = str(output)
    for key, filename in (("ckpt_filepath", "best.ckpt"), ("train_log_filepath", "train.log"), ("eval_log_filepath", "val.log")):
        setattr(opt, key, str(output / filename))
    opt.n_epoch = 100
    opt.max_es_cnt = -1
    opt.seed = 3407
    opt.device = "cuda"
    opt.mr_only = True
    opt.lw_saliency = 0
    sys.path.insert(0, str(ROOT / "training/moment_detr_gmr"))
    from training.moment_detr_gmr.train import main, build_dataset_config
    from dataset import StartEndDataset
    coverage = {}
    for role in ("inner_train", "inner_seen_val", "inner_novel_dev"):
        ds = StartEndDataset(**build_dataset_config(opt, manifest["files"][role]["path"], keep_empty_gt=True))
        if len(ds.data) != len(roles[role]):
            raise RuntimeError(f"Missing features in {role}")
        coverage[role] = {"rows": len(ds.data), "missing": 0}
    audit = {"fold": fold, "configuration_source": str(checkpoint),
             "configuration_checkpoint_hash": digest(checkpoint), "weights_loaded": False,
             "initialization": "random", "manifest_hash": digest(directory / "manifest.json"),
             "epochs_planned": 100, "seed": 3407,
             "selection": "inner Seen-val MR-full-mAP", "coverage": coverage,
             "feature_order": list(opt.v_feat_dirs), "source_exposure": 0,
             "video_role_overlap": 0, "novel_dev_used_for_selection": False}
    write_json(output / "input_audit.json", audit)
    write_json(output / "resolved_config.json", dict(opt))
    if verify_only:
        print(audit)
        return
    if (output / "best.ckpt").exists() or (output / "train.log").exists():
        raise RuntimeError(f"Existing training artifacts: {output}; use an explicit new run directory")
    started = time.time()
    status = dict(audit, status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(output / "status.json", status)
    try:
        main(opt, resume=None)
        if not (output / "best.ckpt").is_file():
            raise RuntimeError("Missing trained checkpoint")
        status.update(status="completed", epochs_trained=100, elapsed_seconds=time.time() - started,
                      checkpoint_hash=digest(output / "best.ckpt"))
    except BaseException as exc:
        status.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        raise
    finally:
        write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run(args.fold, args.verify_only)
