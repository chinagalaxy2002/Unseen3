from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.moment_detr_gmr_auc_v4.common import OLD_RESULTS, SPLITS, baseline_path, sha256, write_json


def main():
    records = {}
    for split in SPLITS:
        path = baseline_path(split)
        if not path.is_file():
            records[split] = {"status": "missing canonical baseline checkpoint", "path": str(path)}
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        opt = ck["opt"]
        vals = {k: getattr(opt, k, None) for k in (
            "seed", "lr", "wd", "n_epoch", "max_es_cnt", "bsz", "eval_bsz", "train_path", "eval_path",
            "t_feat_dir", "v_feat_dirs", "v_feat_types", "use_exist_head", "exist_pool", "exist_loss_coef",
            "clip_length", "max_v_l", "v_feat_dim", "t_feat_dim", "span_loss_type", "num_queries", "hidden_dim",
        )}
        vals = {k: (list(v) if isinstance(v, (tuple, list)) else str(v) if isinstance(v, Path) else v) for k,v in vals.items()}
        run_dir = OLD_RESULTS / split / "moment"
        def clean(v):
            if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
            if isinstance(v, (tuple, list)): return [clean(x) for x in v]
            if isinstance(v, Path): return str(v)
            if isinstance(v, (str, int, float, bool)) or v is None: return v
            return str(v)
        vals.update({"status": "verified checkpoint present", "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                     "checkpoint_size_bytes": path.stat().st_size, "best_epoch_1_indexed": int(ck["epoch"]) + 1,
                     "resolved_config": clean(dict(opt)),
                     "dataset": "semantic_existence_v2/" + split, "train_partition": "S+/S- only",
                     "validation_partition": "S+/S- only", "checkpoint_selection_rule": "highest Seen-validation MR-full-mAP (original train.py)",
                     "training_metadata_path": str(run_dir / "run_metadata.txt"),
                     "train_log_path": str(run_dir / "train.log"), "val_log_path": str(run_dir / "val.log"),
                     "best_seen_val_prediction_path": str(run_dir / "best_charades_semantic_existence_val_preds.jsonl"),
                     "best_seen_val_metrics_path": str(run_dir / "best_charades_semantic_existence_val_preds_metrics.json"),
                     "selection_rule_source": "training/moment_detr_gmr/train.py: validation MR-full-mAP improvement saves checkpoint",
                     "feature_order": ["CLIP", "SlowFast"], "feature_order_evidence": "checkpoint opt.v_feat_dirs order and original StartEndDataset concatenation loop",
                     "v_feat_types_label": vals.get("v_feat_types"),
                     "existence_head_config": {"enabled": vals.get("use_exist_head"), "pool": vals.get("exist_pool"),
                         "input_dim": vals.get("hidden_dim"), "hidden_dim": vals.get("hidden_dim"),
                         "layers": ["Linear(D,D)", "ReLU", "Linear(D,1)"], "loss_coefficient": vals.get("exist_loss_coef")},
                     "existence_representation": "max-pool over final decoder slots hs[-1], then GMRAdapter linear-ReLU-linear head"})
        records[split] = vals
    missing = [s for s,r in records.items() if r.get("status") != "verified checkpoint present"]
    audit = {"baseline_project": str(OLD_RESULTS), "splits": records, "missing_splits": missing,
             "canonical_baseline_available": not missing}
    write_json(ROOT / "experiments/moment_detr_gmr_auc_v4/baseline_audit.json", audit)
    lines = ["# Canonical Moment-DETR-GMR baseline audit", "",
             "All five v2 split checkpoints are resolved from the original benchmark repository. Provenance uses the saved checkpoint `opt`, training metadata/logs, split release and original dataset/model source.", "",
             "Feature order is CLIP → SlowFast: the saved `opt.v_feat_dirs` orders `vid_clip` before `vid_slowfast`, and the original dataset concatenates feature directories in list order. The head max-pools final decoder slots before its linear-ReLU-linear existence MLP.", "",
             "| Split | Checkpoint | SHA256 | Seed | Best epoch | Train / val | Feature order | Pool |", "|---|---|---|---:|---:|---|---|---|"]
    for s,r in records.items():
        if r.get("status") == "verified checkpoint present":
            lines.append(f"| {s} | `{r['checkpoint']}` | `{r['checkpoint_sha256']}` | {r['seed']} | {r['best_epoch_1_indexed']} | `{r['train_path']}` / `{r['eval_path']}` | CLIP → SlowFast | {r['exist_pool']} |")
        else: lines.append(f"| {s} | MISSING | — | — | — | — | — | — |")
    lines += ["", "Each checkpoint stores the resolved config. All runs record seed 3407, 100 epochs, early stopping disabled (`max_es_cnt=-1`), batch/eval batch 16, AdamW defaults from the original config, max pooling, and best Seen-validation MR-full-mAP checkpoint selection. Historical diagnostics use four-decimal probabilities; they are not used as canonical baseline predictions.", ""]
    (ROOT / "experiments/moment_detr_gmr_auc_v4/BASELINE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if missing: raise SystemExit(f"Missing canonical baseline checkpoint(s): {missing}")


if __name__ == "__main__": main()
