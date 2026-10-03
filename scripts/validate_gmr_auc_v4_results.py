from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.moment_detr_gmr_auc_v4.common import RELEASE_ROOT, SPLITS, auc, read_jsonl, write_json


def main():
    checks = {}
    for split in SPLITS:
        d = ROOT / "results/moment_detr_gmr_auc_v4" / split
        preds = read_jsonl(d / "test_predictions.jsonl")
        source = read_jsonl(RELEASE_ROOT / split / "test.jsonl")
        summ = json.loads((d / "v4_summary.json").read_text())
        qids = [str(p["qid"]) for p in preds]
        if len(qids) != len(set(qids)) or set(qids) != {str(r["qid"]) for r in source}:
            raise AssertionError(f"{split}: test qid coverage/uniqueness failure")
        rows = {str(r["qid"]): r for r in source}
        for p in preds:
            r = rows[str(p["qid"])]
            if (p["partition"], int(p["exist_label"])) != (r["partition"], int(r["exist_label"])):
                raise AssertionError(f"{split}: test metadata mismatch for {p['qid']}")
            for name in ("pred_exist_logit_base", "pred_exist_score_base", "pred_exist_logit_v4", "pred_exist_score_v4", "pred_exist_delta"):
                if not math.isfinite(float(p[name])): raise AssertionError(f"{split}: nonfinite {name}")
            for variant in ("base", "v4"):
                if abs(1.0 / (1.0 + math.exp(-p[f"pred_exist_logit_{variant}"])) - p[f"pred_exist_score_{variant}"]) > 1e-15:
                    raise AssertionError(f"{split}: probability serialization mismatch")
            if abs(p["pred_exist_delta"]) > 2.0 + 1e-7: raise AssertionError(f"{split}: residual out of bounds")
            if p["raw_spans_cxw_normalized_base"] != p["raw_spans_cxw_normalized_v4"] or p["raw_spans_seconds_base"] != p["raw_spans_seconds_v4"] or p["raw_class_logits_base"] != p["raw_class_logits_v4"]:
                raise AssertionError(f"{split}: localization arrays differ")
        y = np.asarray([p["exist_label"] for p in preds]); part = np.asarray([p["partition"] for p in preds])
        base = np.asarray([p["pred_exist_logit_base"] for p in preds]); v4 = np.asarray([p["pred_exist_logit_v4"] for p in preds])
        calc = {}
        for label, mask in (("seen", np.char.startswith(part, "S")), ("unseen", np.char.startswith(part, "U"))):
            calc[f"baseline_{label}_auroc"] = auc(y[mask], base[mask]); calc[f"v4_{label}_auroc"] = auc(y[mask], v4[mask])
            if abs(calc[f"baseline_{label}_auroc"] - summ[f"baseline_{label}_auroc"]) > 1e-12: raise AssertionError(f"{split}: baseline {label} AUROC mismatch")
            if abs(calc[f"v4_{label}_auroc"] - summ[f"v4_{label}_auroc"]) > 1e-12: raise AssertionError(f"{split}: V4 {label} AUROC mismatch")
        tr = np.load(d / "feature_bank_train.npz", allow_pickle=False); va = np.load(d / "feature_bank_val.npz", allow_pickle=False)
        if np.char.startswith(tr["partition"].astype(str), "U").any() or np.char.startswith(va["partition"].astype(str), "U").any(): raise AssertionError(f"{split}: U leaked into a feature bank")
        history = json.loads((d / "adapter_training_history.json").read_text())
        h = history["history"]
        highest = max(x["val"]["pooled_auroc"] for x in h)
        earliest = min(x["epoch"] for x in h if x["val"]["pooled_auroc"] == highest)
        if earliest != summ["best_adapter_epoch"]: raise AssertionError(f"{split}: checkpoint selection mismatch")
        if summ["best_adapter_epoch"] == 0 and summ["seen_val_v4_best_auroc"] != summ["seen_val_baseline_auroc"]: raise AssertionError(f"{split}: epoch0 did not equal baseline")
        checks[split] = {"test_qids": len(qids), "train_bank_count": len(tr["qid"]), "val_bank_count": len(va["qid"]),
                         "predictions_full_precision_roundtrip": True, "delta_bounded": True, "localization_identical": True,
                         "seen_unseen_auc_recomputed": True, "best_epoch_rule_verified": True,
                         "no_u_in_train_or_val_banks": True, "nan_inf": False}
    report = {"status": "pass", "splits": checks}
    write_json(ROOT / "experiments/moment_detr_gmr_auc_v4/RESULT_VALIDATION.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
