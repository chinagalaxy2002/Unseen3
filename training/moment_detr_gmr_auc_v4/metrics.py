from __future__ import annotations

import numpy as np

from training.moment_detr_gmr_auc_v4.common import auc


def balanced_accuracy_threshold(y, scores):
    y = np.asarray(y, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best_t, best = 0.5, -1.0
    for t in candidates:
        pred = scores >= t
        tpr = np.mean(pred[y == 1]); tnr = np.mean(~pred[y == 0])
        ba = 0.5 * (tpr + tnr)
        if ba > best:
            best, best_t = float(ba), float(t)
    return float(best_t), float(best)


def paired_bootstrap(y, base, v4, seed=3407, replicates=2000):
    y = np.asarray(y, dtype=np.int8); base = np.asarray(base, dtype=np.float64); v4 = np.asarray(v4, dtype=np.float64)
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(p) or not len(n): raise ValueError("stratified AUROC bootstrap requires both classes")
    rng = np.random.default_rng(seed); deltas = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        idx = np.concatenate([rng.choice(p, size=len(p), replace=True), rng.choice(n, size=len(n), replace=True)])
        deltas[b] = auc(y[idx], v4[idx]) - auc(y[idx], base[idx])
    return {"replicates": replicates, "seed": seed, "delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "delta_median": float(np.median(deltas)), "delta_mean": float(np.mean(deltas))}


def raw_r1_at_05(rows, predictions):
    vals = []
    for r in rows:
        if int(r["exist_label"]) != 1: continue
        pred = predictions[str(r["qid"])]
        if not r.get("relevant_windows"): continue
        j = int(np.asarray(pred["raw_class_logits"])[:, 0].argmax())
        st, ed = pred["raw_spans_seconds"][j]
        best = 0.0
        for gs, ge in r["relevant_windows"]:
            inter = max(0.0, min(ed, ge) - max(st, gs)); union = max(ed, ge) - min(st, gs)
            best = max(best, inter / union if union > 0 else 0.0)
        vals.append(best >= 0.5)
    return float(np.mean(vals)) if vals else float("nan")


def matched_pair_acc(pair_rows, scores):
    vals = []
    for p in pair_rows:
        pos, neg = str(p["positive_qid"]), str(p["negative_qid"])
        if pos in scores and neg in scores:
            vals.append(1.0 if scores[pos] > scores[neg] else (0.5 if scores[pos] == scores[neg] else 0.0))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)

