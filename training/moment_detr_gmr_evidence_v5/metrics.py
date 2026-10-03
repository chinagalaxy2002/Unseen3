from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels, scores):
    return float(roc_auc_score(np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)))


def source_pair_metrics(rows, scores):
    indices = {str(r["qid"]): i for i, r in enumerate(rows)}
    outcomes, gaps = [], []
    for i, row in enumerate(rows):
        if row["exist_label"] != 0:
            continue
        source = indices.get(str(row.get("source_qid")))
        if source is None or rows[source]["exist_label"] != 1 or rows[source]["vid"] != row["vid"]:
            continue
        difference = float(scores[source] - scores[i])
        outcomes.append(float(difference > 0) + 0.5 * float(difference == 0))
        gaps.append(difference)
    return {"pair_acc": float(np.mean(outcomes)) if outcomes else None, "pair_count": len(outcomes),
            "mean_score_gap": float(np.mean(gaps)) if gaps else None}


def cluster_delta_interval(rows, base, scores, replicates=5000, seed=3407):
    """Two models share video weights; every query on a video keeps its weight."""
    vids = np.asarray([str(r["vid"]) for r in rows])
    _, inverse = np.unique(vids, return_inverse=True)
    n = int(inverse.max()) + 1
    labels = np.asarray([r["exist_label"] for r in rows], dtype=np.int8)
    rng, deltas, attempts = np.random.default_rng(seed), [], 0
    while len(deltas) < replicates and attempts < replicates * 10:
        attempts += 1
        draws = rng.integers(0, n, size=n)
        weights = np.bincount(draws, minlength=n)[inverse]
        if any(weights[labels == y].sum() == 0 for y in (0, 1)):
            continue
        deltas.append(roc_auc_score(labels, scores, sample_weight=weights) - roc_auc_score(labels, base, sample_weight=weights))
    if len(deltas) != replicates:
        return {"estimable": False, "valid_replicates": len(deltas), "attempts": attempts}
    return {"estimable": True, "replicates": replicates, "attempts": attempts, "seed": seed,
            "unit": "video", "delta_ci95": np.quantile(deltas, [.025, .975]).tolist()}


def diagnostic(rows, scores):
    labels = [r["exist_label"] for r in rows]
    return {"pooled_auroc": auc(labels, scores), "rows": len(rows),
            "videos": len({r["vid"] for r in rows}), "source_pairs": source_pair_metrics(rows, scores)}
