from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def _sample(rng, pool, count):
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) == 0:
        raise ValueError("cannot sample from an empty pool")
    replace = len(pool) < count
    return rng.choice(pool, size=count, replace=replace), replace


class PairSampler:
    def __init__(self, bank, seed=3407, same_pairs=256):
        self.rng = np.random.default_rng(seed)
        labels = bank["exist_label"].astype(np.int8)
        self.pos = np.flatnonzero(labels == 1)
        self.neg = np.flatnonzero(labels == 0)
        if len(self.pos) == 0 or len(self.neg) == 0:
            raise ValueError("global AUC requires both Seen labels")
        by_query, by_comp, by_action = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, (q, c, a) in enumerate(zip(bank["normalized_query"], bank["composition_id"], bank["action_id"])):
            if labels[i] == 0:
                by_query[str(q)].append(i); by_comp[str(c)].append(i); by_action[str(a)].append(i)
        self.same_anchors = []
        self.same_pools = []
        self.same_levels = []
        self.unpaired = 0
        for p in self.pos:
            pools = (by_query[str(bank["normalized_query"][p])],
                     by_comp[str(bank["composition_id"][p])],
                     by_action[str(bank["action_id"][p])])
            for level, pool in enumerate(pools):
                if pool:
                    self.same_anchors.append(int(p)); self.same_pools.append(np.asarray(pool, dtype=np.int64))
                    self.same_levels.append(level); break
            else:
                self.unpaired += 1
        self.same_anchors = np.asarray(self.same_anchors, dtype=np.int64)
        self.same_levels = np.asarray(self.same_levels, dtype=np.int8)
        self.same_pairs = int(same_pairs)

    def global_indices(self):
        pos, rep_pos = _sample(self.rng, self.pos, 128)
        neg, rep_neg = _sample(self.rng, self.neg, 128)
        return pos, neg, bool(rep_pos or rep_neg)

    def semantic_indices(self):
        if not len(self.same_anchors):
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=np.int8)
        anchors, _ = _sample(self.rng, self.same_anchors, self.same_pairs)
        # Anchor index identifies one priority pool. Duplicate anchors remain independent draws.
        # Keep the original sampled pool index so repeated positive anchors can
        # still select among their own priority-qualified negatives.
        anchor_pool_index = np.asarray([self.rng.choice(np.flatnonzero(self.same_anchors == a)) for a in anchors])
        negs = np.asarray([self.rng.choice(self.same_pools[j]) for j in anchor_pool_index], dtype=np.int64)
        levels = self.same_levels[anchor_pool_index]
        return anchors, negs, levels


def ranking_losses(scores, pos_idx, neg_idx, same_pos_idx, same_neg_idx):
    if len(pos_idx):
        diff = scores[pos_idx][:, None] - scores[neg_idx][None, :]
        global_loss = F.softplus(1.0 - diff).mean()
    else:
        global_loss = scores.sum() * 0.0
    if len(same_pos_idx):
        same_loss = F.softplus(1.0 - scores[same_pos_idx] + scores[same_neg_idx]).mean()
    else:
        same_loss = scores.sum() * 0.0
    return global_loss, same_loss
