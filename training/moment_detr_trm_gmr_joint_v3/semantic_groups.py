"""Frozen semantic supervision and deterministic Seen-only balanced batches."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler
from sklearn.metrics import roc_auc_score


def normalize_query(query):
    return ' '.join(query.lower().strip().split())


def semantic_keys(row):
    graph = row.get('semantic_graph', {})
    action = graph.get('action_base')
    if not action:
        raise ValueError(f'Missing frozen action lemma for {row.get("qid")}')
    # Use the released parser's object concept directly; no re-parsing or U fitting.
    obj = graph.get('object_concept') or graph.get('object') or '<no_object>'
    return action, (action, obj)


def label(row):
    if row['partition'] not in ('S+', 'S-'):
        raise ValueError('Training/selection group construction requires Seen-only rows')
    y = int(row['partition'] == 'S+')
    if int(row['exist_label']) != y:
        raise ValueError(f'Inconsistent existence label: {row["qid"]}')
    return y


def group_pools(rows, level):
    pools = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(rows):
        pools[semantic_keys(row)[level]][label(row)].append(index)
    return dict(pools)


def serialize_group(key):
    return list(key) if isinstance(key, tuple) else key


class SemanticGroupBatchSampler(Sampler):
    """Alternate action/composition batches, each 4 groups x (2 S+ + 2 S-).

    Groups need at least one example per class; draws use replacement only when
    a class has fewer than two examples. Epoch length is ceil(N/16), as before.
    One pair in each block prefers identical query, then identical composition;
    the second pair explores the full selected group. This retains broader action
    coverage than sampling only jointly supported compositions.
    """
    def __init__(self, rows, seed=3407):
        self.rows, self.seed, self.epoch = rows, int(seed), 0
        self.pools = [group_pools(rows, level) for level in (0, 1)]
        self.eligible = [sorted(k for k, v in pools.items() if v[0] and v[1]) for pools in self.pools]
        if min(map(len, self.eligible)) < 4:
            raise ValueError('At least four positive/negative-supported groups per level required')
        self.anchor = [{}, {}]
        for level in (0, 1):
            for key in self.eligible[level]:
                pools = self.pools[level][key]
                exact = defaultdict(lambda: {0: [], 1: []})
                comp = defaultdict(lambda: {0: [], 1: []})
                for y in (0, 1):
                    for i in pools[y]:
                        exact[normalize_query(rows[i]['query'])][y].append(i)
                        comp[semantic_keys(rows[i])[1]][y].append(i)
                pairs = [v for k, v in sorted(exact.items()) if v[0] and v[1]]
                if not pairs:
                    pairs = [v for k, v in sorted(comp.items()) if v[0] and v[1]]
                self.anchor[level][key] = pairs

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.rows) / 16)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for step in range(len(self)):
            level = (step + self.epoch) % 2
            chosen = rng.choice(len(self.eligible[level]), size=4, replace=False)
            batch = []
            for gi in chosen:
                key = self.eligible[level][int(gi)]
                pools, anchors = self.pools[level][key], self.anchor[level][key]
                anchor = anchors[int(rng.integers(len(anchors)))] if anchors else pools
                pos = [int(rng.choice(anchor[1]))]
                neg = [int(rng.choice(anchor[0]))]
                for y, selected in ((1, pos), (0, neg)):
                    remaining = [i for i in pools[y] if i not in selected]
                    selected.append(int(rng.choice(remaining if remaining else pools[y])))
                batch.extend(pos + neg)
            yield batch

    def audit(self):
        levels = {}
        for level, name in ((0, 'action'), (1, 'composition')):
            covered = set(i for k in self.eligible[level] for y in (0, 1) for i in self.pools[level][k][y])
            levels[name] = {
                'total_groups': len(self.pools[level]), 'eligible_groups': len(self.eligible[level]),
                'eligible_rows': len(covered), 'unpaired_rows': len(self.rows) - len(covered),
                'unpaired_qids': [r['qid'] for i, r in enumerate(self.rows) if i not in covered],
                'groups': [{'group': serialize_group(k), 'n_positive': len(v[1]), 'n_negative': len(v[0]), 'eligible': k in self.eligible[level]} for k,v in sorted(self.pools[level].items())],
            }
        return {'batch_size':16, 'groups_per_batch':4, 'positive_per_group':2, 'negative_per_group':2,
                'epoch_batches':len(self), 'alternating_levels':['action','composition'], 'levels':levels}


def eligible_validation_groups(train_rows, val_rows, min_per_class=5):
    result = {}
    for level, name in ((0,'action'),(1,'composition')):
        tp, vp = group_pools(train_rows, level), group_pools(val_rows, level)
        keys = [k for k,v in vp.items() if min(len(v[0]),len(v[1])) >= min_per_class and k in tp and tp[k][0] and tp[k][1]]
        if not keys:
            raise ValueError(f'No eligible Seen-val {name} groups with >= {min_per_class} per label')
        result[name] = [serialize_group(k) for k in sorted(keys)]
    return result


def group_auroc(rows, predictions, eligible=None):
    by_id = {str(p['qid']):p for p in predictions}
    result, selection_values = {}, []
    for level, name in ((0,'action'),(1,'composition')):
        pools = group_pools(rows, level)
        allowed = None if eligible is None else {tuple(k) if isinstance(k,list) else k for k in eligible[name]}
        items=[]
        for key, buckets in sorted(pools.items()):
            y=[label(rows[i]) for i in buckets[0]+buckets[1]]
            scores=[float(by_id[str(rows[i]['qid'])]['pred_exist_logit']) for i in buckets[0]+buckets[1]]
            auc=float(roc_auc_score(y,scores)) if buckets[0] and buckets[1] else None
            selected=auc is not None and (allowed is None or key in allowed)
            if selected: selection_values.append(auc)
            items.append({'group':serialize_group(key), 'n_positive':len(buckets[1]),'n_negative':len(buckets[0]),'auroc':auc,'selection_eligible':selected})
        values=[d['auroc'] for d in items if d['selection_eligible']]
        result[name]={'groups':items, 'worst_auroc':min(values) if values else None,'macro_auroc':float(np.mean(values)) if values else None,'eligible_group_count':len(values)}
        if allowed is not None and len(values)!=len(allowed): raise ValueError(f'Frozen {name} validation groups missing predictions')
    result['worst_semantic_auroc']=min(selection_values) if selection_values else None
    return result
