"""Cartesian train reference, exact leave-self-out centers and video null."""
from __future__ import annotations
import numpy as np
from core import auc, labels


class Evidence:
    def __init__(self, rows, source, methods):
        self.rows, self.methods = rows, methods
        self.train_videos = sorted({str(r['vid']) for r in rows['inner_train']})
        self.train_queries = sorted({r['query'] for r in rows['inner_train']})
        all_rows = [r for part in rows.values() for r in part]
        query_values = source.unique_queries(all_rows)
        self.queries = self.train_queries + sorted(set(query_values) - set(self.train_queries))
        self.videos = self.train_videos + sorted({str(r['vid']) for r in all_rows} - set(self.train_videos))
        self.vi = {v: i for i, v in enumerate(self.videos)}
        self.qi = {q: i for i, q in enumerate(self.queries)}
        query_matrix = np.stack([query_values[q] for q in self.queries]).T
        self.matrices = {m: np.empty((len(self.videos), len(self.queries)), dtype=np.float64) for m in methods}
        # Bound matrix-multiplication temporaries, without approximate references.
        for i, vid in enumerate(self.videos):
            video = source.video(vid)
            for start in range(0, len(self.queries), 512):
                stop = min(start + 512, len(self.queries))
                sim = video @ query_matrix[:, start:stop]
                for method in methods:
                    if method == 'max':
                        result = sim.max(0)
                    elif method == 'mean':
                        result = sim.mean(0)
                    elif method == 'top3':
                        k = min(3, len(sim))
                        result = np.partition(sim, len(sim) - k, axis=0)[-k:].mean(0)
                    else:
                        raise ValueError(method)
                    self.matrices[method][i, start:stop] = result
            if i % 500 == 0:
                print(f'Cartesian evidence: {i + 1}/{len(self.videos)} videos', flush=True)
        self.nv, self.nq = len(self.train_videos), len(self.train_queries)
        if min(self.nv, self.nq) < 2:
            raise ValueError('Leave-self-out reference requires at least two videos and queries')
        self.sums = {m: (a[:self.nv].sum(0), a[:, :self.nq].sum(1), a[:self.nv, :self.nq].sum())
                     for m, a in self.matrices.items()}

    def mapping(self, seed):
        rng = np.random.default_rng(seed)
        result = {}
        for role, rows in self.rows.items():
            videos = sorted({str(r['vid']) for r in rows})
            if len(videos) < 2:
                raise ValueError('Cannot derange single-video role')
            order = rng.permutation(videos)
            result[role] = dict(zip(order, np.roll(order, int(rng.integers(1, len(videos))))))
        return result

    def columns(self, method, mapping=None, centering='double'):
        a = self.matrices[method]
        column_sum, row_sum, total = self.sums[method]
        result, raw = {}, {}
        for role, rows in self.rows.items():
            cols, zs = [], []
            for row in rows:
                recipient, query = str(row['vid']), row['query']
                donor = recipient if mapping is None else mapping[role][recipient]
                i, j = self.vi[donor], self.qi[query]
                z = float(a[i, j])
                # Under shuffled input train is a bijection: remove the donor of
                # the recipient, rather than the original feature row.
                ev = int(recipient in self.vi and self.vi[recipient] < self.nv)
                if mapping is not None and ev:
                    # The train donor determines the leave-self-out reference,
                    # even for an overlapping eval recipient (normally none).
                    ref_donor = mapping['inner_train'][recipient]
                    excluded_i = self.vi[ref_donor]
                else:
                    excluded_i = self.vi[recipient] if ev else -1
                eq = int(j < self.nq)
                muq = (column_sum[j] - (a[excluded_i, j] if ev else 0)) / (self.nv - ev)
                muv = (row_sum[i] - (a[i, j] if eq else 0)) / (self.nq - eq)
                mu = (total - (row_sum[excluded_i] if ev else 0) - (column_sum[j] if eq else 0)
                      + (a[excluded_i, j] if ev and eq else 0)) / ((self.nv - ev) * (self.nq - eq))
                residual = z - muq - muv + mu if centering == 'double' else (
                    z - muq if centering == 'query' else z - muv if centering == 'video' else z)
                cols.append((muq, muv, residual))
                zs.append(z)
            result[role] = np.asarray(cols)
            raw[role] = np.asarray(zs)
        return result, raw

    def select(self):
        # Only inner_train labels enter selection.
        values = {}
        for method in self.methods:
            _, raw = self.columns(method)
            values[method] = auc(labels(self.rows['inner_train']), raw['inner_train'])
        if any(v is None for v in values.values()):
            raise ValueError('Aggregator selection requires both train classes')
        return max(self.methods, key=lambda m: values[m]), values
