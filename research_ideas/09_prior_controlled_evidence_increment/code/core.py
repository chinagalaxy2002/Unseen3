"""Local PEI readers, fixed-offset regression and immutable artifacts."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from audit_stage0 import IDEA, ROOT, ROLES, digest, read_jsonl, normalize

FOLDS = ('A1_action_01', 'A1_action_02', 'C1_composition_01', 'C1_composition_02')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_freeze(path):
    freeze = json.loads(Path(path).read_text())
    if set(freeze['folds']) != set(FOLDS):
        raise ValueError('Exactly the four registered strict-inner folds are required')
    for name, checksum in freeze['code_sha256'].items():
        if digest(IDEA / name) != checksum:
            raise ValueError(f'Frozen code changed: {name}; register a new revision')
    return freeze


def load_rows(fold, role, freeze):
    if fold not in FOLDS or role not in ROLES:
        raise ValueError('Disallowed fold/role')
    directory = ROOT / 'experiments/moment_detr_gmr_evidence_v5/inner_folds' / fold
    manifest_path = directory / 'manifest.json'
    if digest(manifest_path) != freeze['folds'][fold]['manifest_sha256']:
        raise ValueError(f'Manifest changed: {fold}')
    manifest = json.loads(manifest_path.read_text())
    path = directory / (role + '.jsonl')
    if digest(path) != manifest['files'][role]['sha256']:
        raise ValueError(f'Annotation changed: {fold}/{role}')
    rows = read_jsonl(path)
    if len({str(r['qid']) for r in rows}) != len(rows):
        raise ValueError('Duplicate qid')
    for row in rows:
        if row.get('exist_label') not in (0, 1) or row.get('partition') != ('S+' if row['exist_label'] else 'S-'):
            raise ValueError('Forbidden partition or inconsistent label')
    return rows


def labels(rows):
    return np.asarray([r['exist_label'] for r in rows], dtype=int)


def auc(y, s, weights=None):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite score')
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if any(weights[y == label].sum() <= 0 for label in (0, 1)):
        return None
    return float(roc_auc_score(y, s, sample_weight=weights))


class Features:
    def __init__(self, clip_root, text_base, split, max_tokens=32):
        self.clip_root = Path(clip_root)
        self.text_root = Path(text_base) / split / 'clip_text'
        self.max_tokens = max_tokens
        self.hashes, self.videos, self.tokens = {}, {}, {}

    def video(self, vid):
        vid = str(vid)
        if vid not in self.videos:
            path = self.clip_root / (vid + '.npz')
            self.hashes[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                value = normalize(archive['features'])
            if value.ndim != 2 or value.shape[1] != 512 or not len(value):
                raise ValueError(f'Invalid CLIP video: {vid}')
            self.videos[vid] = value.astype(np.float32)
        return self.videos[vid]

    def query(self, row):
        qid = str(row['qid'])
        if qid not in self.tokens:
            path = self.text_root / ('qid' + qid + '.npz')
            self.hashes[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                value = normalize(archive['last_hidden_state'])[:self.max_tokens]
            if value.ndim != 2 or value.shape[1] != 512 or not len(value):
                raise ValueError(f'Invalid text tokens: {qid}')
            self.tokens[qid] = value.astype(np.float32)
        return self.tokens[qid]

    def means(self, rows):
        return np.asarray([self.query(r).mean(0) for r in rows], dtype=np.float32)

    def unique_queries(self, rows):
        values, token_identity = {}, {}
        for row in rows:
            text = row['query']
            tokens = self.query(row)
            if text in token_identity and not np.array_equal(tokens, token_identity[text]):
                raise ValueError(f'Exact text has different encoded tokens: {text!r}')
            token_identity[text] = tokens
            values[text] = normalize(tokens.mean(0)).astype(np.float32)
        return values


class Standardizer:
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.active = self.std > 1e-12
        self.std = np.where(self.active, self.std, 1)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.std * self.active

    def record(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(), 'active': self.active.tolist()}


class OffsetLogistic:
    """Balanced mean BCE + ridge*||slopes||²/2, unpenalized intercept."""
    def fit(self, x, y, offset, ridge=1e-4):
        x, y, offset = np.asarray(x, float), np.asarray(y, int), np.asarray(offset, float)
        if set(y) != {0, 1}:
            raise ValueError('Logistic fit needs both classes')
        design = np.column_stack((np.ones(len(x)), x))
        weights = np.where(y == 1, .5 / (y == 1).sum(), .5 / (y == 0).sum())
        def objective(coef):
            logits = offset + design @ coef
            loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) + ridge * np.dot(coef[1:], coef[1:]) / 2
            gradient = design.T @ (weights * (expit(logits) - y))
            gradient[1:] += ridge * coef[1:]
            return loss, gradient
        fit = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f'Logistic optimization failed: {fit.message}')
        self.coef = fit.x
        return self

    def predict(self, x, offset):
        return np.asarray(offset) + self.coef[0] + np.asarray(x) @ self.coef[1:]


def fit_scores(features, b, y):
    scale = Standardizer().fit(features['inner_train'])
    x = {r: scale.transform(v) for r, v in features.items()}
    m2 = OffsetLogistic().fit(x['inner_train'][:, :2], y, b['inner_train'])
    m3 = OffsetLogistic().fit(x['inner_train'], y, b['inner_train'])
    zero_scale = Standardizer().fit(np.column_stack((b['inner_train'], features['inner_train'])))
    scores = {}
    for role in features:
        equal = zero_scale.transform(np.column_stack((b[role], features[role])))
        scores[role] = {'M0': b[role], 'M2': m2.predict(x[role][:, :2], b[role]),
                        'M3': m3.predict(x[role], b[role]),
                        'zero_M2': equal[:, :3].sum(1), 'zero_M3': equal.sum(1)}
    record = {'standardizer': scale.record(), 'M2_coef': m2.coef.tolist(), 'M3_coef': m3.coef.tolist(),
              'zero_standardizer': zero_scale.record()}
    return scores, record, x
