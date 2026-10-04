#!/usr/bin/env python3
"""PEI-0 Stage 0. Read only strict-inner annotations/features; never train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA.parents[1]
ROLES = ('inner_train', 'inner_seen_val', 'inner_novel_dev')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, len(scores)]
    ranks = np.empty(len(scores), dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def incidence(rows):
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            node, parent[node] = parent[node], root
        return root

    cells, queries = defaultdict(set), defaultdict(list)
    for row in rows:
        v, q = ('v', str(row['vid'])), ('q', row['query'])
        parent[find(v)] = find(q)
        cells[(v[1], q[1])].add(int(row['exist_label']))
        queries[q[1]].append(row)
    sizes = Counter(find(node) for node in parent)
    dual = {q: rr for q, rr in queries.items() if {int(r['exist_label']) for r in rr} == {0, 1}}
    components = {find(('q', q)) for q in dual}
    # Count only already labelled closed 2x2 cells, without adding negatives.
    # A query pair is counted once; every disjoint opposite video pairing is unique.
    signatures = defaultdict(lambda: [set(), set()])
    by_video = defaultdict(dict)
    for (v, q), labels in cells.items():
        if len(labels) == 1:
            by_video[v][q] = next(iter(labels))
    for v, labels in by_video.items():
        positives = sorted(q for q, y in labels.items() if y == 1 and q in dual)
        negatives = sorted(q for q, y in labels.items() if y == 0 and q in dual)
        for p, n in itertools.product(positives, negatives):
            key = tuple(sorted((p, n)))
            signatures[key][0 if p == key[0] else 1].add(v)
    closed = sum(len(a) * len(b) for a, b in signatures.values())
    return {
        'rows': len(rows), 'labels': dict(Counter(str(r['exist_label']) for r in rows)),
        'unique_videos': len(by_video), 'unique_exact_queries': len(queries),
        'dual_label_queries': len(dual),
        'same_query_positive_negative_row_pairs': sum(
            sum(r['exist_label'] == 1 for r in rr) * sum(r['exist_label'] == 0 for r in rr)
            for rr in dual.values()),
        'connected_components': len(sizes), 'dual_query_components': len(components),
        'component_node_sizes_descending': sorted(sizes.values(), reverse=True),
        'contradictory_video_query_cells': sum(len(y) > 1 for y in cells.values()),
        'duplicate_video_query_label_rows': len(rows) - sum(len(y) for y in cells.values()),
        'closed_labelled_quartets': closed,
        'construction_type': dict(Counter(str(r.get('construction_type', '<missing>')) for r in rows)),
        'verification_status_by_label': {
            str(y): dict(Counter(str(r.get('verification_status', '<missing>')) for r in rows if r['exist_label'] == y))
            for y in (0, 1)},
        'semantic_groups': dict(Counter(
            str((r.get('semantic_graph') or {}).get('action_base', '<missing>')) + '::' +
            str((r.get('semantic_graph') or {}).get('object_concept', '<missing>')) for r in rows)),
    }


def source_pairs(rows):
    lookup = {str(r['qid']): r for r in rows}
    counts = Counter()
    for row in rows:
        if row['exist_label'] != 0:
            continue
        source = lookup.get(str(row.get('source_qid')))
        reason = ('source_missing' if source is None else
                  'source_not_positive' if source['exist_label'] != 1 else
                  'video_mismatch' if source['vid'] != row['vid'] else
                  'identical_query' if source['query'] == row['query'] else
                  'source_query_mismatch' if row.get('source_query') and row['source_query'] != source['query'] else
                  'unsupported_absence_provenance' if not str(row.get('verification_status', '')).startswith(
                      ('user_attested', 'human', 'verified', 'reviewed')) else 'eligible')
        counts[reason] += 1
    return dict(counts)


def normalize(array):
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite features')
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError('Zero feature vector')
    return array / norm


def cluster_mean_ci(values, clusters, repetitions, rng):
    groups = defaultdict(list)
    for value, cluster in zip(values, clusters):
        groups[cluster].append(value)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    totals = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    boot = []
    for _ in range(repetitions):
        ix = rng.integers(0, len(keys), len(keys))
        boot.append(float(totals[ix].sum() / counts[ix].sum()))
    return np.quantile(boot, [.025, .975]).tolist()


def feature_audit(fold, rows, args, freeze):
    """Feature diagnostics use inner_train only; nominal times never pass the gate."""
    split = freeze['folds'][fold]['parent_split']
    text_root = args.text_base / split / 'clip_text'
    hashes, failures, text_hashes = {}, [], defaultdict(set)

    @lru_cache(maxsize=None)
    def video(vid):
        path = args.clip_root / (vid + '.npz')
        hashes[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as archive:
            features = normalize(archive['features'])
            if features.ndim != 2 or features.shape[1] != 512:
                raise ValueError(f'Unexpected video feature shape: {path}')
            return features

    queries, usable = [], []
    alignment, alignment_vids = [], []
    timestamp_map = {}
    if args.timestamps is not None:
        timestamp_map = json.loads(args.timestamps.read_text())
        if not timestamp_map.get('provenance'):
            raise ValueError('Timestamp map must declare extraction provenance')
    for row in rows:
        try:
            path = text_root / ('qid' + str(row['qid']) + '.npz')
            hashes[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                tokens = normalize(archive['last_hidden_state'])[:freeze['features']['max_query_tokens']]
            if tokens.ndim != 2 or tokens.shape[1] != 512 or not len(tokens):
                raise ValueError(f'Unexpected text feature shape: {path}')
            text_hashes[row['query']].add(hashlib.sha256(tokens.tobytes()).hexdigest())
            q = normalize(tokens.mean(axis=0))
            vv = video(str(row['vid']))
            queries.append(q)
            usable.append(row)
            if row['exist_label'] == 1:
                explicit = timestamp_map.get('videos', {}).get(str(row['vid']))
                if args.timestamps is not None and explicit is None:
                    raise ValueError('Missing extraction timestamps for positive video')
                times = np.asarray(explicit, dtype=float) if explicit is not None else (
                    np.arange(len(vv)) + .5) * freeze['features']['nominal_clip_seconds']
                if times.shape != (len(vv),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                    raise ValueError('Invalid clip center timestamps')
                inside = np.zeros(len(vv), dtype=bool)
                for start, end in row.get('relevant_windows', []):
                    inside |= (times >= start) & (times <= end)
                a = auc(inside.astype(int), vv @ q)
                if a is not None:
                    alignment.append(a)
                    alignment_vids.append(str(row['vid']))
        except (OSError, KeyError, ValueError) as exc:
            failures.append({'qid': str(row['qid']), 'reason': str(exc)})
    complete = not failures and len(usable) == len(rows)
    report = {'input_rows': len(rows), 'usable_rows': len(usable), 'failures': failures,
              'feature_sha256': hashes,
              'exact_queries_with_different_encoded_tokens': sum(len(h) > 1 for h in text_hashes.values()),
              'alignment_positive_rows_with_inside_and_outside': len(alignment),
              'alignment_row_macro_auc': float(np.mean(alignment)) if alignment else None,
              'alignment_video_cluster_ci95': cluster_mean_ci(alignment, alignment_vids,
                  freeze['statistics']['bootstrap_repetitions'], np.random.default_rng(freeze['seed'])),
              'time_mapping': 'extraction_clip_centers' if args.timestamps else 'nominal_centers_diagnostic_only',
              'timestamps_sha256': digest(args.timestamps) if args.timestamps else None,
              'timestamp_provenance': timestamp_map.get('provenance'),
              'alignment_gate': 'pending'}
    ci = report['alignment_video_cluster_ci95']
    if complete and args.timestamps and ci:
        report['alignment_gate'] = 'pass' if ci[0] > .5 else 'not_passed'
    if not complete:
        report['alignment_gate'] = 'blocked_feature_or_timestamp_errors'
    if not complete or len(usable) < 2:
        report['shuffle_null'] = {'status': 'blocked_incomplete_inputs'}
        return report
    # Different videos: circular shifts of a random unique-video ordering are
    # bijective derangements. Broadcast each mapping to all queries of a video.
    vids = sorted({str(r['vid']) for r in usable})
    if len(vids) < 2:
        report['shuffle_null'] = {'status': 'blocked_single_video'}
        return report
    labels = [r['exist_label'] for r in usable]
    rng = np.random.default_rng(freeze['seed'])
    null = {method: [] for method in freeze['aggregators']}
    def aggregate(sim, method):
        return float(sim.max() if method == 'max' else sim.mean() if method == 'mean' else
                     np.sort(sim)[-min(3, len(sim)):].mean())
    # Compute null before real evidence scores. These AUCs are diagnostics,
    # not null values for the later, refitted M3-minus-M2 statistic.
    for _ in range(freeze['statistics']['shuffle_repetitions']):
        order = rng.permutation(vids)
        shifted = np.roll(order, int(rng.integers(1, len(order))))
        mapping = dict(zip(order, shifted))
        scores = {m: [] for m in null}
        for row, q in zip(usable, queries):
            sim = video(mapping[str(row['vid'])]) @ q
            for method in scores:
                scores[method].append(aggregate(sim, method))
        for method in null:
            null[method].append(auc(labels, scores[method]))
    report['shuffle_null'] = {
        'status': 'diagnostic_only', 'assignment': 'unique_video_bijection_no_fixed_points',
        'pooled_auc_draws': null, 'm3_minus_m2_mde': None,
        'mde_status': 'requires_crossfit_Cq_and_full_stage2_refit_with_shared_video_resampling'}
    real = {m: [] for m in null}
    for row, q in zip(usable, queries):
        sim = video(str(row['vid'])) @ q
        for method in real:
            real[method].append(aggregate(sim, method))
    report['inner_train_raw_pooled_auc'] = {m: auc(labels, s) for m, s in real.items()}
    report['aggregator_selection'] = max(real, key=lambda m: report['inner_train_raw_pooled_auc'][m])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=IDEA / 'records/FREEZE.json')
    parser.add_argument('--features', action='store_true', help='Also compute train-only alignment and shuffle diagnostics')
    parser.add_argument('--clip-root', type=Path, default=Path('/home/guoxiangyu/paper/新建文件夹/charades/vid_clip'))
    parser.add_argument('--text-base', type=Path, default=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/features/semantic_existence_v2'))
    parser.add_argument('--timestamps', type=Path, help='JSON: provenance plus videos mapping vid to extraction clip centers in seconds')
    parser.add_argument('--output', type=Path, default=IDEA / 'records/STAGE0_AUDIT.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(IDEA.resolve()):
        parser.error('Output must remain inside direction 09')
    if args.output.exists():
        parser.error('Output already exists; choose a new filename to preserve prior records')
    freeze = json.loads(args.freeze.read_text())
    report = {'stage': 'PEI-0_stage0', 'formal_u_read': False,
              'freeze_sha256': digest(args.freeze), 'script_sha256': digest(__file__), 'folds': {}}
    for fold, spec in freeze['folds'].items():
        if fold not in {'A1_action_01', 'A1_action_02', 'C1_composition_01', 'C1_composition_02'}:
            raise ValueError('Disallowed fold')
        directory = ROOT / 'experiments/moment_detr_gmr_evidence_v5/inner_folds' / fold
        manifest_path = directory / 'manifest.json'
        if digest(manifest_path) != spec['manifest_sha256']:
            raise ValueError(f'Manifest changed: {fold}')
        manifest = json.loads(manifest_path.read_text())
        parts, audit = {}, {}
        for role in ROLES:
            path = directory / (role + '.jsonl')
            if digest(path) != manifest['files'][role]['sha256']:
                raise ValueError(f'Annotation changed: {fold}/{role}')
            rr = read_jsonl(path)
            # Strict-inner Novel also has original S partitions; never permit U.
            if any(r.get('partition') not in {'S+', 'S-'} or r.get('exist_label') not in (0, 1) or
                   r['partition'] != ('S+' if r['exist_label'] == 1 else 'S-') for r in rr):
                raise ValueError(f'Forbidden partition or invalid label: {fold}/{role}')
            if len({str(r['qid']) for r in rr}) != len(rr):
                raise ValueError('Duplicate qid')
            parts[role] = rr
            audit[role] = dict(incidence(rr), source_pairs=source_pairs(rr), annotation_sha256=digest(path))
        audit['role_overlap'] = {}
        for a, b in itertools.combinations(ROLES, 2):
            audit['role_overlap'][a + '__' + b] = {
                'videos': len({r['vid'] for r in parts[a]} & {r['vid'] for r in parts[b]}),
                'exact_queries': len({r['query'] for r in parts[a]} & {r['query'] for r in parts[b]}),
                'qids': len({str(r['qid']) for r in parts[a]} & {str(r['qid']) for r in parts[b]})}
        audit['seen_train_plus_val'] = incidence(parts['inner_train'] + parts['inner_seen_val'])
        if args.features:
            audit['train_features'] = feature_audit(fold, parts['inner_train'], args, freeze)
        report['folds'][fold] = audit
        print(f'{fold}: annotation audit complete', flush=True)
    all_videos = {}
    for fold in freeze['folds']:
        all_videos[fold] = {r['vid'] for role in ROLES for r in read_jsonl(
            ROOT / 'experiments/moment_detr_gmr_evidence_v5/inner_folds' / fold / (role + '.jsonl'))}
    report['cross_fold_shared_videos'] = {a + '__' + b: len(all_videos[a] & all_videos[b])
                                         for a, b in itertools.combinations(all_videos, 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
