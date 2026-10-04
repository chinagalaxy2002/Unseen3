#!/usr/bin/env python3
"""Run registered PEI-0 stages. Full experiment is opt-in, outputs stay local."""
from __future__ import annotations
import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from core import (IDEA, ROOT, FOLDS, ROLES, Features, Standardizer, auc, digest, fit_scores,
                  labels, load_freeze, load_rows, read_jsonl, write_json)
from crossfit import train_crossfit, predict_crossfit
from evidence import Evidence
from analysis_stats import decide, metrics, power_simulation, shared_macro_interval


def validate_gate(path, freeze_path, freeze):
    audit = json.loads(path.read_text())
    if audit.get('formal_u_read') is not False or audit['freeze_sha256'] != digest(freeze_path):
        raise ValueError('Stage 0 audit must match current freeze and exclude formal U')
    if audit['script_sha256'] != freeze['code_sha256']['code/audit_stage0.py']:
        raise ValueError('Stage 0 script version mismatch')
    for fold in FOLDS:
        feature = audit['folds'][fold].get('train_features', {})
        if feature.get('alignment_gate') != 'pass' or feature.get('exact_queries_with_different_encoded_tokens') != 0:
            raise ValueError(f'{fold}: alignment/encoding gate has not passed; obtain timestamp provenance or register revised features')
        for name, checksum in feature['feature_sha256'].items():
            if digest(name) != checksum:
                raise ValueError(f'Audited feature changed: {name}')
    return audit


def replay_cv(fold, rows, freeze):
    policy = freeze['stage2']['cv_replay'][fold]
    base = ROOT / 'results/moment_detr_gmr_evidence_v5/readouts' / fold / 'Cv_P3_bce_seed3407'
    path, status_path = base / 'predictions.jsonl', base / 'status.json'
    if digest(path) != policy['predictions_sha256'] or digest(status_path) != policy['status_sha256']:
        raise ValueError('Registered Cv replay changed')
    status = json.loads(status_path.read_text())
    if status['variant'] != 'Cv' or status['fold'] != fold or status['status'] != 'evaluated':
        raise ValueError('Cv artifact identity mismatch')
    predictions = read_jsonl(path)
    if any(r['role'] not in {'inner_seen_val', 'inner_novel_dev'} for r in predictions):
        raise ValueError('Cv replay contains disallowed roles')
    index = {(r['role'], str(r['qid'])): r for r in predictions}
    if len(index) != len(predictions):
        raise ValueError('Duplicate Cv replay rows')
    output = {}
    for role in ('inner_seen_val', 'inner_novel_dev'):
        scores = []
        for row in rows[role]:
            pred = index[(role, str(row['qid']))]
            if str(row['vid']) != str(pred['vid']) or row['exist_label'] != pred['exist_label']:
                raise ValueError('Cv replay row identity mismatch')
            scores.append(pred['learned_logit'])
        output[role] = np.asarray(scores)
        by_video = {}
        for row, score in zip(rows[role], output[role]):
            v = str(row['vid'])
            if v in by_video and score != by_video[v]:
                raise ValueError('Cv replay is not video-only')
            by_video[v] = score
    if len(predictions) != len(rows['inner_seen_val']) + len(rows['inner_novel_dev']):
        raise ValueError('Cv replay has extra rows')
    return output, {'prediction_sha256': digest(path), 'status_sha256': digest(status_path),
                    'score': 'learned_logit; never baseline_fallback',
                    'limitation': 'historical single-seed Seen-val-selected Cv; not matched crossfit recipe'}


def same_run_record(path, record):
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(f'Run identity differs: {path}; choose new --run-id')
    else:
        write_json(path, record)


def run(args):
    freeze = load_freeze(args.freeze)
    audit = validate_gate(args.audit, args.freeze, freeze)
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('run-id must use letters/digits/underscore/hyphen')
    output = IDEA / 'runs' / args.run_id
    if (output / 'RESULTS.json').exists():
        raise ValueError('Completed run exists; choose new --run-id')
    identity = {'freeze_sha256': digest(args.freeze), 'stage0_audit_sha256': digest(args.audit),
                'stage': args.stage, 'clip_root': str(args.clip_root.resolve()), 'text_base': str(args.text_base.resolve()),
                'device': args.device, 'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
                'formal_u_read': False, 'blas_threads': freeze['runtime']['blas_threads']}
    same_run_record(output / 'RUN_IDENTITY.json', identity)
    started = time.monotonic()
    datasets = {}
    for fold in FOLDS:
        fold_dir = output / fold
        train_rows = load_rows(fold, 'inner_train', freeze)
        source = Features(args.clip_root, args.text_base, freeze['folds'][fold]['parent_split'],
                          freeze['features']['max_query_tokens'])
        source.unique_queries(train_rows)
        means = source.means(train_rows)
        b = {}
        if args.stage == 'full':
            # No Novel/Seen-val file is read by the Cq training function.
            b['inner_train'] = train_crossfit(train_rows, means, fold_dir / 'cq', freeze, identity['freeze_sha256'], args.device)
        rows = {'inner_train': train_rows}
        for role in ('inner_seen_val', 'inner_novel_dev'):
            rows[role] = load_rows(fold, role, freeze)
            if args.stage == 'full':
                b[role] = predict_crossfit(source.means(rows[role]), fold_dir / 'cq', freeze, identity['freeze_sha256'], args.device)
        for a in ROLES:
            for b_role in ROLES:
                if a != b_role and {r['vid'] for r in rows[a]} & {r['vid'] for r in rows[b_role]}:
                    raise ValueError('Cross-role video overlap violates strict OOF/eval isolation')
        evidence = Evidence(rows, source, freeze['aggregators'])
        audited_hashes = audit['folds'][fold]['train_features']['feature_sha256']
        for name, checksum in audited_hashes.items():
            if source.hashes.get(name) != checksum:
                raise ValueError('Pipeline input roots do not reproduce all audited train inputs')
        selected, selection_auc = evidence.select()
        features, raw = evidence.columns(selected)
        cv, cv_record = replay_cv(fold, rows, freeze)
        datasets[fold] = {'rows': rows, 'b': b, 'evidence': evidence, 'selected': selected,
                          'features': features, 'raw': raw, 'source_hashes': source.hashes,
                          'selection_auc': selection_auc, 'cv': cv, 'cv_record': cv_record}
        # Persist full precision inputs, rather than decoder-dependent features.
        for role in ROLES:
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez(fold_dir / (role + '_evidence.npz'), columns=features[role], raw=raw[role],
                     b=b.get(role, np.empty(0)))
        same_run_record(fold_dir / 'INPUTS.json', {'selected_aggregator': selected, 'train_selection_auc': selection_auc,
                         'feature_sha256': source.hashes, 'cv_replay': cv_record})
        print(f'{fold}: evidence and input identities ready', flush=True)
    null_draws = []
    # Calculate the same primary refitted null statistic before real M3 deltas.
    for repeat in range(freeze['statistics']['shuffle_repetitions']):
        delta, stage1_null = [], {}
        for fold, ds in datasets.items():
            mapping = ds['evidence'].mapping(freeze['seed'] + 100000 + repeat)
            columns, raw = ds['evidence'].columns(ds['selected'], mapping)
            if args.stage == 'full':
                scores, _, _ = fit_scores(columns, ds['b'], labels(ds['rows']['inner_train']))
                y = labels(ds['rows']['inner_novel_dev'])
                delta.append(auc(y, scores['inner_novel_dev']['M3']) - auc(y, scores['inner_novel_dev']['M2']))
            else:
                stage1_null[fold] = {role: auc(labels(ds['rows'][role]), raw[role]) for role in ROLES}
        null_draws.append(float(np.mean(delta)) if delta else stage1_null)
        if repeat % 10 == 0:
            print(f'Shuffle-refit null {repeat + 1}/{freeze["statistics"]["shuffle_repetitions"]}', flush=True)
    same_run_record(output / 'SHUFFLE_NULL.json', {'stage': args.stage, 'draws': null_draws})
    # Model/feature-dependent power simulation precedes real Novel increments.
    if args.stage == 'full':
        for ds in datasets.values():
            scale = Standardizer().fit(ds['features']['inner_train'])
            ds['x'] = {role: scale.transform(x) for role, x in ds['features'].items()}
        power_path = output / 'CONDITIONAL_POWER.json'
        if power_path.exists():
            power = json.loads(power_path.read_text())
        else:
            power = power_simulation(datasets, freeze)
            write_json(power_path, power)
    else:
        power = {'status': 'not_applicable_stage1', 'mde_auc': None}
    report = {'identity': identity, 'folds': {}, 'power': power,
              'limitations': ['inner Novel-dev is a previously used development set',
                  'four folds share videos', 'train Cq OOF predictions can differ for exact queries across outer models',
                  'conditional MDE assumes fixed Cq/features and specified latent effects']}
    novel_rows, novel_scores, seen_rows, seen_scores, deltas, within_ci = {}, {}, {}, {}, [], []
    for fold, ds in datasets.items():
        rows = ds['rows']
        fold_report = {'selected_aggregator': ds['selected'], 'train_selection_auc': ds['selection_auc'],
                       'stage1': {}, 'Cv': {}, 'controls': {}}
        for method in freeze['aggregators']:
            columns, raw = ds['evidence'].columns(method)
            fold_report['stage1'][method] = {
                role: {'raw': metrics(rows[role], raw[role], freeze),
                       'double_centered': metrics(rows[role], columns[role][:, 2], freeze)} for role in ROLES}
            # Descriptive Seen union; train labels selected this aggregator.
            union_rows = rows['inner_train'] + rows['inner_seen_val']
            fold_report['stage1'][method]['seen_train_plus_val'] = metrics(union_rows,
                np.concatenate((columns['inner_train'][:, 2], columns['inner_seen_val'][:, 2])), freeze)
        for role, scores in ds['cv'].items():
            fold_report['Cv'][role] = metrics(rows[role], scores, freeze)
        if args.stage == 'full':
            scores, fit, _ = fit_scores(ds['features'], ds['b'], labels(rows['inner_train']))
            fold_report['stage2'] = {role: {name: metrics(rows[role], score, freeze) for name, score in by_name.items()}
                                     for role, by_name in scores.items()}
            fold_report['stage2_fit'] = fit
            fold_report['cq_log_loss'] = {role: float(np.mean(np.logaddexp(0, ds['b'][role]) - labels(rows[role]) * ds['b'][role]))
                                           for role in ROLES}
            for method in freeze['aggregators']:
                for centering in ('double', 'query', 'video'):
                    columns, _ = ds['evidence'].columns(method, centering=centering)
                    ctrl, _, _ = fit_scores(columns, ds['b'], labels(rows['inner_train']))
                    fold_report['controls'][method + '_' + centering] = {
                        role: {name: auc(labels(rows[role]), ctrl[role][name]) for name in ('M2', 'M3')}
                        for role in ('inner_seen_val', 'inner_novel_dev')}
            novel_rows[fold], novel_scores[fold] = rows['inner_novel_dev'], scores['inner_novel_dev']
            seen_rows[fold], seen_scores[fold] = rows['inner_seen_val'], scores['inner_seen_val']
            y = labels(rows['inner_novel_dev'])
            delta = auc(y, scores['inner_novel_dev']['M3']) - auc(y, scores['inner_novel_dev']['M2'])
            deltas.append(delta)
            within_ci.append(fold_report['stage1'][ds['selected']]['inner_novel_dev']['double_centered']['within_query']['ci95'])
            fold_report['novel_delta'] = delta
            predictions_path = output / fold / 'PREDICTIONS.jsonl'
            if not predictions_path.exists():
                with predictions_path.open('x', encoding='utf-8') as stream:
                    for role, rr in rows.items():
                        for i, row in enumerate(rr):
                            record = {'role': role, 'qid': row['qid'], 'vid': row['vid'], 'query': row['query'],
                                      'exist_label': row['exist_label'],
                                      'scores': {name: float(value[i]) for name, value in scores[role].items()}}
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        report['folds'][fold] = fold_report
    if args.stage == 'full':
        ci = shared_macro_interval(novel_rows, novel_scores, freeze['statistics']['bootstrap_repetitions'], freeze['seed'])
        seen_ci = shared_macro_interval(seen_rows, seen_scores, freeze['statistics']['bootstrap_repetitions'], freeze['seed'])
        delta, positive, null_upper = float(np.mean(deltas)), int(sum(v > 0 for v in deltas)), float(np.quantile(null_draws, .95))
        report['primary'] = {'macro_novel_delta': delta, 'positive_folds': positive, 'novel_ci': ci,
                             'seen_val_ci': seen_ci, 'shuffle_null_upper95': null_upper,
                             'decision': decide(delta, positive, ci['ci95'], null_upper, power, within_ci, freeze),
                             'wording': 'specified-prior-controlled evidence increment; conditional compatibility requires additional video-only comparison'}
    report['elapsed_seconds'] = time.monotonic() - started
    report['process_peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(output / 'RESULTS.json', report)
    manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'OUTPUT_SHA256.json', manifest)
    print(output / 'RESULTS.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=IDEA / 'records/FREEZE.json')
    parser.add_argument('--audit', type=Path, required=True, help='Matching Stage 0 feature audit with alignment gates passed')
    parser.add_argument('--stage', choices=('stage1', 'full'), default='stage1')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--clip-root', type=Path, default=Path('/home/guoxiangyu/paper/新建文件夹/charades/vid_clip'))
    parser.add_argument('--text-base', type=Path, default=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/features/semantic_existence_v2'))
    args = parser.parse_args()
    try:
        freeze = load_freeze(args.freeze)
        torch.set_num_threads(freeze['runtime']['torch_threads'])
        with threadpool_limits(limits=freeze['runtime']['blas_threads']):
            run(args)
    except Exception as exc:
        print(f'PEI stopped: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
