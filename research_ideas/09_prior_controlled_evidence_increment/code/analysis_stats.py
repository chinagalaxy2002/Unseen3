"""Shared-video bootstrap, component metrics and conditional power simulation."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from core import OffsetLogistic, auc, labels


def shared_macro_interval(rows, scores, repetitions, seed, left='M2', right='M3'):
    videos = sorted({str(r['vid']) for rr in rows.values() for r in rr})
    index = {v: i for i, v in enumerate(videos)}
    row_ix = {f: np.asarray([index[str(r['vid'])] for r in rr]) for f, rr in rows.items()}
    rng, draws, attempts = np.random.default_rng(seed), [], 0
    while len(draws) < repetitions and attempts < repetitions * 10:
        attempts += 1
        counts = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)))
        delta = []
        for fold in rows:
            weights = counts[row_ix[fold]]
            a = auc(labels(rows[fold]), scores[fold][left], weights)
            b = auc(labels(rows[fold]), scores[fold][right], weights)
            if a is None or b is None:
                break
            delta.append(b - a)
        if len(delta) == len(rows):
            draws.append(np.mean(delta))
    return {'unit': 'shared_video_across_folds', 'valid_draws': len(draws), 'attempts': attempts,
            'ci95': np.quantile(draws, [.025, .975]).tolist() if len(draws) == repetitions else None}


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        parent[find(('v', str(r['vid'])))] = find(('q', r['query']))
    return [find(('q', r['query'])) for r in rows]


def within_query(rows, scores, repetitions, seed, policy):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['query']].append(i)
    comp = components(rows)
    values, units = [], []
    for ix in groups.values():
        value = auc(labels([rows[i] for i in ix]), scores[ix])
        if value is not None:
            values.append(value)
            units.append(comp[ix[0]])
    counts = defaultdict(list)
    for unit, value in zip(units, values):
        counts[unit].append(value)
    largest = max((len(v) for v in counts.values()), default=0) / max(1, len(values))
    result = {'query_macro_auc': float(np.mean(values)) if values else None,
              'dual_label_queries': len(values), 'support_components': len(counts),
              'largest_support_component_fraction': largest, 'ci95': None,
              'ci_status': 'insufficient_components_or_dominant_component'}
    if len(counts) >= policy['minimum_components'] and largest <= policy['largest_component_fraction_max']:
        totals = np.asarray([sum(v) for v in counts.values()])
        ns = np.asarray([len(v) for v in counts.values()])
        rng, boot = np.random.default_rng(seed), []
        for _ in range(repetitions):
            ix = rng.integers(0, len(ns), len(ns))
            boot.append(totals[ix].sum() / ns[ix].sum())
        result.update(ci95=np.quantile(boot, [.025, .975]).tolist(), ci_status='component_bootstrap')
    return result


def pair_metrics(rows, scores, repetitions, seed):
    lookup = {str(r['qid']): i for i, r in enumerate(rows)}
    by_video = defaultdict(list)
    for i, r in enumerate(rows):
        if r['exist_label'] != 0:
            continue
        j = lookup.get(str(r.get('source_qid')))
        if (j is None or rows[j]['exist_label'] != 1 or rows[j]['vid'] != r['vid'] or
            rows[j]['query'] == r['query'] or
            (r.get('source_query') and r['source_query'] != rows[j]['query']) or
            not str(r.get('verification_status', '')).startswith(('user_attested', 'human', 'verified', 'reviewed'))):
            continue
        d = scores[j] - scores[i]
        by_video[str(r['vid'])].append(float(d > 0) + .5 * float(d == 0))
    vals = [v for vv in by_video.values() for v in vv]
    ci = None
    if len(by_video) >= 2:
        totals = np.asarray([sum(v) for v in by_video.values()])
        ns = np.asarray([len(v) for v in by_video.values()])
        rng, boot = np.random.default_rng(seed), []
        for _ in range(repetitions):
            ix = rng.integers(0, len(ns), len(ns))
            boot.append(totals[ix].sum() / ns[ix].sum())
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {'pair_count': len(vals), 'video_clusters': len(by_video),
            'pair_acc': float(np.mean(vals)) if vals else None, 'ci95': ci}


def metrics(rows, scores, freeze):
    st = freeze['statistics']
    scores = np.asarray(scores)
    return {'pooled_auc': auc(labels(rows), scores), 'rows': len(rows),
            'within_query': within_query(rows, scores, st['bootstrap_repetitions'], freeze['seed'], st['component_policy']),
            'same_video_pairs': pair_metrics(rows, scores, st['bootstrap_repetitions'], freeze['seed'])}


def power_simulation(datasets, freeze):
    """Conditional on observed b/features; common video latent and row uniforms.

    Refits M2/M3 for each synthetic train label set. Does not retrain Cq.
    Sensitivities are fixed before observing actual Novel increments.
    """
    policy = freeze['statistics']['power_simulation']
    videos = sorted({str(r['vid']) for ds in datasets.values() for part in ds['rows'].values() for r in part})
    keys = sorted({(str(r['vid']), r['query']) for ds in datasets.values() for part in ds['rows'].values() for r in part})
    vi, qi = {v: i for i, v in enumerate(videos)}, {q: i for i, q in enumerate(keys)}
    templates = {}
    for fold, ds in datasets.items():
        m2 = OffsetLogistic().fit(ds['x']['inner_train'][:, :2], labels(ds['rows']['inner_train']), ds['b']['inner_train'])
        templates[fold] = {role: {'base': m2.predict(ds['x'][role][:, :2], ds['b'][role]),
                                  'vi': np.asarray([vi[str(r['vid'])] for r in rr]),
                                  'qi': np.asarray([qi[(str(r['vid']), r['query'])] for r in rr])}
                           for role, rr in ds['rows'].items() if role in ('inner_train', 'inner_novel_dev')}
    results, mdes = [], []
    for sigma in policy['video_random_effect_sd']:
        strength_records, null_critical = [], None
        for strength in policy['coefficient_grid']:
            estimates, oracle, valid = [], [], 0
            for repeat in range(policy['repetitions']):
                rng = np.random.default_rng(freeze['seed'] + 500000 + repeat)
                latent = rng.normal(0, sigma, len(videos))
                uniforms = rng.random(len(keys))
                deltas, truths = [], []
                for fold, ds in datasets.items():
                    synthetic = {}
                    for role, template in templates[fold].items():
                        logits = template['base'] + strength * ds['x'][role][:, 2] + latent[template['vi']]
                        target_rate = float(labels(ds['rows'][role]).mean())
                        shift = brentq(lambda c: float(expit(logits + c).mean()) - target_rate, -1000, 1000)
                        synthetic[role] = (uniforms[template['qi']] < expit(logits + shift)).astype(int)
                    if any(set(y) != {0, 1} for y in synthetic.values()):
                        break
                    xtr, xtest = ds['x']['inner_train'], ds['x']['inner_novel_dev']
                    btr, btest = ds['b']['inner_train'], ds['b']['inner_novel_dev']
                    m2 = OffsetLogistic().fit(xtr[:, :2], synthetic['inner_train'], btr)
                    m3 = OffsetLogistic().fit(xtr, synthetic['inner_train'], btr)
                    ytest = synthetic['inner_novel_dev']
                    deltas.append(auc(ytest, m3.predict(xtest, btest)) - auc(ytest, m2.predict(xtest[:, :2], btest)))
                    base = templates[fold]['inner_novel_dev']['base']
                    truths.append(auc(ytest, base + strength * xtest[:, 2]) - auc(ytest, base))
                if len(deltas) == len(datasets):
                    estimates.append(float(np.mean(deltas)))
                    oracle.append(float(np.mean(truths)))
                    valid += 1
            if valid != policy['repetitions']:
                return {'status': 'inconclusive_invalid_simulations', 'valid': valid, 'mde_auc': None}
            if strength == 0:
                null_critical = float(np.quantile(estimates, 1 - freeze['statistics']['mde_one_sided_alpha']))
            power = float(np.mean(np.asarray(estimates) > null_critical))
            strength_records.append({'coefficient': strength, 'power': power,
                                     'oracle_mean_macro_auc_increment': float(np.mean(oracle)),
                                     'valid_repetitions': valid})
        candidates = [v['oracle_mean_macro_auc_increment'] for i, v in enumerate(strength_records)
                      if v['coefficient'] > 0 and v['oracle_mean_macro_auc_increment'] > 0
                      and all(w['power'] >= freeze['statistics']['mde_target_power'] for w in strength_records[i:])]
        mde = min(candidates) if candidates else None
        mdes.append(mde)
        results.append({'video_random_effect_sd': sigma, 'null_critical_macro_delta': null_critical,
                        'mde_auc': mde, 'grid': strength_records})
        print(f'Conditional power sigma={sigma}: MDE={mde}', flush=True)
    return {'status': 'conditional_model_dependent' if all(v is not None for v in mdes) else 'inconclusive_grid',
            'mde_auc': max(mdes) if all(v is not None for v in mdes) else None,
            'sensitivity_results': results,
            'limitation': 'Conditional on fitted Cq/features; finite coefficient grid; assumed video latent effects; no Cq retraining uncertainty'}


def decide(delta, positive, ci, null_upper, power, within_intervals, freeze):
    mde = power.get('mde_auc')
    if mde is None or mde > freeze['statistics']['mde_limit_auc']:
        return 'inconclusive_power'
    e, n = freeze['decisions']['E'], freeze['decisions']['N']
    if (delta >= e['macro_delta_min'] and positive >= e['positive_folds_min'] and ci is not None and
        ci[0] > e['paired_macro_ci95_lower_gt'] and null_upper <= e['shuffle_null_same_statistic_upper_95_quantile_max']):
        return 'E_evidence_increment'
    if (delta < n['macro_delta_lt'] and positive <= n['positive_folds_max'] and
        all(v is not None and v[0] <= .5 <= v[1] for v in within_intervals)):
        return 'N_no_increment_detected_in_this_channel'
    return 'inconclusive'
