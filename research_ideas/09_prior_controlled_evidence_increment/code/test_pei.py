"""Focused protocol checks with synthetic data; no benchmark inputs or training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np

from core import OffsetLogistic, Standardizer, auc
from crossfit import Cq, predict, video_folds, heldin_split
from evidence import Evidence
from analysis_stats import decide, shared_macro_interval, within_query


def row(vid, query, label, qid):
    return {'vid': vid, 'query': query, 'exist_label': label, 'qid': qid}


def toy_evidence(matrix):
    obj = object.__new__(Evidence)
    obj.rows = {
        'inner_train': [row('v0', 'q0', 1, 'a'), row('v1', 'q1', 0, 'b')],
        'inner_seen_val': [row('v2', 'q0', 1, 'c'), row('v3', 'q2', 0, 'd')],
        'inner_novel_dev': [row('v2', 'q1', 1, 'e'), row('v3', 'q2', 0, 'f')]}
    obj.videos, obj.queries = ['v0', 'v1', 'v2', 'v3'], ['q0', 'q1', 'q2']
    obj.vi = {v: i for i, v in enumerate(obj.videos)}
    obj.qi = {q: i for i, q in enumerate(obj.queries)}
    obj.train_videos, obj.train_queries = obj.videos[:2], obj.queries[:2]
    obj.nv, obj.nq = 2, 2
    obj.methods, obj.matrices = ['max'], {'max': matrix}
    obj.sums = {'max': (matrix[:2].sum(0), matrix[:, :2].sum(1), matrix[:2, :2].sum())}
    return obj


class ProtocolChecks(unittest.TestCase):
    def test_exact_leave_self_out_and_shuffle_references(self):
        matrix = np.array([[1., 2, 4], [5, 8, 9], [10, 20, 25], [30, 40, 50]])
        obj = toy_evidence(matrix)
        for mapping in (None, obj.mapping(42)):
            cols, raw = obj.columns('max', mapping)
            for role, rr in obj.rows.items():
                for k, r in enumerate(rr):
                    donor = r['vid'] if mapping is None else mapping[role][r['vid']]
                    allowed_v = [v for v in obj.train_videos if v != r['vid']]
                    if mapping is not None:
                        allowed_v = [mapping['inner_train'][v] for v in allowed_v]
                    allowed_q = [q for q in obj.train_queries if q != r['query']]
                    iq = obj.qi[r['query']]
                    iv = obj.vi[donor]
                    muq = np.mean([matrix[obj.vi[v], iq] for v in allowed_v])
                    muv = np.mean([matrix[iv, obj.qi[q]] for q in allowed_q])
                    mu = np.mean([matrix[obj.vi[v], obj.qi[q]] for v in allowed_v for q in allowed_q])
                    np.testing.assert_allclose(cols[role][k], [muq, muv, raw[role][k] - muq - muv + mu])

    def test_additive_priors_cancel(self):
        matrix = np.arange(4)[:, None] + np.array([2., 8, 16])[None, :]
        obj = toy_evidence(matrix)
        cols, _ = obj.columns('max')
        for v in cols.values():
            np.testing.assert_allclose(v[:, 2], 0, atol=1e-12)

    def test_video_shuffle_bijective_broadcast(self):
        obj = toy_evidence(np.zeros((4, 3)))
        obj.rows['inner_train'].append(row('v0', 'q1', 0, 'other_query'))
        mapping = obj.mapping(5)
        self.assertEqual(mapping, obj.mapping(5))
        for mm in mapping.values():
            self.assertEqual(set(mm), set(mm.values()))
            self.assertTrue(all(v != d for v, d in mm.items()))

    def test_video_crossfit_and_internal_validation_no_overlap(self):
        rows = [row(str(v), str(q), (v + q) % 2, f'{v}_{q}') for v in range(30) for q in range(2)]
        outer, assignment = video_folds(rows, 3407, 5)
        for k in range(5):
            fit, val = heldin_split(rows, outer, k, 50 + k, .2)
            fit_v = {rows[i]['vid'] for i in fit}
            val_v = {rows[i]['vid'] for i in val}
            oof_v = {r['vid'] for i, r in enumerate(rows) if outer[i] == k}
            self.assertFalse(fit_v & val_v or fit_v & oof_v or val_v & oof_v)
            self.assertEqual(fit_v | val_v | oof_v, set(assignment))

    def test_fixed_offset_preserved_with_no_predictor_signal(self):
        offset = np.array([-3., -1, 1, 3])
        model = OffsetLogistic().fit(np.zeros((4, 2)), [0, 1, 0, 1], offset)
        np.testing.assert_allclose(np.diff(model.predict(np.zeros((4, 2)), offset)), np.diff(offset), atol=1e-10)
        self.assertEqual(auc([0, 1, 0, 1], model.predict(np.zeros((4, 2)), offset)), auc([0, 1, 0, 1], offset))

    def test_zero_variance_and_train_only_standardization(self):
        scale = Standardizer().fit([[1, 2], [1, 4]])
        np.testing.assert_allclose(scale.transform([[100, 6]]), [[0, 3]])

    def test_shared_fold_bootstrap_does_not_create_independent_units(self):
        rr = [row('v0', 'q0', 0, '0'), row('v1', 'q1', 1, '1'), row('v2', 'q2', 0, '2'), row('v3', 'q3', 1, '3')]
        scores = {'M2': np.array([0., 1, 1, 0]), 'M3': np.array([0., 1, 0, 1])}
        one = shared_macro_interval({'a': rr}, {'a': scores}, 100, 42)
        two = shared_macro_interval({'a': rr, 'b': rr}, {'a': scores, 'b': scores}, 100, 42)
        self.assertEqual(one, two)

    def test_dominant_incidence_component_cannot_supply_ci(self):
        rr = [row('v0', 'same', 0, 'a'), row('v1', 'same', 1, 'b')]
        result = within_query(rr, np.array([0., 1]), 100, 42,
                              {'minimum_components': 5, 'largest_component_fraction_max': .5})
        self.assertEqual(result['query_macro_auc'], 1)
        self.assertIsNone(result['ci95'])

    def test_mde_gate_prevents_positive_and_negative_overclaims(self):
        freeze = {'statistics': {'mde_limit_auc': .02},
                  'decisions': {'E': {'macro_delta_min': .02, 'positive_folds_min': 3,
                      'paired_macro_ci95_lower_gt': 0., 'shuffle_null_same_statistic_upper_95_quantile_max': .005},
                      'N': {'macro_delta_lt': .01, 'positive_folds_max': 1}}}
        self.assertEqual(decide(.03, 4, [.01, .05], 0., {'mde_auc': None}, [], freeze), 'inconclusive_power')
        self.assertEqual(decide(.03, 4, [.01, .05], 0., {'mde_auc': .01}, [], freeze), 'E_evidence_increment')
        self.assertEqual(decide(0., 0, [-.01, .01], 0., {'mde_auc': .01}, [None] * 4, freeze), 'inconclusive')


    def test_query_only_exact_input_scores_are_identical(self):
        x = np.random.default_rng(0).normal(size=(300, 512)).astype(np.float32)
        x[299] = x[0]
        model = Cq()
        torch.set_num_threads(2)
        result = predict(model, x, 'cpu')
        self.assertEqual(result[0], result[299])

    def test_complete_pipeline_with_synthetic_features_only(self):
        import run_pipeline as pipeline
        config = json.loads((Path(__file__).resolve().parents[1] / 'records/FREEZE.json').read_text())
        config['crossfit_seeds'] = [3407]
        config['stage2']['cq_epochs'] = 2
        config['statistics']['shuffle_repetitions'] = 3
        config['statistics']['bootstrap_repetitions'] = 20
        config['statistics']['power_simulation'].update(repetitions=3, coefficient_grid=[0., .25], video_random_effect_sd=[0.])
        parts = {}
        for role, count in [('inner_train', 30), ('inner_seen_val', 10), ('inner_novel_dev', 10)]:
            parts[role] = [row(f'{role}_v{v}', f'q{q}', q % 2, f'{role}_{v}_{q}')
                           for v in range(count) for q in range(2)]
        class SyntheticFeatures:
            def __init__(self, *args):
                self.hashes = {}
            def query(self, r):
                x = np.zeros((2, 512), dtype=np.float32)
                x[:, int(r['query'][1:])] = 1
                return x
            def means(self, rows):
                return np.stack([self.query(r).mean(0) for r in rows])
            def unique_queries(self, rows):
                return {r['query']: self.query(r).mean(0) for r in rows}
            def video(self, vid):
                rng = np.random.default_rng(sum(ord(c) for c in vid))
                x = rng.normal(size=(4, 512)).astype(np.float32)
                return x / np.linalg.norm(x, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            freeze_path = directory / 'synthetic_freeze.json'
            freeze_path.write_text(json.dumps(config))
            audit_path = directory / 'synthetic_audit.json'
            audit_path.write_text('{}')
            audit = {'folds': {f: {'train_features': {'feature_sha256': {}}} for f in config['folds']}}
            args = SimpleNamespace(freeze=freeze_path, audit=audit_path, run_id='synthetic', stage='full',
                                   clip_root=directory, text_base=directory, device='cpu')
            cv = {r: np.zeros(len(parts[r])) for r in ('inner_seen_val', 'inner_novel_dev')}
            torch.set_num_threads(2)
            with patch.object(pipeline, 'IDEA', directory), patch.object(pipeline, 'load_freeze', return_value=config), \
                 patch.object(pipeline, 'validate_gate', return_value=audit), \
                 patch.object(pipeline, 'load_rows', side_effect=lambda fold, role, freeze: parts[role]), \
                 patch.object(pipeline, 'Features', SyntheticFeatures), \
                 patch.object(pipeline, 'replay_cv', return_value=(cv, {'synthetic': True})):
                pipeline.run(args)
                results = json.loads((directory / 'runs/synthetic/RESULTS.json').read_text())
                self.assertEqual(len(results['folds']), 4)
                self.assertTrue(np.isfinite(results['primary']['macro_novel_delta']))
                self.assertTrue((directory / 'runs/synthetic/OUTPUT_SHA256.json').exists())
                with self.assertRaisesRegex(ValueError, 'Completed run'):
                    pipeline.run(args)


    def test_formal_u_annotations_are_rejected_before_feature_loading(self):
        import core
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'experiments/moment_detr_gmr_evidence_v5/inner_folds/A1_action_01'
            directory.mkdir(parents=True)
            path = directory / 'inner_train.jsonl'
            path.write_text(json.dumps(dict(row('v', 'q', 1, 'qid'), partition='U+')) + '\n')
            manifest_path = directory / 'manifest.json'
            manifest_path.write_text(json.dumps({'files': {'inner_train': {'sha256': core.digest(path)}}}))
            freeze = {'folds': {'A1_action_01': {'manifest_sha256': core.digest(manifest_path)}}}
            with patch.object(core, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Forbidden partition'):
                    core.load_rows('A1_action_01', 'inner_train', freeze)

    def test_pending_alignment_gate_cannot_be_bypassed(self):
        import run_pipeline as pipeline
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fp, ap = directory / 'freeze.json', directory / 'audit.json'
            fp.write_text('{}')
            freeze = {'code_sha256': {'code/audit_stage0.py': 'script_hash'}}
            audit = {'formal_u_read': False, 'freeze_sha256': pipeline.digest(fp),
                     'script_sha256': 'script_hash', 'folds': {'A1_action_01': {'train_features': {'alignment_gate': 'pending'}}}}
            ap.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'alignment/encoding gate'):
                pipeline.validate_gate(ap, fp, freeze)


if __name__ == '__main__':
    unittest.main()
