# Implementation Log: Moment-DETR-TRM Migration

## Stage 1: Source Audits & Planning
- [x] Read `Unseen3/plan.md` completely.
- [x] Audited baseline repository `Unseen` (commit `ddc78aa`).
- [x] Verified Moment-DETR architecture in `Unseen2` and `generalized-moment-retrieval`.
- [x] Cloned official TRM repo (`minghangz/TRM`, commit `a0fc1f97927bc1a5e53ae30d1032e343f862911f`).
- [x] Downloaded and extracted AAAI 2023 TRM paper text.
- [x] Audited TRM source code mechanisms A through K.
- [x] Confirmed disparity in `AttentivePooling` (global sentence feature is unused in code).
- [x] Confirmed disparity in `configs/charades.yaml` (`EXC_WEIGHT: 0.0` in repo vs active exclusiveness in paper).
- [x] Decided full mechanism policy: enable consistency, contrastive negative, and exclusiveness with frozen weights.
- [x] Generated `SOURCE_MANIFEST.json`, `BASELINE_AUDIT.md`, `TRM_SOURCE_AUDIT.md`, `METHOD_SPEC.md`, `EXPERIMENT_PLAN.md`.

## Stage 2: Codebase Copy & Directory Setup
- [ ] Rsync baseline into `Unseen3` (preserving `plan.md`, `_external/`, and `experiments/`).
- [ ] Initialize git repo in `Unseen3` and make baseline commit.
- [ ] Verify symlinks to features and datasets.
- [ ] Verify baseline Moment-DETR modules are present and isolated.

## Stage 3: Offline Phrase Extraction & CLIP Caching
- [ ] Implement `extract_phrases.py` using TRM dictionary lookup + fallback constituent parser.
- [ ] Generate phrase metadata `trm_phrase_metadata.jsonl`.
- [ ] Implement `extract_phrase_clip_features.py` using shared CLIP ViT-B/32.
- [ ] Cache phrase token features to `features/semantic_existence_v2/{split}/clip_phrase/`.

## Stage 4: Moment-DETR-TRM Model Implementation
- [ ] Implement `MomentDETR_TRM` in `models/moment_detr_trm/moment_detr_trm.py`.
- [ ] Implement `AttentivePooling` and similarity projections in `models/moment_detr_trm/trm_modules.py`.
- [ ] Implement `TRMCriterion` in `models/moment_detr_trm/trm_loss.py`.
- [ ] Implement `DatasetTRM` in `training/moment_detr_trm/dataset_trm.py`.
- [ ] Implement training and evaluation scripts in `training/moment_detr_trm/`.

## Stage 5: Verification & Smoke Testing
- [ ] Step A: Import test.
- [ ] Step B: Dataset smoke test (4 samples).
- [ ] Step C: Model forward test (check shapes).
- [ ] Step D: Loss test (finite values, empty GT check).
- [ ] Step E: Backward test (gradient check on all modules).
- [ ] Step F: 1-step optimizer update test.
- [ ] Step G: 15-step tiny overfit test (loss decreases).

## Stage 6: TRM-PT Branch Implementation
- [ ] Implement paper-derived pseudo temporal label module.
- [ ] Mark annotations as `UNSPECIFIED_BY_PAPER` where paper omitted exact hyperparameters.

## Stage 7: Formal Training & Benchmark Evaluation
- [ ] Freeze experiment configurations: `EXPERIMENT_FREEZE.json`.
- [ ] Run formal training for split A1.
- [ ] Evaluate split A1 and report metrics.
- [ ] Run full 5-split suite (`A1`, `A2_alt`, `A3`, `C1`, `C2_alt`).
- [ ] Compile final results and comparison tables.
