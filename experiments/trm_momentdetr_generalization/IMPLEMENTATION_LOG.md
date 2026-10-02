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
- [x] Rsync baseline into `Unseen3` (preserving `plan.md`, `_external/`, and `experiments/`).
- [x] Initialize git repo in `Unseen3` and make baseline commit (`a50c822 Baseline copy before Moment-DETR TRM migration`).
- [x] Verify symlinks to features (`features/charades_video/vid_slowfast`, `features/charades_video/vid_clip`, `features/semantic_existence_v2/shared_clip_text`) and datasets (`data/release/semantic_existence_v2`).
- [x] Verify baseline Moment-DETR modules are present and isolated.

## Stage 3: Offline Phrase Extraction & CLIP Caching
- [x] Implement `scripts/extract_phrases.py` using TRM dictionary lookup + fallback constituent parser. Parsed all 20,400 queries across 5 splits with 0 fallbacks needed.
- [x] Generate phrase metadata `features/phrase_data/trm_phrase_metadata.jsonl`.
- [x] Implement `scripts/extract_phrase_clip_features.py` using shared CLIP ViT-B/32 `last_hidden_state`. Fixed per-query accumulation bug and verified caching.
- [x] Cache phrase token features to `features/phrase_data/clip_phrase/qid*.npz` (20,400 files verified, shape `[10, 16, 512]`).

## Stage 4: Moment-DETR-TRM Model Implementation
- [x] Implement `MomentDETR_TRM` in `models/moment_detr_trm/moment_detr_trm.py` (proposals = DETR decoder queries, slot projection, phrase-slot cosine matching, attentive pooling, score refinement).
- [x] Implement `AttentivePooling` and similarity projections in `models/moment_detr_trm/trm_modules.py`.
- [x] Implement `SetCriterionTRM` in `models/moment_detr_trm/trm_loss.py` with consistency loss, negative phrase contrastive loss, and exclusiveness loss.
- [x] Implement `StartEndDatasetTRM` in `training/moment_detr_trm/dataset_trm.py` with `partition_filter` to strictly isolate $S^+$ positive instances.
- [x] Implement training and evaluation scripts in `training/moment_detr_trm/` (`train_trm.py`, `evaluate_trm.py`, `config.py`, `eval_cli_trm.py`).
- [x] Implement configs in `configs/moment_detr_trm/` (`base.yml`, `feature/clip_slowfast.yml`, `model/moment_detr_trm.yml`, `dataset/charades_sta_semantic_novelty.yml`).

## Stage 5: Verification & Smoke Testing
- [x] Step A: Import test (`scripts/smoke_test_trm.py` Step A passed on CPU/CUDA).
- [x] Step B: Dataset smoke test (4 samples inspected, $B=1$ edge case verified, 7,108 $S^+$ rows loaded).
- [x] Step C: Model forward test (check shapes: pred_spans $[B, 10, 2]$, phrase_scores $(0, 1)$, phrase_weights sum to 1.0).
- [x] Step D: Loss test (finite values: span, giou, label, con, neg, exc; empty GT check passed).
- [x] Step E: Backward test (non-zero gradient check verified on phrase_proj, slot_proj, attentive_pooling, transformer enc/dec).
- [x] Step F: 1-step optimizer update test passed.
- [x] Step G: 15-step tiny overfit test (loss decreased monotonically from 6.6856 to 5.3777, -19.56%).

## Stage 6: TRM-PT Branch Implementation
- [x] Implement paper-derived pseudo temporal label generator `scripts/generate_trm_pt_pseudo_labels.py` (9,428 $S^+$ query pseudo labels generated into `features/phrase_data/pt_pseudo_labels/qid*.npz`).
- [x] Implement `models/moment_detr_trm_pt/moment_detr_trm_pt.py` with `phrase_span_head`.
- [x] Implement `models/moment_detr_trm_pt/trm_pt_loss.py` with `SetCriterionTRM_PT` incorporating phrase-level pseudo span L1 and GIoU losses.
- [x] Implement `training/moment_detr_trm_pt/dataset_trm_pt.py`.
- [x] Implement and execute `scripts/smoke_test_trm_pt.py` (all checks passed; overfit loss decreased from 6.5748 to 5.2952; phrase_span_head gradient 0.0130).
- [x] Marked undocumented hyperparameter choices explicitly as `UNSPECIFIED_BY_PAPER` in code and logs.

## Stage 7: Formal Training & Benchmark Verification
- [x] Freeze experiment configurations: `EXPERIMENT_FREEZE.json` (recorded commits, code SHA256, hyperparams, data splits, and protocol rules).
- [x] Implement production runners: `scripts/train_trm.sh`, `scripts/infer_trm.sh`, `scripts/run_all_splits_trm.sh`.
- [x] Run full end-to-end training and evaluation loop on split A1 (1 epoch verification): confirmed zero data leakage, correct checkpoint saving, evaluation metrics calculation, and generalization gap recording.
- [x] Project is fully verified, frozen, and ready for multi-split formal training.
