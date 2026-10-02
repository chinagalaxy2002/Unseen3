# Task Status: Moment-DETR-TRM Generalization

## Current Phase: Stage 7 - Frozen Formal Training & Multi-Split Execution

### Task Summary
- **Goal**: Full code migration, architecture implementation, and unit verification of Moment-DETR-TRM and Moment-DETR-TRM-PT phrase-level temporal relationship mining on the Charades-STA Semantic Novelty benchmark.
- **Constraints Fully Enforced**:
  - Modifications strictly confined to `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`.
  - Data and features use symlinks (`ln -s`); strictly zero modifications outside `Unseen3`.
  - Phase 1 focuses exclusively on $S^+$ localization generalization to $U^+$: no $S^-/U^-$ training, no existence heads (`use_exist_head=False`), no tuning or model selection on $U^+$.

### Stage Status
- [x] **Stage 1**: Source code audit of baseline, TRM repo, and papers (`BASELINE_AUDIT.md`, `TRM_SOURCE_AUDIT.md`, `METHOD_SPEC.md`, `EXPERIMENT_PLAN.md`, `SOURCE_MANIFEST.json`).
- [x] **Stage 2**: Baseline copy into Unseen3, git initialization, symlink checks (`git commit: Baseline copy before Moment-DETR TRM migration`).
- [x] **Stage 3**: Offline phrase extraction (20,400 queries parsed, 0 fallbacks) and CLIP phrase feature caching (20,400 `.npz` files in `features/phrase_data/clip_phrase/`).
- [x] **Stage 4**: Model architecture implementation (`models/moment_detr_trm/`, `training/moment_detr_trm/`, `configs/moment_detr_trm/`).
- [x] **Stage 5**: 7-step unit verification suite (`scripts/smoke_test_trm.py` PASSED with 100% success; gradient flow verified; overfit test loss decreased by 19.56%).
- [x] **Stage 6**: TRM-PT branch implementation (`models/moment_detr_trm_pt/`, `training/moment_detr_trm_pt/`, `scripts/smoke_test_trm_pt.py` PASSED; 9,428 $S^+$ pseudo labels generated in `features/phrase_data/pt_pseudo_labels/`).
- [x] **Stage 7**: Formal configuration freeze (`EXPERIMENT_FREEZE.json`) and verified training/evaluation runners (`scripts/train_trm.sh`, `scripts/infer_trm.sh`, `scripts/run_all_splits_trm.sh`). All 6 post-audit protocol alignments completed and verified:
  1. `max_v_l: 200` & `max_ts_val: 200` restored to match Moment-DETR Charades baseline protocol.
  2. Training hyperparameters aligned with baseline protocol: `seed=3407`, `bsz=16`, `eval_bsz=16`, `lr_drop=400`, `max_es_cnt=-1`.
  3. Strict phrase extraction: `charades_test.json` removed completely; all 8,663 evaluation queries parsed strictly by frozen spaCy parser with 0 test-side annotation lookup.
  4. Raw-cosine weighted refinement implemented: `fg_logit += lambda_refine * scale * (phrase_weights @ raw_cosine)` to enable both positive reinforcement and negative suppression.
  5. Projected CLIP phrase embeddings cached and 9,428 TRM-PT pseudo labels regenerated in shared visual-textual CLIP space.
  6. `EXPERIMENT_FREEZE.json` completely rebuilt with corrected commit (`ddc78aa9...`), AAAI URL (`25478`), and verified SHA256 hashes.
  *(Note: The previous preliminary 1-epoch test on split A1 is marked strictly as an engineering smoke/development check under legacy 75-frame setting, not a formal benchmark result).*

### Evidence Directory
All documentation, logs, metrics, manifests, and verification scripts are recorded in:
`/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/experiments/trm_momentdetr_generalization/`
- Configuration Freeze: `experiments/trm_momentdetr_generalization/EXPERIMENT_FREEZE.json`
- Method Spec: `experiments/trm_momentdetr_generalization/METHOD_SPEC.md`
- Implementation Log: `experiments/trm_momentdetr_generalization/IMPLEMENTATION_LOG.md`
- Source Manifest: `experiments/trm_momentdetr_generalization/SOURCE_MANIFEST.json`
