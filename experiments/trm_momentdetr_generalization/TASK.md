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
- [x] **Stage 7**: Formal configuration freeze (`EXPERIMENT_FREEZE.json`) and verified training/evaluation runners (`scripts/train_trm.sh`, `scripts/infer_trm.sh`, `scripts/run_all_splits_trm.sh`). Full training epoch and evaluation cycle tested and verified end-to-end on Split A1 (`results/moment_detr_trm/A1/best.ckpt`, `results/moment_detr_trm/A1/eval_output/generalization_summary.json`: S+ R@1@0.5=30.43%, U+ R@1@0.5=23.87%, Gap=6.56%).

### Evidence Directory
All documentation, logs, metrics, manifests, and verification scripts are recorded in:
`/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/experiments/trm_momentdetr_generalization/`
- Full 1-epoch test outputs: `results/moment_detr_trm/A1/`
- Checkpoint: `results/moment_detr_trm/A1/best.ckpt`
- Generalization Metrics: `results/moment_detr_trm/A1/eval_output/generalization_summary.json`
