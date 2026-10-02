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
  3. Unified frozen parser protocol: 100% of all queries across train, val, and test are parsed by the exact same frozen spaCy constituent parser (20,400 queries 100% parsed), completely eliminating train/test distribution shift. Official TRM annotations kept purely for audit.
  4. Raw-cosine weighted refinement implemented: `fg_logit += lambda_refine * scale * (phrase_weights @ raw_cosine)` to enable both positive reinforcement and negative suppression.
  5. Projected CLIP phrase embeddings cached and 9,428 TRM-PT pseudo labels regenerated using single contiguous connected components strictly constrained within sentence GT intervals (0 violations across all 9,428 instances).
  6. `EXPERIMENT_FREEZE.json` completely rebuilt with corrected commit (`ddc78aa9...`), AAAI URL (`25478`), and verified fresh SHA256 hashes.
  *(Note: The previous preliminary 1-epoch test on split A1 is marked strictly as an engineering smoke/development check under legacy 75-frame setting, not a formal benchmark result).*

### Evidence Directory
All documentation, logs, metrics, manifests, and verification scripts are recorded in:
`/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/experiments/trm_momentdetr_generalization/`
- Configuration Freeze: `experiments/trm_momentdetr_generalization/EXPERIMENT_FREEZE.json`
- Method Spec: `experiments/trm_momentdetr_generalization/METHOD_SPEC.md`
- Implementation Log: `experiments/trm_momentdetr_generalization/IMPLEMENTATION_LOG.md`
- Source Manifest: `experiments/trm_momentdetr_generalization/SOURCE_MANIFEST.json`

### Multi-Split Benchmark Execution Progress (2-GPU Parallel Queue)
- **Split A1**: **COMPLETED (100 Epochs, max_v_l=200, seed=3407)**
  - Seen Positive ($S^+$, N=2218): R@1@0.3 = **57.03%**, R@1@0.5 = **43.87%** (+6.90% vs baseline 36.97%), R@1@0.7 = **24.62%**, mIoU = **39.24%**
  - Unseen Positive ($U^+$, N=465): R@1@0.3 = **36.34%**, R@1@0.5 = **23.01%**, R@1@0.7 = **11.83%**, mIoU = **24.29%**
  - Generalization Gap: Gap R1@0.5 = **20.86%**, Gap mIoU = **14.95%**
  - Checkpoint: `results/moment_detr_trm/A1/best.ckpt`
  - Summary: `results/moment_detr_trm/A1/eval_output/generalization_summary.json`
- **Split A2_alt**: **COMPLETED (100 Epochs, max_v_l=200, seed=3407)**
  - Seen Positive ($S^+$, N=2798): R@1@0.3 = **48.25%**, R@1@0.5 = **36.38%**, R@1@0.7 = **19.98%**, mIoU = **33.88%**
  - Unseen Positive ($U^+$, N=168): R@1@0.3 = **37.50%**, R@1@0.5 = **25.60%**, R@1@0.7 = **11.90%**, mIoU = **24.20%**
  - Generalization Gap: Gap R1@0.5 = **10.79%**, Gap mIoU = **9.68%**
  - Checkpoint: `results/moment_detr_trm/A2_alt/best.ckpt`
  - Summary: `results/moment_detr_trm/A2_alt/eval_output/generalization_summary.json`
- **Split A3**: **COMPLETED (100 Epochs, max_v_l=200, seed=3407)**
  - Seen Positive ($S^+$, N=2784): R@1@0.3 = **52.01%**, R@1@0.5 = **36.85%**, R@1@0.7 = **20.01%**, mIoU = **35.08%**
  - Unseen Positive ($U^+$, N=192): R@1@0.3 = **68.75%**, R@1@0.5 = **51.56%**, R@1@0.7 = **28.12%**, mIoU = **46.13%**
  - Generalization Gap: Gap R1@0.5 = **-14.71%**, Gap mIoU = **-11.05%** (Super-generalization on A3)
  - Checkpoint: `results/moment_detr_trm/A3/best.ckpt`
  - Summary: `results/moment_detr_trm/A3/eval_output/generalization_summary.json`
- **Split C1**: **IN PROGRESS** on GPU 1 (Epoch ~80/100, finishing in ~15 mins)
- **Split C2_alt**: **IN PROGRESS** on GPU 0 (Epoch ~2/100, automatically claimed upon A3 completion)



