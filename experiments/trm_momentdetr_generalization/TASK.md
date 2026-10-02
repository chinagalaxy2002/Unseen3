# Task Status: Moment-DETR-TRM Generalization

## Current Phase: Stage 2 - Baseline Copy & Setup

### Task Summary
- **Goal**: Full code migration and training verification of Moment-DETR with TRM phrase-level temporal relationship mining on the Charades semantic novelty benchmark.
- **Constraints**:
  - Modifications strictly confined to `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`.
  - Data and features must use symlinks (`ln -s`).
  - No external files may be modified.
  - Phase 1 focuses exclusively on $S^+$ localization generalization to $U^+$.

### Stage Status
- [x] **Stage 1**: Source code audit of baseline, TRM repo, and papers.
  - Deliverables: `SOURCE_MANIFEST.json`, `BASELINE_AUDIT.md`, `TRM_SOURCE_AUDIT.md`, `METHOD_SPEC.md`, `EXPERIMENT_PLAN.md`.
- [ ] **Stage 2**: Baseline copy into Unseen3, git initialization, symlink checks.
- [ ] **Stage 3**: Offline phrase extraction and CLIP phrase feature caching.
- [ ] **Stage 4**: Model architecture implementation (`models/moment_detr_trm/`, `training/moment_detr_trm/`).
- [ ] **Stage 5**: 7-step smoke testing and verification suite.
- [ ] **Stage 6**: TRM-PT branch implementation (isolated).
- [ ] **Stage 7**: Formal training and multi-split evaluation.

### Evidence Directory
All documentation, logs, metrics, and evidence are recorded in:
`/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/experiments/trm_momentdetr_generalization/`
