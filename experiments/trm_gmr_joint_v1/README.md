# Experiment: Moment-DETR-TRM-GMR-Joint-v1 (Split A1)

## Experiment Goal
Evaluate whether coupling candidate-conditioned phrase attention, candidate-wise gated phrase refinement, and an evidence-aware existence head under end-to-end joint gradients can mitigate Seen -> Unseen degradation and reduce false refusal of unseen queries on Charades-STA Semantic Novelty benchmark (`semantic_existence_v2`), focusing initially on **Split A1** (`put` / `take` holdout).

## Baseline Reference (Moment-DETR-GMR vs Current TRM on Split A1)
- **Moment-DETR-GMR Baseline**:
  - Seen AUROC: `0.8044`
  - Unseen AUROC: `0.4973`
  - Seen-Unseen AUROC Gap: `0.3071`
- **Moment-DETR-TRM Phase 1 Localization Baseline**:
  - Seen S+ R1@0.5: `43.87%`
  - Unseen U+ R1@0.5: `23.01%`
  - Localization Gap: `20.86%`

## Frozen Protocol
1. **Model**: `MomentDETR_TRM_GMR_Joint` (`models/moment_detr_trm_gmr_joint/`)
2. **Data**: `data/release/semantic_existence_v2/A1/`
   - Train on $S^+$ (7,108) and $S^-$ (1,500).
   - Validate on Seen $S^+ + S^-$ (1,759).
   - Test on all four quadrants $S^+$ (2,218), $S^-$ (1,244), $U^+$ (465), $U^-$ (1,243).
3. **Hyperparameters**:
   - `seed: 3407`, `lr: 0.0001`, `lr_drop: 400`, `bsz: 16`, `n_epoch: 100`, `max_es_cnt: -1`
   - `lambda_refine: 1.0`, `phrase_scale: 10.0`, `lambda_con: 1.0`, `lambda_neg: 0.5`, `lambda_exc: 1.0`, `exist_loss_coef: 1.0`
   - `max_v_l: 200`, `max_ts_val: 200.0`
4. **Execution Command**:
   ```bash
   bash scripts/train_trm_gmr_joint.sh A1
   bash scripts/infer_trm_gmr_joint.sh A1
   ```
