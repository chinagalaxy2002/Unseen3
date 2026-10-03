# Joint-v2 AUROC Results (Interim)

> Interim snapshot: only A1 and A2_alt have completed full-test inference. A3 and C1 are fitting their split-specific PCA; C2_alt is queued after A3. The macro row below averages the two completed splits only and is not the final five-split result.

Joint-v2 is an exploratory follow-up motivated by Joint-v1 failure modes. The Joint-v1 U results were viewed before this experiment was designed, so these results are not an untouched unseen evaluation. U was excluded from training, PCA fitting, checkpoint selection, threshold calibration, and hyperparameter selection. Canonical AUROC uses full-precision `pred_exist_logit`.

## Unseen AUROC

| Split | Joint-v1 U | V2 semantic-only U | V2 fused U | Fusion − semantic | V2 − V1 | V2 seen fused | Seen − unseen gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.462494 | 0.486633 | 0.480969 | -0.005664 | 0.018475 | 0.827954 | 0.346985 |
| A2_alt | 0.506067 | 0.522341 | 0.530697 | 0.008356 | 0.024630 | 0.774744 | 0.244047 |
| **Macro (completed 2/5 only)** | 0.484281 | 0.504487 | 0.505833 | 0.001346 | 0.021552 | 0.801349 | 0.295516 |

Among the completed splits, unseen AUROC improved over Joint-v1 on **2/2** splits and degraded on **0/2**. Visual fusion changed A1 unseen AUROC by -0.005664 and A2_alt by +0.008356; its contribution is not yet consistent.

## Run Snapshot

| Split | Best epoch | Epochs run | Seen-val MR-full-mAP | Frozen threshold | alpha_visual | AUC-valid batch fraction | PCA background samples | NaN/Inf | Feature fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A1 | 28 | 75 | 24.28 | 0.988989 | 0.957952 | 0.952959 | 100,000 | False | False |
| A2_alt | 20 | 64 | 24.39 | 0.994621 | 0.947816 | 0.916869 | 100,000 | False | False |

A1 was manually stopped at epoch 75. A2_alt was stopped at epoch 64 after the already-running process had accumulated 43 validation epochs without a new Seen-val MR-full-mAP best; its selected checkpoint is from epoch 20. No NaN/Inf or missing-feature fallback was reported for either split.

The full-precision per-split predictions and detailed summaries are retained in the local `results/moment_detr_trm_gmr_joint_v2/{A1,A2_alt}/` directories. The versioned, compact machine-readable AUROC snapshot is `CURRENT_AUROC_RESULTS.json`.

This file is an interim report. The five-split report should replace or supersede it after A3, C1, and C2_alt complete.
