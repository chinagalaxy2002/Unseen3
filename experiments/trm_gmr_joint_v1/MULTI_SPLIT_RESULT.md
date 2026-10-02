# Moment-DETR-TRM-GMR-Joint vs Baseline Comparison Across Splits

**Completed splits**: 1/5 (A1)

## 1. Existence AUROC Comparison vs Moment-DETR-GMR

| Split | Baseline Seen AUROC | Baseline Unseen AUROC | Baseline Gap | Joint Seen AUROC | Joint Unseen AUROC | Joint Gap | Gap Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 0.8044 | 0.4973 | 0.3071 | 0.8131 | 0.4625 | **0.3506** | **+0.0435** |

## 2. Localization Generalization & Existence Robustness

| Split | S+ Raw R1@0.5 | U+ Raw R1@0.5 | Raw Loc Gap | U+ Soft-Gated R1@0.5 | U+ Hard-Gated R1@0.5 | U+ FRR (误拒率) | U- RR (拒识率) | Matched Pair Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 40.22% | 25.59% | 14.63% | 26.02% | 21.72% | 11.83% | 10.90% | 59.78% |
