# GMR-AUC-v4 five-split result

Exploratory follow-up; not an untouched unseen evaluation. U was excluded from training, pair construction, validation selection and threshold calibration.

| Split | Baseline Seen AUROC | V4 Seen AUROC | ΔSeen | Baseline Unseen AUROC | V4 Unseen AUROC | ΔUnseen (paired bootstrap 95% CI) | Baseline gap | V4 gap | Δgap (V4−base) | Baseline PairAcc | V4 PairAcc | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.80341 | 0.80491 | 0.00150 | 0.49559 | 0.49603 | 0.00044 [-0.00009, 0.00098] | 0.30782 | 0.30888 | 0.00106 | 0.59936 | 0.59936 | 9 |
| A2_alt | 0.76923 | 0.76842 | -0.00080 | 0.54747 | 0.54749 | 0.00002 [-0.00221, 0.00225] | 0.22176 | 0.22094 | -0.00082 | 0.67089 | 0.65823 | 24 |
| A3 | 0.74712 | 0.74694 | -0.00018 | 0.56502 | 0.56514 | 0.00012 [-0.00118, 0.00153] | 0.18210 | 0.18180 | -0.00030 | 0.48062 | 0.48062 | 46 |
| C1 | 0.76112 | 0.76112 | 0.00000 | 0.56834 | 0.56834 | 0.00000 [0.00000, 0.00000] | 0.19278 | 0.19278 | 0.00000 | 0.68056 | 0.68056 | 0 |
| C2_alt | 0.67593 | 0.69154 | 0.01561 | 0.46864 | 0.47001 | 0.00137 [-0.04139, 0.04691] | 0.20729 | 0.22153 | 0.01424 | 0.72727 | 0.63636 | 46 |

Macro baseline unseen AUROC: **0.52901**
Macro V4 unseen AUROC: **0.52940**
Macro Δ unseen AUROC: **0.00039**
Macro baseline seen AUROC: **0.75136**
Macro V4 seen AUROC: **0.75459**
Macro Δ seen AUROC: **0.00323**
Improved / degraded / unchanged unseen splits: **4 / 0 / 1**

A positive Δgap means the Seen−Unseen gap widened; negative means it narrowed. Every split's unseen paired-bootstrap CI includes zero, so the small positive macro delta is uncertain.

## Secondary diagnostics

| Split | Seen-val AUROC base → selected | U+ FRR | U− RR | S+ raw R1@0.5 | U+ raw R1@0.5 | S+ raw mIoU | U+ raw mIoU | Δ mean / std / max abs | Same pairs exact / composition / action | Train S+ / S− / total | Val rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.83665 → 0.83843 (epoch 9) | 0.22796 | 0.25201 | 0.36655 | 0.23011 | 0.35083 | 0.25079 | 0.10019 / 0.25976 / 1.38191 | 0.24196 / 0.36937 / 0.38867 | 7108 / 1500 / 8608 | 1257 |
| A2_alt | 0.82074 → 0.82253 (epoch 24) | 0.00000 | 0.00962 | 0.35633 | 0.26786 | 0.34277 | 0.25838 | 0.19062 / 0.61138 / 1.95810 | 0.24686 / 0.37157 / 0.38157 | 8823 / 1500 / 10323 | 1529 |
| A3 | 0.79248 → 0.79301 (epoch 46) | 0.29167 | 0.32492 | 0.33980 | 0.38021 | 0.32886 | 0.37440 | 0.13711 / 0.39814 / 1.87964 | 0.23942 / 0.37748 / 0.38310 | 8852 / 1500 / 10352 | 1548 |
| C1 | 0.78826 → 0.78826 (epoch 0) | 0.03704 | 0.07778 | 0.34691 | 0.48765 | 0.33267 | 0.44223 | 0.00000 / 0.00000 / 0.00000 | 0.13895 / 0.44596 / 0.41509 | 9016 / 1500 / 10516 | 1451 |
| C2_alt | 0.69500 → 0.71555 (epoch 46) | 0.85217 | 0.82283 | 0.32888 | 0.25217 | 0.30795 | 0.24085 | 0.24622 / 0.80771 / 1.96043 | 0.14555 / 0.46791 / 0.38653 | 9112 / 1500 / 10612 | 1476 |

All test qids and full-precision logits were checked. In every split baseline and V4 raw spans, localization class logits, S+ and U+ raw R1@0.5, and raw mIoU are identical. All train/val banks include every Seen sample, contain no U rows, and have zero missing features. NaN/Inf checks passed.

See per-split `v4_summary.json`, `test_predictions.jsonl`, `adapter_training_history.json`, and paired bootstrap fields for full-precision detail.
