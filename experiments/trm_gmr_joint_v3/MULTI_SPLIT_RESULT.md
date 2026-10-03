# Joint-v3 exploratory results

Exploratory follow-up motivated by inspected Joint-v1/v2 U results. U was excluded from training, grouping decisions, checkpoint selection, reference localization floors, and calibration.

| split | status | selection_mode | localization_constraint_satisfied | fallback_checkpoint_used | best_epoch | epochs_trained | threshold | joint_v1_unseen_auroc | unseen_auroc | delta_v3_vs_v1_unseen_auroc | seen_auroc | seen_unseen_gap | U+_FRR | U-_RR | matched_pair_acc | U+_raw_R1@0.5 | S+_raw_R1@0.5 | best_seen_worst_semantic_auroc | best_seen_val_mAP | localization_floor_mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | completed | fallback_seen_mAP | False | True | 10 | 50 | 0.865610 | 0.462494 | 0.511307 | 0.048813 | 0.695052 | 0.183744 | 0.262366 | 0.298481 | 0.625000 | 21.720430 | 36.834986 | 0.428571 | 22.640000 | 23.700000 |
| A2_alt | completed | fallback_seen_mAP | False | True | 36 | 50 | 0.998628 | 0.506067 | 0.517666 | 0.011600 | 0.730245 | 0.212579 | 0.005952 | 0.035256 | 0.329114 | 23.809524 | 29.949964 | 0.416667 | 22.380000 | 25.490000 |
| A3 | completed | fallback_seen_mAP | False | True | 20 | 50 | 0.986806 | 0.671327 | 0.617152 | -0.054175 | 0.723543 | 0.106391 | 0.151042 | 0.272727 | 0.496124 | 46.354167 | 30.316092 | 0.228571 | 21.980000 | 23.640000 |
| C1 | completed | fallback_seen_mAP | False | True | 9 | 50 | 0.859451 | 0.602481 | 0.501349 | -0.101132 | 0.692625 | 0.191276 | 0.703704 | 0.711111 | 0.486111 | 44.444444 | 32.618792 | 0.316017 | 22.670000 | 23.930000 |
| C2_alt | completed | fallback_seen_mAP | False | True | 17 | 50 | 0.920745 | 0.450719 | 0.496063 | 0.045344 | 0.720479 | 0.224416 | 0.826087 | 0.889764 | 0.484848 | 30.434783 | 33.204498 | 0.277778 | 23.170000 | 23.810000 |

Completed splits: 5/5. Improved: 3; degraded: 2.

Macros include explicitly labelled fallbacks; floor failures are not claimed as constraint successes.

macro_seen_auroc: 0.7123889473545374

macro_unseen_auroc: 0.5287075598009654

macro_seen_unseen_gap: 0.18368138755357183

macro_delta_v3_vs_v1: -0.00990997787979142
