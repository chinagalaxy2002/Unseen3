# Seen / Novel pooled AUROC and gap

Gap is Seen-validation AUROC minus exploratory inner Novel-dev AUROC. Values come from each successful, non-smoke natural-input task; shuffled controls are excluded. The official U split was not accessed. Folds share videos and are not independent.

| Idea | Fold | Arm | Seen AUROC | Novel-dev AUROC | Seen−Novel gap |
|---|---|---|---:|---:|---:|
| 01 | A1_action_01 | BCE | 0.8719 | 0.6277 | 24.42 pp |
| 01 | A1_action_01 | BCE_pair | 0.8735 | 0.6157 | 25.78 pp |
| 01 | C1_composition_01 | BCE | 0.8117 | 0.5536 | 25.81 pp |
| 01 | C1_composition_01 | BCE_pair | 0.8112 | 0.5546 | 25.66 pp |
| 02 | A1_action_01 | centered | 0.6849 | 0.6811 | 0.37 pp |
| 02 | A1_action_01 | raw | 0.6774 | 0.6798 | -0.24 pp |
| 02 | C1_composition_01 | centered | 0.6176 | 0.5478 | 6.98 pp |
| 02 | C1_composition_01 | raw | 0.6128 | 0.5427 | 7.01 pp |
| 04 | A1_action_01 | M0_natural_duplicate | 0.8726 | 0.6169 | 25.57 pp |
| 04 | A1_action_01 | M1_original_ratio | 0.8731 | 0.6229 | 25.02 pp |
| 04 | A1_action_01 | M2_balanced_query | 0.8554 | 0.6472 | 20.82 pp |
| 04 | C1_composition_01 | M0_natural_duplicate | 0.8093 | 0.5471 | 26.22 pp |
| 04 | C1_composition_01 | M1_original_ratio | 0.8084 | 0.5482 | 26.02 pp |
| 04 | C1_composition_01 | M2_balanced_query | 0.8066 | 0.5498 | 25.68 pp |
| 05 | A1_action_01 | P_pooled | 0.8901 | 0.6707 | 21.95 pp |
| 05 | A1_action_01 | R_raw_sequence | 0.8826 | 0.6335 | 24.91 pp |
| 05 | C1_composition_01 | P_pooled | 0.8021 | 0.5529 | 24.92 pp |
| 05 | C1_composition_01 | R_raw_sequence | 0.8362 | 0.5925 | 24.38 pp |
| 06 | A1_action_01 | A0_bag_BCE | 0.8865 | 0.6434 | 24.31 pp |
| 06 | A1_action_01 | A2_verified_negative_local | 0.8768 | 0.6346 | 24.22 pp |
| 06 | C1_composition_01 | A0_bag_BCE | 0.8050 | 0.5492 | 25.58 pp |
| 06 | C1_composition_01 | A2_verified_negative_local | 0.8093 | 0.5490 | 26.03 pp |
| 07 | A1_action_01 | B0_natural_pair_replay | 0.8782 | 0.7107 | 16.76 pp |
| 07 | A1_action_01 | B1_pair_ERM | 0.8706 | 0.7107 | 15.99 pp |
| 07 | C1_composition_01 | B0_natural_pair_replay | 0.8504 | 0.5751 | 27.54 pp |
| 07 | C1_composition_01 | B1_pair_ERM | 0.8541 | 0.5923 | 26.18 pp |

Gap is descriptive on different partitions; it is not a paired effect estimate. Use each direction’s clustered uncertainty and conditional metrics before interpreting a smaller gap as improvement.
