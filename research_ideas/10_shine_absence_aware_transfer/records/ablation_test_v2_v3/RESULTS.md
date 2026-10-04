# A1/C1 v2/v3 exploratory formal Seen/Unseen evaluation

All adapted arms: 10 epochs, seed3407; checkpoint selection and thresholds use Seen-val only. Formal test is exploratory. Canonical is an initialization reference, not a 10-epoch adapted arm.

| Arm | A1 Seen | A1 U | C1 Seen | C1 U | Macro ΔSeen vs B0 (pp) | Macro ΔU vs B0 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.8034 | 0.4956 | 0.7611 | 0.5683 | -0.07 | +0.07 |
| B0 | 0.8060 | 0.4846 | 0.7599 | 0.5780 | +0.00 | +0.00 |
| S1_saliency_only | 0.8060 | 0.4805 | 0.7651 | 0.5896 | +0.26 | +0.38 |
| E1_rotated_exist_only | 0.7519 | 0.5044 | 0.7296 | 0.5571 | -4.23 | -0.05 |
| SE1_full | 0.7517 | 0.5017 | 0.7365 | 0.5672 | -3.89 | +0.32 |
| BCE_only | 0.7538 | 0.5020 | 0.7433 | 0.5696 | -3.45 | +0.45 |
| Pair_only | 0.7873 | 0.4991 | 0.7402 | 0.5749 | -1.92 | +0.57 |
| Weak_BCE_pair | 0.7759 | 0.5056 | 0.7403 | 0.5672 | -2.49 | +0.51 |

Per-split intervals, conditions, localization and fresh shuffled-video diagnostics are in A1/RESULTS.json and C1/RESULTS.json. Macro is descriptive; splits share benchmark/video sources. Gap reduction alone does not establish success. V4 weights were fixed from Seen results before this evaluation and will not be adjusted using these test outcomes.
