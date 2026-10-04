# A1: v2/v3 exploratory formal test

All adapted arms: 10 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| canonical | canonical | 0.8034 | 0.4956 | -0.26 | +1.10 | — |
| B0 | 5 | 0.8060 | 0.4846 | +0.00 | +0.00 | — |
| S1_saliency_only | 5 | 0.8060 | 0.4805 | -0.00 | -0.40 | [-0.88, +0.09] |
| E1_rotated_exist_only | 7 | 0.7519 | 0.5044 | -5.42 | +1.98 | [-1.14, +5.04] |
| SE1_full | 3 | 0.7517 | 0.5017 | -5.43 | +1.72 | [-1.23, +4.49] |
| BCE_only | 3 | 0.7538 | 0.5020 | -5.23 | +1.75 | [-1.17, +4.49] |
| Pair_only | 1 | 0.7873 | 0.4991 | -1.87 | +1.45 | [+0.11, +2.81] |
| Weak_BCE_pair | 3 | 0.7759 | 0.5056 | -3.01 | +2.10 | [-0.25, +4.42] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
