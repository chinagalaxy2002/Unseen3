# C1: v2/v3 exploratory formal test

All adapted arms: 10 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| canonical | canonical | 0.7611 | 0.5683 | +0.12 | -0.96 | — |
| B0 | 5 | 0.7599 | 0.5780 | +0.00 | +0.00 | — |
| S1_saliency_only | 1 | 0.7651 | 0.5896 | +0.52 | +1.16 | [+0.00, +2.31] |
| E1_rotated_exist_only | 1 | 0.7296 | 0.5571 | -3.04 | -2.09 | [-4.27, +0.14] |
| SE1_full | 1 | 0.7365 | 0.5672 | -2.34 | -1.08 | [-3.37, +1.33] |
| BCE_only | 1 | 0.7433 | 0.5696 | -1.67 | -0.84 | [-2.99, +1.39] |
| Pair_only | 1 | 0.7402 | 0.5749 | -1.97 | -0.30 | [-2.06, +1.64] |
| Weak_BCE_pair | 1 | 0.7403 | 0.5672 | -1.97 | -1.07 | [-3.02, +0.97] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
