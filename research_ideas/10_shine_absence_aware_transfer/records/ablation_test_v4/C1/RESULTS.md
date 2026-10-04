# C1: v4 low-supervision exploratory formal test

All adapted arms: 10 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| canonical | canonical | 0.7611 | 0.5683 | +0.12 | -0.96 | — |
| B0 | 5 | 0.7599 | 0.5780 | +0.00 | +0.00 | — |
| S1_saliency_only | 1 | 0.7651 | 0.5896 | +0.52 | +1.16 | [+0.00, +2.31] |
| Low_BCE_only | 1 | 0.7634 | 0.5874 | +0.35 | +0.94 | [-0.23, +2.13] |
| Low_Pair_only | 1 | 0.7617 | 0.5869 | +0.17 | +0.89 | [-0.29, +2.07] |
| Low_BCE_pair | 1 | 0.7602 | 0.5850 | +0.03 | +0.71 | [-0.49, +1.98] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
