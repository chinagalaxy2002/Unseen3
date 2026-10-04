# A1: v4 low-supervision exploratory formal test

All adapted arms: 10 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| canonical | canonical | 0.8034 | 0.4956 | -0.26 | +1.10 | — |
| B0 | 5 | 0.8060 | 0.4846 | +0.00 | +0.00 | — |
| S1_saliency_only | 5 | 0.8060 | 0.4805 | -0.00 | -0.40 | [-0.88, +0.09] |
| Low_BCE_only | 4 | 0.8018 | 0.4901 | -0.42 | +0.55 | [-0.20, +1.41] |
| Low_Pair_only | 4 | 0.8014 | 0.4872 | -0.46 | +0.26 | [-0.47, +1.04] |
| Low_BCE_pair | 3 | 0.7999 | 0.4929 | -0.61 | +0.83 | [-0.13, +1.91] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
