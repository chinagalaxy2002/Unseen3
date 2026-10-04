# A1: 50-epoch saliency-only exploratory formal test

Both matched arms: 50 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| B0 | 43 | 0.8045 | 0.4871 | +0.00 | +0.00 | — |
| S1_saliency_only | 35 | 0.8086 | 0.4810 | +0.41 | -0.61 | [-1.40, +0.19] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
