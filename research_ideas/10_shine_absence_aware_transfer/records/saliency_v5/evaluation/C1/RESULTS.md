# C1: 50-epoch saliency-only exploratory formal test

Both matched arms: 50 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.

| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| B0 | 5 | 0.7599 | 0.5780 | +0.00 | +0.00 | — |
| S1_saliency_only | 1 | 0.7651 | 0.5896 | +0.52 | +1.16 | [+0.00, +2.31] |

CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.
