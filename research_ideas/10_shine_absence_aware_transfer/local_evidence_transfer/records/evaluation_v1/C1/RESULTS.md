# C1: Moment local evidence, exploratory Unseen

Seed3407,10epochs, Seen-val selected/thresholded; references from immutable same-budget v2 predictions.

| Arm | Seen | U | Gap(pp) | ΔU vs B0(pp) | ΔU vs S1(pp) | U CI vs S1(pp) |
|---|---:|---:|---:|---:|---:|---|
| B0 | 0.7599 | 0.5780 | 18.19 | +0.00 | -1.16 | — |
| S1_saliency_only | 0.7651 | 0.5896 | 17.55 | +1.16 | +0.00 | — |
| Local_CF | 0.7653 | 0.5898 | 17.56 | +1.18 | +0.02 | [-0.08,+0.11] |
| Uniform_CF | 0.7654 | 0.5893 | 17.60 | +1.13 | -0.02 | [-0.13,+0.07] |
| Local_noCF | 0.7599 | 0.5779 | 18.20 | -0.01 | -1.16 | [-2.30,+0.03] |

CI is1000 paired video-cluster resamples conditional on one trained seed. Previously examined A1/C1 is exploratory. A smaller gap caused by Seen loss is not sufficient. Conditions, branch decomposition and fresh shuffle diagnostics are in RESULTS.json.
