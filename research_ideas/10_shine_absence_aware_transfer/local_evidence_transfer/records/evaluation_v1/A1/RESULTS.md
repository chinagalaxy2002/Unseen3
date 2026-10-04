# A1: Moment local evidence, exploratory Unseen

Seed3407,10epochs, Seen-val selected/thresholded; references from immutable same-budget v2 predictions.

| Arm | Seen | U | Gap(pp) | ΔU vs B0(pp) | ΔU vs S1(pp) | U CI vs S1(pp) |
|---|---:|---:|---:|---:|---:|---|
| B0 | 0.8060 | 0.4846 | 32.15 | +0.00 | +0.40 | — |
| S1_saliency_only | 0.8060 | 0.4805 | 32.55 | -0.40 | +0.00 | — |
| Local_CF | 0.8063 | 0.4799 | 32.64 | -0.47 | -0.06 | [-0.13,+0.01] |
| Uniform_CF | 0.8063 | 0.4805 | 32.59 | -0.41 | -0.01 | [-0.07,+0.05] |
| Local_noCF | 0.8061 | 0.4843 | 32.18 | -0.03 | +0.37 | [-0.13,+0.86] |

CI is1000 paired video-cluster resamples conditional on one trained seed. Previously examined A1/C1 is exploratory. A smaller gap caused by Seen loss is not sufficient. Conditions, branch decomposition and fresh shuffle diagnostics are in RESULTS.json.
