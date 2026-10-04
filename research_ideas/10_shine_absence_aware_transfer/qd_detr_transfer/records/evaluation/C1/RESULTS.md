# QD-DETR-GMR C1: 50 epochs

canonical initialization; seed3407; Seen-val selection; exploratory formal test.

| Arm | Seen | Unseen | Gap |
|---|---:|---:|---:|
| canonical | 0.7800 | 0.5655 | 0.2145 |
| B0 | 0.7810 | 0.5741 | 0.2070 |
| S1_saliency_only | 0.7855 | 0.5846 | 0.2009 |

S1−B0: ΔSeen +0.44pp; ΔUnseen +1.05pp, paired video CI [-0.22,+2.29]pp; gap reduction +0.61pp.

Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.
