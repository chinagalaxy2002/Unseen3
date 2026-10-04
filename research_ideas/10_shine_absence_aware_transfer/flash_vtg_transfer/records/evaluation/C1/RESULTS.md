# FlashVTG-GMR C1: 50 epochs

canonical initialization; seed3407; Seen-val selection; exploratory formal test.

| Arm | Seen | Unseen | Gap |
|---|---:|---:|---:|
| canonical | 0.7716 | 0.6679 | 0.1037 |
| B0 | 0.7658 | 0.6529 | 0.1129 |
| S1_saliency_only | 0.7783 | 0.6068 | 0.1715 |

S1−B0: ΔSeen +1.24pp; ΔUnseen -4.62pp, paired video CI [-7.22,-2.06]pp; gap reduction -5.86pp.

Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.
