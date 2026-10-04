# FlashVTG-GMR A1: 50 epochs

canonical initialization; seed3407; Seen-val selection; exploratory formal test.

| Arm | Seen | Unseen | Gap |
|---|---:|---:|---:|
| canonical | 0.8253 | 0.4703 | 0.3550 |
| B0 | 0.8288 | 0.4657 | 0.3631 |
| S1_saliency_only | 0.8282 | 0.4629 | 0.3653 |

S1−B0: ΔSeen -0.06pp; ΔUnseen -0.28pp, paired video CI [-1.12,+0.59]pp; gap reduction -0.22pp.

Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.
