# QD-DETR-GMR A1: 50 epochs

canonical initialization; seed3407; Seen-val selection; exploratory formal test.

| Arm | Seen | Unseen | Gap |
|---|---:|---:|---:|
| canonical | 0.7860 | 0.5072 | 0.2788 |
| B0 | 0.7998 | 0.4983 | 0.3015 |
| S1_saliency_only | 0.8013 | 0.5008 | 0.3005 |

S1−B0: ΔSeen +0.15pp; ΔUnseen +0.25pp, paired video CI [-0.82,+1.33]pp; gap reduction +0.10pp.

Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.
