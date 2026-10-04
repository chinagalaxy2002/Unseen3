# QD-DETR-GMR A1: interim snapshots

INTERIM: training is still running, budgets differ; seed3407; selected only by Seen-val. This is not the final50-epoch comparison.

| Arm | Seen | Unseen | Gap |
|---|---:|---:|---:|
| B0 | 0.7998 | 0.4983 | 0.3015 |
| S1_saliency_only | 0.8013 | 0.5008 | 0.3005 |
| canonical | 0.7860 | 0.5072 | 0.2788 |

S1−B0: ΔSeen +0.15pp; ΔUnseen +0.25pp, paired video CI [-0.82,+1.33]pp; gap reduction +0.10pp.

Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.
