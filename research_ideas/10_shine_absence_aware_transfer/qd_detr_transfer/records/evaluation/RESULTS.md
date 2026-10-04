# QD-DETR-GMR + SHINE coarse/fine: A1/C1 50 epochs

| Split | Arm | Seen | Unseen | Gap |
|---|---|---:|---:|---:|
| A1 | canonical | 0.7860 | 0.5072 | 0.2788 |
| A1 | B0 | 0.7998 | 0.4983 | 0.3015 |
| A1 | S1_saliency_only | 0.8013 | 0.5008 | 0.3005 |
| C1 | canonical | 0.7800 | 0.5655 | 0.2145 |
| C1 | B0 | 0.7810 | 0.5741 | 0.2070 |
| C1 | S1_saliency_only | 0.7855 | 0.5846 | 0.2009 |
| Macro | canonical | 0.7830 | 0.5364 | 0.2467 |
| Macro | B0 | 0.7904 | 0.5362 | 0.2542 |
| Macro | S1_saliency_only | 0.7934 | 0.5427 | 0.2507 |

S1−B0 macro: ΔSeen +0.30pp; ΔUnseen +0.65pp; gap reduction +0.36pp.

Two-split single-seed exploratory comparison; gap reduction alone is not success. Per-split paired intervals and conditions in RESULTS.json.
