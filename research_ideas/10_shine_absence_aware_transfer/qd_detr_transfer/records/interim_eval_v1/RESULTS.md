# QD-DETR-GMR + SHINE coarse/fine: A1/C1 interim snapshots

Training remains active; this is not the final50-epoch result. A1 snapshots observed B0=28/S1=26 completed epochs; C1 both16. Checkpoints selected by Seen-val only.

| Split | Arm | Seen | Unseen | Gap |
|---|---|---:|---:|---:|
| A1 | B0 | 0.7998 | 0.4983 | 0.3015 |
| A1 | S1_saliency_only | 0.8013 | 0.5008 | 0.3005 |
| A1 | canonical | 0.7860 | 0.5072 | 0.2788 |
| C1 | B0 | 0.7810 | 0.5741 | 0.2070 |
| C1 | S1_saliency_only | 0.7855 | 0.5846 | 0.2009 |
| C1 | canonical | 0.7800 | 0.5655 | 0.2145 |
| Macro | canonical | 0.7830 | 0.5364 | 0.2467 |
| Macro | B0 | 0.7904 | 0.5362 | 0.2542 |
| Macro | S1_saliency_only | 0.7934 | 0.5427 | 0.2507 |

S1−B0 macro: ΔSeen +0.30pp; ΔUnseen +0.65pp; gap reduction +0.36pp.

Two-split single-seed exploratory comparison; gap reduction alone is not success. Per-split paired intervals and conditions in RESULTS.json.

Common-budget Seen-only check: A1 both selected snapshots equal the earliest best over the first26 completed epochs; C1 both equal earliest best over first16. Thus observed28/26 and16/16 training statuses yield snapshots equivalent to matched26/16 budget comparisons. See COMMON_BUDGET_CHECK.json. No checkpoint was changed based on test.
