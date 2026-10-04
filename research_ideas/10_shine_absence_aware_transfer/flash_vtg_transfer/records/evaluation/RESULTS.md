# FlashVTG-GMR + SHINE coarse/fine: A1/C1 50 epochs

| Split | Arm | Seen | Unseen | Gap |
|---|---|---:|---:|---:|
| A1 | canonical | 0.8253 | 0.4703 | 0.3550 |
| A1 | B0 | 0.8288 | 0.4657 | 0.3631 |
| A1 | S1_saliency_only | 0.8282 | 0.4629 | 0.3653 |
| C1 | canonical | 0.7716 | 0.6679 | 0.1037 |
| C1 | B0 | 0.7658 | 0.6529 | 0.1129 |
| C1 | S1_saliency_only | 0.7783 | 0.6068 | 0.1715 |
| Macro | canonical | 0.7984 | 0.5691 | 0.2293 |
| Macro | B0 | 0.7973 | 0.5593 | 0.2380 |
| Macro | S1_saliency_only | 0.8032 | 0.5348 | 0.2684 |

S1−B0 macro: ΔSeen +0.59pp; ΔUnseen -2.45pp; gap reduction -3.04pp.

Two-split single-seed exploratory comparison; gap reduction alone is not success. Per-split paired intervals and conditions in RESULTS.json.
