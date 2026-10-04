# C1 Reduced BCE/pair supervision

seed3407; 10 epochs; Seen-only; epoch0 excluded from selection. V2 controls are frozen historical references with verified exposure matches.

| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δ vs B0 (pp) | Δ vs S1 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.7826 | 0.7808 | 0.7796 | 5 / 0.7844 | +0.00 | -0.57 |
| S1_saliency_only | 0.7901 | 0.7845 | 0.7846 | 1 / 0.7901 | +0.57 | +0.00 |
| SE1_full | 0.7599 | 0.7390 | 0.6764 | 1 / 0.7599 | -2.44 | -3.01 |
| BCE_only | 0.7675 | 0.7528 | 0.6868 | 1 / 0.7675 | -1.68 | -2.25 |
| Pair_only | 0.7667 | 0.7590 | 0.7416 | 1 / 0.7667 | -1.77 | -2.34 |
| Weak_BCE_pair | 0.7689 | 0.7609 | 0.7388 | 1 / 0.7689 | -1.55 | -2.12 |
| Low_BCE_only | 0.7890 | 0.7832 | 0.7820 | 1 / 0.7890 | +0.46 | -0.11 |
| Low_Pair_only | 0.7865 | 0.7798 | 0.7772 | 1 / 0.7865 | +0.21 | -0.36 |
| Low_BCE_pair | 0.7855 | 0.7790 | 0.7779 | 1 / 0.7855 | +0.11 | -0.46 |

These are Seen development results, not evidence of Unseen gain. Inspect gradient geometry, GMR loss, conditionals and localization before selecting a further repair.
