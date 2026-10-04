# A1 Reduced BCE/pair supervision

seed3407; 10 epochs; Seen-only; epoch0 excluded from selection. V2 controls are frozen historical references with verified exposure matches.

| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δ vs B0 (pp) | Δ vs S1 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8397 | 0.8423 | 0.8439 | 5 / 0.8456 | +0.00 | -0.17 |
| S1_saliency_only | 0.8412 | 0.8454 | 0.8452 | 5 / 0.8473 | +0.17 | +0.00 |
| SE1_full | 0.7743 | 0.7810 | 0.7715 | 3 / 0.7810 | -6.46 | -6.63 |
| BCE_only | 0.7758 | 0.7832 | 0.7761 | 3 / 0.7832 | -6.25 | -6.41 |
| Pair_only | 0.8331 | 0.8230 | 0.8149 | 1 / 0.8331 | -1.25 | -1.42 |
| Weak_BCE_pair | 0.8058 | 0.8108 | 0.8006 | 3 / 0.8108 | -3.48 | -3.65 |
| Low_BCE_only | 0.8374 | 0.8415 | 0.8384 | 4 / 0.8418 | -0.38 | -0.55 |
| Low_Pair_only | 0.8398 | 0.8430 | 0.8400 | 4 / 0.8435 | -0.21 | -0.38 |
| Low_BCE_pair | 0.8368 | 0.8404 | 0.8338 | 3 / 0.8404 | -0.53 | -0.69 |

These are Seen development results, not evidence of Unseen gain. Inspect gradient geometry, GMR loss, conditionals and localization before selecting a further repair.
