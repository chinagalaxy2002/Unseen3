# A1/C1 v4 exploratory formal Seen/Unseen evaluation

All adapted arms: 10 epochs, seed3407; checkpoint selection and thresholds use Seen-val only. Formal test is exploratory. Canonical is an initialization reference, not a 10-epoch adapted arm.

| Arm | A1 Seen | A1 U | C1 Seen | C1 U | Macro ΔSeen vs B0 (pp) | Macro ΔU vs B0 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.8034 | 0.4956 | 0.7611 | 0.5683 | -0.07 | +0.07 |
| B0 | 0.8060 | 0.4846 | 0.7599 | 0.5780 | +0.00 | +0.00 |
| S1_saliency_only | 0.8060 | 0.4805 | 0.7651 | 0.5896 | +0.26 | +0.38 |
| Low_BCE_only | 0.8018 | 0.4901 | 0.7634 | 0.5874 | -0.04 | +0.75 |
| Low_Pair_only | 0.8014 | 0.4872 | 0.7617 | 0.5869 | -0.15 | +0.58 |
| Low_BCE_pair | 0.7999 | 0.4929 | 0.7602 | 0.5850 | -0.29 | +0.77 |

Per-split intervals, conditions, localization and fresh shuffled-video diagnostics are in A1/RESULTS.json and C1/RESULTS.json. Macro is descriptive; splits share benchmark/video sources. Gap reduction alone does not establish success. Weights and Seen-val selected checkpoints remain unchanged by this evaluation.
