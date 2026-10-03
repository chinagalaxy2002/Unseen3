# P3 inner readout results

Seen-only inner development; formal U was not read.

Completed 8/36 runs. Partial results do not change the frozen queue.

| Fold | Variant | Epoch | Seen base | Seen learned | Novel base | Novel learned | ΔNovel | Deployment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1_action_01 | R1 | 45 | 0.86009 | 0.87465 | 0.61604 | 0.60418 | -0.01186 | learned |
| A1_action_01 | R1_matched | 38 | 0.86009 | 0.87994 | 0.61604 | 0.60746 | -0.00858 | learned |
| A1_action_02 | R1 | 34 | 0.83132 | 0.84557 | 0.49944 | 0.49516 | -0.00427 | learned |
| A1_action_02 | R1_matched | 43 | 0.83132 | 0.84771 | 0.49944 | 0.48887 | -0.01057 | learned |
| C1_composition_01 | R1 | 2 | 0.81377 | 0.80940 | 0.55533 | 0.54688 | -0.00845 | baseline_fallback |
| C1_composition_01 | R1_matched | 2 | 0.81377 | 0.80857 | 0.55533 | 0.55090 | -0.00442 | baseline_fallback |
| C1_composition_02 | R1 | 1 | 0.84279 | 0.84474 | 0.51936 | 0.48702 | -0.03234 | learned |
| C1_composition_02 | R1_matched | 1 | 0.84279 | 0.84415 | 0.51936 | 0.48319 | -0.03617 | learned |

| Variant | Folds | Macro ΔNovel | Macro ΔSeen | Positive novel folds | Parameters |
|---|---:|---:|---:|---:|---:|
| R1 | 4/4 | -0.01423 | +0.00660 | 0 | 50049 |
| R1_matched | 4/4 | -0.01494 | +0.00810 | 0 | 200433 |

R2 and R3 have exactly equal parameter counts. R1_matched controls pooled readout capacity.
Learned and baseline-fallback policies are reported separately; fallback does not erase failed learned models.
R4 is pending temporal provenance; no conclusion about local-video evidence follows from this table.
These are seed-3407 development results, not the three-seed G2 gate or confirmatory evidence.
