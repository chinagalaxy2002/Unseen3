# P3 inner readout results

Seen-only inner development; formal U was not read.

Completed 36/36 runs. Partial results do not change the frozen queue.

| Fold | Variant | Epoch | Seen base | Seen learned | Novel base | Novel learned | ΔNovel | Deployment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1_action_01 | Cq | 37 | 0.86009 | 0.89332 | 0.61604 | 0.65005 | +0.03402 | learned |
| A1_action_01 | Cv | 40 | 0.86009 | 0.56289 | 0.61604 | 0.48119 | -0.13485 | baseline_fallback |
| A1_action_01 | D_no_anchor | 24 | 0.86009 | 0.87492 | 0.61604 | 0.61568 | -0.00036 | learned |
| A1_action_01 | D_no_bound | 38 | 0.86009 | 0.88329 | 0.61604 | 0.62429 | +0.00825 | learned |
| A1_action_01 | D_v4 | 45 | 0.86009 | 0.89062 | 0.61604 | 0.60476 | -0.01128 | learned |
| A1_action_01 | R1 | 45 | 0.86009 | 0.87465 | 0.61604 | 0.60418 | -0.01186 | learned |
| A1_action_01 | R1_matched | 38 | 0.86009 | 0.87994 | 0.61604 | 0.60746 | -0.00858 | learned |
| A1_action_01 | R2 | 47 | 0.86009 | 0.90658 | 0.61604 | 0.59903 | -0.01701 | learned |
| A1_action_01 | R3 | 20 | 0.86009 | 0.89943 | 0.61604 | 0.68083 | +0.06479 | learned |
| A1_action_02 | Cq | 14 | 0.83132 | 0.88338 | 0.49944 | 0.55780 | +0.05837 | learned |
| A1_action_02 | Cv | 13 | 0.83132 | 0.54823 | 0.49944 | 0.45878 | -0.04065 | baseline_fallback |
| A1_action_02 | D_no_anchor | 31 | 0.83132 | 0.84153 | 0.49944 | 0.49033 | -0.00911 | learned |
| A1_action_02 | D_no_bound | 22 | 0.83132 | 0.85188 | 0.49944 | 0.47672 | -0.02272 | learned |
| A1_action_02 | D_v4 | 22 | 0.83132 | 0.85245 | 0.49944 | 0.48009 | -0.01934 | learned |
| A1_action_02 | R1 | 34 | 0.83132 | 0.84557 | 0.49944 | 0.49516 | -0.00427 | learned |
| A1_action_02 | R1_matched | 43 | 0.83132 | 0.84771 | 0.49944 | 0.48887 | -0.01057 | learned |
| A1_action_02 | R2 | 46 | 0.83132 | 0.88671 | 0.49944 | 0.53104 | +0.03160 | learned |
| A1_action_02 | R3 | 17 | 0.83132 | 0.90499 | 0.49944 | 0.51282 | +0.01338 | learned |
| C1_composition_01 | Cq | 8 | 0.81377 | 0.84948 | 0.55533 | 0.56437 | +0.00904 | learned |
| C1_composition_01 | Cv | 7 | 0.81377 | 0.56383 | 0.55533 | 0.50899 | -0.04633 | baseline_fallback |
| C1_composition_01 | D_no_anchor | 1 | 0.81377 | 0.81407 | 0.55533 | 0.55592 | +0.00059 | learned |
| C1_composition_01 | D_no_bound | 1 | 0.81377 | 0.81389 | 0.55533 | 0.55493 | -0.00039 | learned |
| C1_composition_01 | D_v4 | 1 | 0.81377 | 0.81392 | 0.55533 | 0.55493 | -0.00039 | learned |
| C1_composition_01 | R1 | 2 | 0.81377 | 0.80940 | 0.55533 | 0.54688 | -0.00845 | baseline_fallback |
| C1_composition_01 | R1_matched | 2 | 0.81377 | 0.80857 | 0.55533 | 0.55090 | -0.00442 | baseline_fallback |
| C1_composition_01 | R2 | 50 | 0.81377 | 0.80636 | 0.55533 | 0.55326 | -0.00206 | baseline_fallback |
| C1_composition_01 | R3 | 5 | 0.81377 | 0.82784 | 0.55533 | 0.55081 | -0.00452 | learned |
| C1_composition_02 | Cq | 27 | 0.84279 | 0.88218 | 0.51936 | 0.54447 | +0.02511 | learned |
| C1_composition_02 | Cv | 7 | 0.84279 | 0.56783 | 0.51936 | 0.62170 | +0.10234 | baseline_fallback |
| C1_composition_02 | D_no_anchor | 0 | 0.84279 | 0.84279 | 0.51936 | 0.51936 | +0.00000 | baseline_fallback |
| C1_composition_02 | D_no_bound | 1 | 0.84279 | 0.84308 | 0.51936 | 0.50681 | -0.01255 | learned |
| C1_composition_02 | D_v4 | 1 | 0.84279 | 0.84308 | 0.51936 | 0.50681 | -0.01255 | learned |
| C1_composition_02 | R1 | 1 | 0.84279 | 0.84474 | 0.51936 | 0.48702 | -0.03234 | learned |
| C1_composition_02 | R1_matched | 1 | 0.84279 | 0.84415 | 0.51936 | 0.48319 | -0.03617 | learned |
| C1_composition_02 | R2 | 42 | 0.84279 | 0.86059 | 0.51936 | 0.50234 | -0.01702 | learned |
| C1_composition_02 | R3 | 45 | 0.84279 | 0.86417 | 0.51936 | 0.49277 | -0.02660 | learned |

| Variant | Folds | Macro ΔNovel | Macro ΔSeen | Positive novel folds | Parameters |
|---|---:|---:|---:|---:|---:|
| R1 | 4/4 | -0.01423 | +0.00660 | 0 | 50049 |
| R1_matched | 4/4 | -0.01494 | +0.00810 | 0 | 200433 |
| R2 | 4/4 | -0.00112 | +0.02807 | 1 | 199809 |
| R3 | 4/4 | +0.01176 | +0.03712 | 2 | 199809 |
| Cq | 4/4 | +0.03163 | +0.04009 | 4 | 83329 |
| Cv | 4/4 | -0.02987 | -0.27630 | 1 | 382849 |
| D_v4 | 4/4 | -0.01089 | +0.01302 | 0 | 17091 |
| D_no_bound | 4/4 | -0.00685 | +0.01104 | 1 | 17091 |
| D_no_anchor | 4/4 | -0.00222 | +0.00633 | 1 | 17091 |

R2 and R3 have exactly equal parameter counts. R1_matched controls pooled readout capacity.
Learned and baseline-fallback policies are reported separately; fallback does not erase failed learned models.
R4 is pending temporal provenance; no conclusion about local-video evidence follows from this table.
These are seed-3407 development results, not the three-seed G2 gate or confirmatory evidence.
