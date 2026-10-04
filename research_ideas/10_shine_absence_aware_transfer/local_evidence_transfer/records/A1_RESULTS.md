# A1: local evidence to existence, Seen-only

10epochs, seed3407; old controls from v2 with verified input/exposure. No formal test access.

| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / Seen AUC | Δ vs B0(pp) | Δ vs S1(pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8397 | 0.8423 | 0.8439 | 5 / 0.8456 | +0.00 | -0.17 |
| S1_saliency_only | 0.8412 | 0.8454 | 0.8452 | 5 / 0.8473 | +0.17 | +0.00 |
| Local_CF | 0.8412 | 0.8453 | 0.8452 | 5 / 0.8473 | +0.16 | -0.00 |
| Uniform_CF | 0.8412 | 0.8455 | 0.8460 | 5 / 0.8468 | +0.11 | -0.05 |
| Local_noCF | 0.8400 | 0.8427 | 0.8432 | 5 / 0.8458 | +0.02 | -0.15 |

Local_CF vs Uniform_CF controls branch capacity/global encoder pooling; Local_CF vs Local_noCF controls saliency supervision. Local_CF vs old S1 measures adding the direct branch. These Seen results alone do not establish Unseen gain.
