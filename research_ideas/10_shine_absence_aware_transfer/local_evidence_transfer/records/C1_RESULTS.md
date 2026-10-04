# C1: local evidence to existence, Seen-only

10epochs, seed3407; old controls from v2 with verified input/exposure. No formal test access.

| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / Seen AUC | Δ vs B0(pp) | Δ vs S1(pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.7826 | 0.7808 | 0.7796 | 5 / 0.7844 | +0.00 | -0.57 |
| S1_saliency_only | 0.7901 | 0.7845 | 0.7846 | 1 / 0.7901 | +0.57 | +0.00 |
| Local_CF | 0.7900 | 0.7844 | 0.7852 | 1 / 0.7900 | +0.57 | -0.00 |
| Uniform_CF | 0.7903 | 0.7845 | 0.7844 | 1 / 0.7903 | +0.59 | +0.03 |
| Local_noCF | 0.7825 | 0.7810 | 0.7796 | 5 / 0.7849 | +0.05 | -0.52 |

Local_CF vs Uniform_CF controls branch capacity/global encoder pooling; Local_CF vs Local_noCF controls saliency supervision. Local_CF vs old S1 measures adding the direct branch. These Seen results alone do not establish Unseen gain.
