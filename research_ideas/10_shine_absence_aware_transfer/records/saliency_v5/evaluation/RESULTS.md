# Coarse/fine-only vs baseline, 50 epochs

Fresh canonical initialization for each arm; seed3407; matched forwards and exposure; best trained epoch by Seen-val only. A1/C1 exploratory test.

| Split | B0 Seen / U | S1 Seen / U | B0 Gap (pp) | S1 Gap (pp) | Gap reduction (pp) | ΔU (pp) | ΔSeen (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.8045 / 0.4871 | 0.8086 / 0.4810 | 31.73 | 32.75 | -1.02 | -0.61 | +0.41 |
| C1 | 0.7599 / 0.5780 | 0.7651 / 0.5896 | 18.19 | 17.55 | +0.64 | +1.16 | +0.52 |
| Macro | 0.7822 / 0.5326 | 0.7868 / 0.5353 | 24.96 | 25.15 | -0.19 | +0.27 | +0.46 |

Positive gap reduction means a smaller Seen−Unseen gap; interpret jointly with ΔU and Seen retention. Per-split paired video-cluster CI/conditionals/localization/shuffle diagnostics are in split RESULTS.json. Macro describes two benchmark splits and has no independence-based CI.
