# Moment-DETR: direct local evidence, formal A1/C1 exploratory evaluation

All new arms10epochs, seed3407. Selected checkpoints and thresholds use Seen-val only. References are frozen v2 matched10epoch B0/S1.

| Arm | A1 Seen | A1 U | C1 Seen | C1 U | Macro ΔSeen vs B0(pp) | Macro ΔU vs B0(pp) | Macro ΔU vs S1(pp) | Macro Gap reduction vs B0(pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.8060 | 0.4846 | 0.7599 | 0.5780 | +0.00 | +0.00 | -0.38 | +0.00 |
| S1_saliency_only | 0.8060 | 0.4805 | 0.7651 | 0.5896 | +0.26 | +0.38 | +0.00 | +0.12 |
| Local_CF | 0.8063 | 0.4799 | 0.7653 | 0.5898 | +0.28 | +0.36 | -0.02 | +0.07 |
| Uniform_CF | 0.8063 | 0.4805 | 0.7654 | 0.5893 | +0.29 | +0.36 | -0.02 | +0.07 |
| Local_noCF | 0.8061 | 0.4843 | 0.7599 | 0.5779 | +0.00 | -0.02 | -0.39 | -0.02 |

Local_CF vs Uniform_CF controls capacity and direct encoder pooling; Local_CF vs Local_noCF tests CF supervision. Compare Seen retention, U and conditionals together. Per-split paired CIs, residual/base decomposition and shuffled-video diagnostics are in split RESULTS.json. No new training or coefficient selection is triggered by this evaluation.
