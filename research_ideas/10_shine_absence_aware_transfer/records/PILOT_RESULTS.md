# Idea 10：单 seed 两折训练结果

已完成。Seed 3407，每 arm 10 epochs；按 Seen-val existence AUROC 选 epoch。Novel 为 strict-inner Novel-dev，正式 U 未参与。

| Fold | Arm | Epoch | Seen AUROC | Novel AUROC | Raw R1@.5 Seen / Novel |
|---|---|---:|---:|---:|---:|
| A1_action_01 | baseline | 3 | 0.8604 | 0.6122 | 0.3688 / 0.3133 |
| A1_action_01 | shine | 3 | 0.8724 | 0.6541 | 0.3869 / 0.3067 |
| C1_composition_01 | baseline | 1 | 0.8070 | 0.5455 | 0.3626 / 0.3585 |
| C1_composition_01 | shine | 2 | 0.8214 | 0.5813 | 0.3569 / 0.3585 |

两折等权 macro：SHINE−微调 baseline 的 Seen Δ=+1.32 pp，Novel-dev Δ=+3.88 pp。

| Fold | Novel ΔAUROC (pp) | Paired video-cluster 95% CI (pp) |
|---|---:|---:|
| A1_action_01 | +4.18 | [+2.38, +6.09] |
| C1_composition_01 | +3.58 | [+0.30, +7.01] |

上述 CI 是固定单 seed 训练结果的共享视频重采样，不包含重训不确定性；两折也不能视为独立域。

损失、支持覆盖、条件排序、阈值、gated R@.5、显存及时间详见 [RESULTS.json](RESULTS.json)。Raw R@IoU 为直接 top1 筛查，不是 official mAP。

本轮采用用户确认的 batch 轮换 absence 和强制编辑距离链；沿用当前 GMR 名义时间映射。额外监督/forward 预算与 baseline 不同，尚未拆分 temporal 与 existence loss 的机制增量。单 seed 两折不支持跨 backbone 稳定泛化结论。

## 条件排序与定位边界

| Fold | Same-query PairAcc baseline→SHINE | Same-video PairAcc baseline→SHINE | Novel raw R1@.5 |
|---|---:|---:|---:|
| A1_action_01 | 0.4389→0.4058 | 0.7063→0.7063 | 0.3133→0.3067 |
| C1_composition_01 | 0.5080→0.6747 | 0.6711→0.6711 | 0.3585→0.3585 |

A1/C1 的 same-query 变化方向不同，same-video PairAcc 均不变，Novel raw 定位未提高。当前证据支持自然 pooled existence AUROC 的改善，不能直接归因为统一的视频条件证据增强；temporal 与 rotated existence loss 的作用仍需单 seed 消融区分。
