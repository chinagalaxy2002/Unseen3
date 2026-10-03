# Joint-v3 阶段评测日志：4/5 splits

评测记录时间：2026-10-03 17:52（Asia/Shanghai）  
代码冻结提交：`bc01b54649debc2290fa4470f7c1ce747bb9c948`  
实验：Moment-DETR-TRM-GMR-Joint-v3，固定训练 50 epochs。

本轮是 exploratory follow-up，设计受到已查看的 Joint-v1/Joint-v2 未见语义结果启发。表中 U 指标仅用于当前冻结模型的事后报告，没有用于本轮训练、PCA、checkpoint selection 或阈值校准。

## 执行状态

- A1、A2_alt、A3、C1：均完成 50 epochs、Seen-only threshold calibration、test inference 和 metrics。
- C2_alt：日志快照时训练到 epoch 15/50，GPU0 训练进程仍在运行；best Seen-val mAP 为 22.96（epoch 6），门槛 23.81。尚无 C2_alt test 评测结果。
- 四个已完成 split 均未达到 Seen-val localization mAP floor，因此按照配置使用 Seen-val mAP fallback checkpoint；它们的 worst-semantic AUROC constrained selection 均未启用。
- 四个已完成 split 均报告 `nan_inf_detected=false`、`missing_feature_or_fallback=false`、`auc_valid_batch_fraction=1.0`。
- 未发现正在运行的 inference/evaluation 进程；本日志不终止 C2_alt 训练。

## 与原始 Moment-DETR-GMR 对比

AUROC 使用 0–1 标度；raw R1@0.5 使用百分比。差值为 Joint-v3 减原始基线。原始基线 AUROC 文件只保存四位小数概率，Joint-v3 使用 full-precision logit；微小差值应视为近似，不能解读为统计显著性。

| Split | Best epoch | Seen AUROC 基线 → V3 (Δ) | Unseen AUROC 基线 → V3 (Δ) | S+ raw R1@0.5 基线 → V3 | U+ raw R1@0.5 基线 → V3 |
|---|---:|---:|---:|---:|---:|
| A1 | 10 | 0.80439 → 0.69505 (−0.10933) | 0.49727 → 0.51131 (+0.01404) | 36.97% → 36.83% | 23.01% → 21.72% |
| A2_alt | 36 | 0.76905 → 0.73025 (−0.03880) | 0.55108 → 0.51767 (−0.03342) | 35.42% → 29.95% | 26.79% → 23.81% |
| A3 | 20 | 0.74880 → 0.72354 (−0.02526) | 0.56432 → 0.61715 (+0.05283) | 34.27% → 30.32% | 39.58% → 46.35% |
| C1 | 9 | 0.76097 → 0.69263 (−0.06834) | 0.56207 → 0.50135 (−0.06072) | 35.73% → 32.62% | 48.77% → 44.44% |

四个 split 等权宏平均：Seen AUROC 基线 0.77080、V3 0.71037（Δ −0.06043）；Unseen AUROC 基线 0.54369、V3 0.53687（Δ −0.00682）。Unseen AUROC 对比原始基线为 2 个 split 提升、2 个 split 下降。当前四 split 结果没有显示总体 unseen AUROC 改善，同时 Seen AUROC 宏平均明显降低。

## Seen-only 阈值下的拒绝诊断

| Split | Frozen threshold | U+ FRR | U− RR | Matched PairAcc |
|---|---:|---:|---:|---:|
| A1 | 0.865610 | 0.26237 | 0.29848 | 0.62500 |
| A2_alt | 0.998628 | 0.00595 | 0.03526 | 0.32911 |
| A3 | 0.986806 | 0.15104 | 0.27273 | 0.49612 |
| C1 | 0.859451 | 0.70370 | 0.71111 | 0.48611 |

阈值只由 Seen validation S+/S− 的 balanced accuracy 决定。A2_alt 的低 U+ FRR/U− RR 表示阈值下几乎全部接收，不能据此替代 AUROC 排序结论。

## Prediction consistency checks

对 A1、A2_alt、A3、C1 的 `test_predictions.jsonl` 重新按官方 partition 计算 Seen/Unseen AUROC，均与 `joint_v3_summary.json` 相同。Full-precision logit AUROC 与 probability AUROC 相同；所有 score 均为有限值、qid 无重复，且预测 qid 集合与原始基线相同。

## 每 split 文件

每个已完成 split 的机器可读结果位于 `results/moment_detr_trm_gmr_joint_v3/<SPLIT>/`：

- `joint_v3_summary.json`
- `diagnostics.json`
- `official_test_metrics.json`
- `training_meta.json`
- `threshold_frozen.json`
- `test_predictions.jsonl`

C2_alt 仍在训练；完成 50 epochs 并运行 test inference 前，不纳入五 split 宏平均或最终结论。
