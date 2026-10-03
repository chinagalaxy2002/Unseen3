# Joint-v1 项目交接

## 项目问题

基准研究 **Semantic Novelty × Event Existence**：S+ / S− 是 seen semantic 下事件存在 / 不存在，U+ / U− 是 unseen semantic 下事件存在 / 不存在。训练只能使用 S+ / S−；checkpoint 选择与 existence threshold 只能使用 Seen validation；U+ / U− 是最终测试集。核心原则是“未见不等于不存在”。

## 当前方法

Moment-DETR-TRM-GMR-Joint-v1 联合了候选条件 phrase attention、candidate-wise gated refinement 和 evidence-aware existence head。代码中的 `L_exist` 直接监督 existence logit，计算图可回传到 existence head、candidate gate、phrase attention、phrase matcher 和 transformer。此项是代码图核验，不代表这些组件已实现稳定的 unseen 泛化。

Checkpoint 按 Seen validation 的 `MR-full-mAP` 选择。threshold 使用 Seen validation 的 S+ / S− score，优化 `0.5 × (TPR + TNR)`，即 balanced accuracy。A1 在 100 epoch 计划完成前手动暂停；best epoch 为 7，标注为 **manual stop after Seen-validation overfitting**，不视为 protocol bug，不要求重跑。日志有 83 次验证记录，`training_meta.json` 记为 82 epochs，这是 bookkeeping 差异。

## 核验结论

- Joint 五 split 的 `pred_exist_score` AUROC 已按 test JSONL 和样本级 prediction 独立重算，与每个 `joint_summary.json`、`diagnostics.json` 一致；预测完整覆盖测试 qid。
- Moment-DETR-GMR baseline 数据来自原始结果报告 `docs/reports/semantic_existence_multisplit_results.md` 和 per-split diagnostics；五 split prediction 覆盖与当前 release 完全相同的 qid。根目录 `metrics/` 属于 Flash DQ-CGP，不是该 baseline。
- 五 split macro AUROC：baseline Seen / Unseen / Gap = **0.751828 / 0.528680 / 0.223148**；Joint = **0.749851 / 0.538618 / 0.211233**。ΔUnseen **+0.009938**，ΔGap **−0.011915**。
- 逐 split 只有 A3、C1 同时满足 unseen AUROC 提高且 gap 缩小；A1、A2_alt、C2_alt 均未缓解。因此准确结论是 **split-dependent improvement**，不支持不加限定地声称 “Joint-v1 alleviates unseen-semantic AUROC degradation”。
- A2_alt 出现 over-accept（U+ FRR / U− RR = 0% / 0%）；C2_alt 出现 over-reject（95.65% / 97.64%）。threshold 改变决策点，但不能修复 AUROC 排序。
- Joint soft-gated localization 与 raw 指标的变化来自 gated window 单独裁剪、取整；对 raw windows 施加相同后处理后复现 soft 指标，candidate 排名没有改变。不能将差异称作 gate 改善定位。
- 主要问题是跨语义 existence discrimination 与 score calibration；现有结果不足以将失败归因到某一个模块。只有一个训练 seed，没有跨 seed 变异或差值显著性证据。

## 真实测试集数量

| Split | S+ | S− | U+ | U− | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A1 | 2218 | 1368 | 465 | 1119 | 5170 |
| A2_alt | 2798 | 1667 | 168 | 312 | 4945 |
| A3 | 2784 | 1723 | 192 | 594 | 5293 |
| C1 | 2799 | 1474 | 162 | 270 | 4705 |
| C2_alt | 2846 | 1490 | 115 | 254 | 4705 |

`EXPERIMENT_FREEZE.json` 中 A1 的 S− / U− 数量误写为 1244 / 1243；正确值为 1368 / 1119。总数仍为 5170，Joint summary 的计数正确。没有修改 dataset 或 freeze 文件。

## 已发现的记录错误

- `scripts/aggregate_multisplit_joint.py` 的 C2_alt TRM localization baseline 硬编码为 37.95 / 36.21 / 1.74；正确 Seen / Unseen / Gap 是 **36.120871 / 31.304348 / 4.816524**，取自 `experiments/trm_momentdetr_generalization/generalization_summary.json`。
- 旧 Joint README 的 C1 epoch / threshold `14 / 0.9329` 和 C2_alt `10 / 0.5484` 错误；正确值分别为 C1 `58 / 0.9964`、C2_alt `16 / 0.9225`。
- Split 描述应为 A2_alt `drink / pour`、A3 `run / walk`。
- A2_alt 的 `pred_exist_score` 只有 10 个不同值。按 score 重算 unseen AUROC 为 0.506067，按保存 logit 排序为 0.497281；报告表按 benchmark 指定的 score 指标。

## 关键文件

- `README.md`：已根据本轮审计更正的主结果说明。
- `EXPERIMENT_FREEZE.json`：协议记录，含上述 A1 计数错误。
- `MULTI_SPLIT_RESULT.md`、`multi_split_summary.json`：旧汇总；使用前留意主 README 的核验备注。
- `results/moment_detr_trm_gmr_joint/<split>/joint_summary.json`、`diagnostics.json`：机器可读 Joint 指标。
- `results/moment_detr_trm_gmr_joint/<split>/test_predictions.jsonl`：本地样本级 Joint predictions，可独立重算；大文件是否随不同工作副本提供需确认。
- `scripts/analyze_semantic_existence.py`、`training/moment_detr_trm_gmr_joint/infer.py`：阈值与评估实现。
- `scripts/aggregate_multisplit_joint.py`：含 C2_alt TRM baseline 硬编码错误；不要把生成结果作为未经检查的 canonical 比较。

## Git 状态与后续

主 README 修改已提交并推送到 GitHub `main`，commit `cb2cb37`（`Correct Joint-v1 experiment README`）。本次审计未改模型或数据，未训练，也未启动 GPU 实验。

交接后的建议：如需修复 `EXPERIMENT_FREEZE.json` 或 aggregate script，应先做单独、可审阅的 bookkeeping 修正；不要改写实验数据来匹配错误记录。任何总体改善论断都应保留 split-dependent 限定，并报告跨 seed 不确定性目前尚未评估。
