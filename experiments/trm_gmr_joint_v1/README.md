# Moment-DETR-TRM-GMR-Joint-v1：实验与核验结果

本目录记录 Joint-v1 在 Charades-STA `semantic_existence_v2` 五个 split 上的实验配置和结果。基准区分语义是否见过与事件是否存在：S+ / S− 为 seen 语义的正 / 负样本，U+ / U− 为 unseen 语义的正 / 负样本。训练只用 S+ / S−；模型选择和 existence threshold 只用 Seen validation；U+ / U− 只用于最终测试。

核心问题是模型能否在 query 语义未见时区分事件存在与否。AUROC 衡量正负样本排序能力；阈值指标另外反映实际 accept / reject 行为。

## 1. 方法和评估协议

Joint-v1 将 Moment-DETR 定位、TRM phrase matching 和 GMR existence prediction 联合训练。模型对每个候选片段计算 candidate-conditioned phrase attention，以 candidate-wise gate 调整定位 logit，再把候选状态、phrase support、gate 和定位 margin 输入 evidence-aware existence head。`L_exist` 是直接作用于 existence logit 的 BCE；实现中没有在此路径上 detach，梯度可传至 existence head、候选 gate、phrase attention、phrase matcher 和 transformer。

每个 epoch 的 checkpoint 按 Seen validation 的 `MR-full-mAP` 选择。推理阈值由 Seen validation 的 S+ / S− score 选择，最大化 `0.5 × (TPR + TNR)`，也就是 balanced-accuracy threshold。测试集的 unseen 样本不用于选模或阈值标定。

A1 在预定的 100 个 epoch 完成前由人工暂停。best checkpoint 仍是第 7 epoch，按 Seen validation 选择；阈值也只使用 Seen validation。因此记录为 **manual stop after Seen-validation overfitting**，不视为 protocol bug，也不因训练不足 100 epoch 要求重跑。训练日志含 83 次验证记录，而 `training_meta.json` 的 `epochs_trained` 为 82；这是轮数记录差异，不影响 best epoch。

## 2. 文件和脚本

- `METHOD_SPEC.md`：模型设计说明。
- `EXPERIMENT_FREEZE.json`：协议和超参数记录。注意其中 A1 的测试集计数有误，详见下方真实数据计数。
- `MULTI_SPLIT_RESULT.md`、`multi_split_summary.json`：已有结果汇总。逐 split Joint 数值与 `joint_summary.json` 一致；如与本 README 的核验说明冲突，以样本级重算和本 README 的更正为准。
- `results/moment_detr_trm_gmr_joint/<split>/joint_summary.json`：核心指标。
- `results/moment_detr_trm_gmr_joint/<split>/diagnostics.json`：四象限指标和定位诊断。
- `results/moment_detr_trm_gmr_joint/<split>/test_predictions.jsonl`：本地保留的样本级预测，用于独立重算。
- `results/moment_detr_trm_gmr_joint/<split>/threshold_frozen.json`、`training_meta.json`：阈值和选模记录。

常用脚本：

- `scripts/train_trm_gmr_joint.sh <SPLIT>`：训练一个 split。
- `scripts/infer_trm_gmr_joint.sh <SPLIT>`：从 Seen validation 标定阈值并评测四象限。
- `scripts/aggregate_multisplit_joint.py`：生成多 split 汇总。脚本中的 Moment-DETR-GMR AUROC 和 TRM localization baseline 是硬编码值；其中 C2_alt TRM baseline 有误，不能将该脚本视为 canonical baseline 来源。
- `scripts/smoke_test_trm_gmr_joint.py`：模型和梯度路径自检。

checkpoint 与大体积 prediction JSONL 保留在运行环境中；仓库中的 summary 和 diagnostics 可供查阅，样本级独立重算需要这些 prediction 文件。

## 3. 测试数据计数

以下数字直接由各 split 的 `test.jsonl` 统计，并与相应 `statistics.json`、Joint summary 的计数交叉核对：

| Split | S+ | S− | U+ | U− | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A1 | 2218 | 1368 | 465 | 1119 | 5170 |
| A2_alt | 2798 | 1667 | 168 | 312 | 4945 |
| A3 | 2784 | 1723 | 192 | 594 | 5293 |
| C1 | 2799 | 1474 | 162 | 270 | 4705 |
| C2_alt | 2846 | 1490 | 115 | 254 | 4705 |

`EXPERIMENT_FREEZE.json` 中 A1 的 `test_s_neg_rows=1244`、`test_u_neg_rows=1243` 与 release 数据不一致；正确值分别为 **1368** 和 **1119**。总测试数 5170 仍正确。该 bookkeeping 错误不影响评测，因为 prediction 覆盖了 5170 个唯一 qid，summary 计数也与数据一致。

## 4. Existence AUROC：Moment-DETR-GMR 对比 Joint-v1

Seen AUROC 为 S+ vs S−，Unseen AUROC 为 U+ vs U−；Gap = Seen AUROC − Unseen AUROC。`ΔUnseen` = Joint Unseen − Baseline Unseen；`ΔGap` = Joint Gap − Baseline Gap。只有 `ΔUnseen > 0` 且 `ΔGap < 0` 才标记为该 split 的退化缓解。

Moment-DETR-GMR baseline 数值来自原始五 split 结果报告 `docs/reports/semantic_existence_multisplit_results.md` 及其 per-split diagnostics；Joint 数值来自本项目的样本级预测。两边均按当前 release 测试 qid 重算，Joint AUROC 重算值与 `joint_summary.json`、`diagnostics.json` 相符。

| Split | Baseline Seen | Baseline Unseen | Baseline Gap | Joint Seen | Joint Unseen | ΔUnseen | Joint Gap | ΔGap | 缓解退化 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| A1 | 0.804386 | 0.497269 | 0.307117 | 0.813078 | 0.462494 | −0.034775 | 0.350584 | +0.043467 | 否 |
| A2_alt | 0.769049 | 0.551082 | 0.217967 | 0.757806 | 0.506067 | −0.045015 | 0.251739 | +0.033772 | 否 |
| A3 | 0.748802 | 0.564319 | 0.184482 | 0.740958 | 0.671327 | +0.107008 | 0.069631 | −0.114852 | 是 |
| C1 | 0.760966 | 0.562071 | 0.198894 | 0.742917 | 0.602481 | +0.040409 | 0.140436 | −0.058458 | 是 |
| C2_alt | 0.675938 | 0.468658 | 0.207280 | 0.694495 | 0.450719 | −0.017939 | 0.243776 | +0.036495 | 否 |
| **五 split macro** | **0.751828** | **0.528680** | **0.223148** | **0.749851** | **0.538618** | **+0.009938** | **0.211233** | **−0.011915** | **split-dependent** |

结论是 **split-dependent improvement**：A3 和 C1 的 unseen AUROC 上升且 gap 缩小；A1、A2_alt、C2_alt 均退化。五 split macro unseen AUROC 上升 0.009938、macro gap 缩小 0.011915，但 macro 结果掩盖了三个 split 的退化。当前结果不支持无条件声称 “Joint-v1 alleviates unseen-semantic AUROC degradation”，也没有跨训练 seed 的不确定性估计。

样本级 prediction 中 `pred_exist_score` 保存为四位小数，ties 会影响 AUROC。A2_alt 的 unseen score 只有 10 个唯一值；使用保存的 logit 排序时 unseen AUROC 为 0.497281，而使用 score 为 0.506067。表中统一报告 benchmark 指定的 `pred_exist_score` AUROC；在高度量化的 split 上应谨慎解释小幅差异。

## 5. Accept / reject 行为

FRR 是 U+ 被拒绝的比例，RR 是 U− 被拒绝的比例。高 AUROC 或单独较好的阈值指标不能替代四象限检查。

| Split | U+ FRR | U− RR | Matched PairAcc | 观察 |
| :--- | ---: | ---: | ---: | :--- |
| A1 | 11.83% | 10.90% | 59.78% | 偏向接受 unseen absent |
| A2_alt | 0.00% | 0.00% | 51.90% | 全部 unseen 均被接受 |
| A3 | 6.77% | 15.82% | 47.29% | AUROC 提升，但 unseen absent 多数仍被接受 |
| C1 | 6.79% | 12.22% | 68.75% | AUROC 和配对排序较好，拒绝率仍低 |
| C2_alt | 95.65% | 97.64% | 81.82% | 几乎全部 unseen 均被拒绝 |

A2_alt 是 over-accept，C2_alt 是 over-reject。C2_alt 的高 PairAcc 不代表阈值决策成功：其 115 个 U+ 中 110 个被拒绝，97.64% 的 U− 被拒绝。A2_alt 两类 unseen 都被接受，也不是成功的 existence generalization。

## 6. Localization 辅助对比

下表按照要求比较 TRM localization-only summary 的 RAW 指标与 Joint summary 的 RAW 指标；没有拿 soft-gated 数字替代 RAW。单位为 R1@0.5 百分比。

| Split | TRM Seen RAW | TRM Unseen RAW | Joint Seen RAW | Joint Unseen RAW | Joint − TRM Unseen |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A1 | 43.87 | 23.01 | 40.22 | 25.59 | +2.58 |
| A2_alt | 36.38 | 25.60 | 38.46 | 27.38 | +1.79 |
| A3 | 36.85 | 51.56 | 35.31 | 51.56 | 0.00 |
| C1 | 37.01 | 47.53 | 36.55 | 53.70 | +6.17 |
| C2_alt | 36.12 | 31.30 | 38.23 | 36.52 | +5.22 |

C2_alt TRM 正确数值取自 `experiments/trm_momentdetr_generalization/generalization_summary.json`：Seen 36.120871、Unseen 31.304348、gap 4.816524。`scripts/aggregate_multisplit_joint.py` 中硬编码的 37.95 / 36.21 / 1.74 是错误值。另需注意，TRM summary 和 Joint RAW 采用的窗口后处理流程不完全一致，因此该表适合作为辅助比较，差异不能完全归因于模型结构。

Joint 的 official soft gate 对同一个 query 的所有候选乘以同一个非负 scalar，不改变候选排序。核验发现 soft-gated R1 与 RAW R1 的差异都可由 postprocessor 对 gated windows 做裁剪和取整解释；对 RAW windows 应用相同后处理即可复现 soft 指标。因此这些变化不代表 existence gate 改善了 localization ranking。

## 7. 汇总结论和记录更正

- Joint 五 split AUROC 和 diagnostics 数值正确；真实计数与 release `statistics.json`、Joint summaries 一致。
- Baseline AUROC 与原始 Moment-DETR-GMR per-split diagnostics 一致，原始报告中的 baseline predictions 覆盖同一组测试 qid。
- Joint README 旧版 C1 best epoch / threshold `14 / 0.9329` 应改为 `58 / 0.9964`；C2_alt 旧版 `10 / 0.5484` 应改为 `16 / 0.9225`。正确值见本 README 与各 split 的 threshold 文件。
- A2_alt holdout 是 `drink / pour`，A3 holdout 是 `run / walk`。
- 当前最明显的失败在 existence discrimination 和跨语义 score calibration：不同 split 分别出现接近随机的 AUROC、over-accept 或 over-reject。阈值只能改变决策点，不能修复 AUROC 排序；现有实验无法将失败因果归到某个单独模块。
