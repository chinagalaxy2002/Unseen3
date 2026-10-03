# 原始基线 AUROC 退化与 Joint-v3 审计

快照时间：2026-10-03T09:05:53.361923+00:00。本次只审计；未修改训练代码、冻结协议或后台任务。

## 结论

目前 A1/A2_alt 均完成 50 epochs，best epoch 分别为 10/36。A3、C1 仍在训练，C2_alt 排队。已完成两项尚未实现“提高 unseen AUROC，同时保留 Seen 与定位能力”的目标。不得据此宣称五 split 的最终结论。

## 原始 Moment-GMR 对照

AUROC 为 0–1；定位 R1 为百分比。以下使用原始基线 diagnostics 与 v3 summary，已核对双方预测 qid 集合相同。

| Split | 指标 | 原始基线 | V3 | 差值 |
|---|---|---:|---:|---:|
| A1 | seen_auroc | 0.804386 | 0.695052 | -0.109335 |
| A1 | unseen_auroc | 0.497269 | 0.511307 | +0.014038 |
| A1 | S+_raw_R1@0.5 | 36.970243 | 36.834986 | -0.135257 |
| A1 | U+_raw_R1@0.5 | 23.010753 | 21.720430 | -1.290323 |
| A1 | matched_pair_acc | 0.559295 | 0.625000 | +0.065705 |
| A2_alt | seen_auroc | 0.769049 | 0.730245 | -0.038804 |
| A2_alt | unseen_auroc | 0.551082 | 0.517666 | -0.033415 |
| A2_alt | S+_raw_R1@0.5 | 35.418156 | 29.949964 | -5.468192 |
| A2_alt | U+_raw_R1@0.5 | 26.785714 | 23.809524 | -2.976190 |
| A2_alt | matched_pair_acc | 0.620253 | 0.329114 | -0.291139 |

A1 的 Seen–Unseen gap 缩小主要来自 Seen 能力下降：Seen AUROC −10.93pp，而 unseen 仅 +1.40pp。A2_alt 的 gap 略缩小，但 Seen 与 unseen 同时下降。gap 不能单独作为成功标准。A2_alt PairAcc 从 0.6203 降至 0.3291。

**精度限制：**原始基线 prediction 仅存四位小数 probability，v3 使用完整精度 logit。上述是已发布指标对照，微小 AUROC 差值不能作为稳健提升证据。应对各 split 已有基线 best.ckpt 重新推理、保存完整 logit 后统一计分，无需重训；对 rounded probability 取 logit 无法恢复排序。单 seed 也不能证明统计显著性。

## 确认的问题与尚未证实的原因

### 1. 分组 sampler 排除了大量正例

`training/moment_detr_trm_gmr_joint_v3/semantic_groups.py` 仅采样同时具有正负例的组。composition 可采样组必然属于可采样 action，交替两种组仍无法覆盖 action 无负例的正例。这些数据采样概率严格为零。

| Split | 总样本 | 永不采样的 S+ | 占总训练集 | 占全部 S+ |
|---|---:|---:|---:|---:|
| A1 | 8608 | 2259 | 26.24% | 31.78% |
| A2_alt | 10323 | 3956 | 38.32% | 44.84% |
| A3 | 10352 | 3991 | 38.55% | 45.09% |
| C1 | 10516 | 1326 | 12.61% | 14.71% |
| C2_alt | 10612 | 1730 | 16.30% | 18.99% |

所有排除项都是正例。负例占比从 A1 17.43%、A2_alt 14.53% 改成每 batch 50%，同时降低每步定位正例数量并反复采样少数组。这是直接确认的数据分布改变；它对退化的因果贡献尚未通过受控实验量化。之前把分组采样直接用于整个训练数据流的实现需要纠正。

### 2. conditional AUC 与主指标并不等价

主指标是 split 内 pooled AUROC，包含所有语义组之间的正负比较：

`AUC = sum_(g,h) P(g|+) P(h|-) P(s_g+ > s_h-)`（ties 计半）。

组内目标只覆盖 g=h，不能约束跨组分数偏移。A1 U 正负 pair 约 50.01% 跨 action，A2_alt 约 60.10% 跨 action。当前 v3 A2_alt 组内 AUROC：drink 0.565859、pour 0.468435；两个组也未同时解决。此前将 conditional ranking 描述成与最终指标直接一致过于绝对。应保留 global ranking，把可靠配对作为辅助。

### 3. 定位门槛没有保护训练过程

A1 最高 Seen-val mAP 22.64，门槛 23.70；A2_alt 最高 22.38，门槛 25.49。两者整个训练均未达标，按用户要求保存并使用 mAP fallback。worst-semantic AUROC 选模没有在这两个最终模型上生效。门槛只是检查点筛选条件，不限制共享骨干受到 existence/ranking 梯度影响。

当前从头联合训练全部参数，未锚定原始基线。强排序项、噪声小组 worst loss、采样变化与共享梯度都可能参与退化，现有结果不能把原因唯一归给其中一个。不能根据 loss 数值大小就断言梯度主导。

### 4. “same-semantic” 大部分并非 same-query

训练统计中 exact-query pairs 仅约 17%，其余回退到 composition/action。一致 action 不等于一致文本。测试 Matched PairAcc 则使用 benchmark 的 matched U 对，不等于训练中任意同语义跨视频 pair；应对齐已发布训练负例的 source_qid 配对语义，并核查源正例、视频一致性及有效标签。U 配对仅用于评测。

## 建议下一版：冻结原始基线，学习受约束的 existence 修正

这是待实现方案，未启动新实验，也未调整正在运行的 v3。优先用最小变化隔离问题，不再同时重构定位、采样与多个强损失。

1. **起点使用各 split 原始 Moment-GMR best.ckpt。** 冻结骨干、投影、定位头、原 existence 头，固定 eval 模式和预处理。只训练独立小型 residual existence adapter：`s_new = s_base + epsilon * tanh(adapter(stopgrad(features)))`，末层零初始化。初始模型即基线，保留 epoch 0 为合法候选。epsilon 必须在 Seen-only 协议下预先确定，不能按 U 调整。
2. **恢复全量训练覆盖。** 主 loader 保留全部 S+/S− 的普通 shuffle，用于 BCE、global pairwise ranking 和基线约束。配对训练通过独立辅助采样实现；不能因为某组没有负例就剔除正例。记录每 epoch 全部 qid 的覆盖率、正例定位曝光量、重复率。
3. **简化 objective。** 以 BCE + global pairwise AUC surrogate 为主；追加低权重、已确认标签的 same-query 跨视频配对及 source-positive/negative 同视频配对；加入 Seen 上对原始 baseline 的输出一致性正则。先不使用 noisy worst-group max，也不重复叠加强 conditional 和 matched 项。权重需在新实验冻结文件里一次固定；当前 U 结果不能用于 sweep。输出约束会限制修正幅度，但不能数学保证 unseen 或 Seen test AUROC 不下降。
4. **选模保护实际目标。** Seen-val pooled AUROC 为主，语义分组指标为诊断；要求 Seen-val AUROC 相对基线不超过预先声明的小幅容差，raw localization 必须与冻结基线一致。无可接受 checkpoint 时回退 epoch 0 基线，同时保留训练中最佳 adapter 检查点并明确其未达标，不能把未达标模型默认为成功方法。继续遵守每实验训练 50 轮。
5. **验证边界。** 冻结全部 baseline 参数、eval 模式及相同推理路径可保持 raw spans/raw localization scores；official existence-gated 指标仍可能变化。Seen-val 约束不保证 Seen-test 保持，更不保证 U 提升。若做 Seen 家族模拟 held-out，必须连初始化 teacher 的训练都排除该家族，不能把 teacher 见过的家族称为 pseudo-unseen。

## 执行优先级与报告标准

- 当前固定 50 轮 v3 保留冻结协议完成，作为探索性失败模式证据；不根据 A1/A2 U 修改剩余 split。
- 下一次实验前先重算原始 checkpoint 的全精度结果，统一 qid、raw localization 后处理和 AUROC score。
- 完成采样覆盖修复、冻结基线一致性和 paired-label 审计后，才启动独立新目录的 adapter 实验；不覆盖 v1/v2/v3。
- 成功标准同时报告原始基线的 unseen AUROC delta、Seen AUROC delta、raw localization delta、PairAcc、各 split improved/degraded count。不要用 gap 缩小代替 unseen 提升。
- 保持 exploratory 声明：设计已受到这些 U 结果启发；未来一轮 U 仍不能参与训练、超参数选择、阈值或选模。

## 证据路径

- `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen/BASELINE_AUROC_DEGRADATION.md`
- `experiments/trm_gmr_joint_v1/HANDOFF.md`
- `experiments/trm_gmr_joint_v3/*_semantic_manifest.json`
- `results/moment_detr_trm_gmr_joint_v3/{A1,A2_alt}/joint_v3_summary.json`
- `results/moment_detr_trm_gmr_joint_v3/{A1,A2_alt}/training_meta.json`
- `training/moment_detr_trm_gmr_joint_v3/semantic_groups.py`
- `models/moment_detr_trm_gmr_joint_v3/robust_auc.py`
- 原始 baseline 根目录：`/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2`

研究背景：[GroupDRO 官方代码与论文说明](https://github.com/kohpangwei/group_DRO)明确讨论了 naïve group DRO 的泛化与正则化问题。这支持审慎使用 worst-group 优化；不构成本实验退化的因果证据。
