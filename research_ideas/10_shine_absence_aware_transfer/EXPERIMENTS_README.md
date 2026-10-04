# 2026-10-04：Idea10 实验与结果汇总

本页整理今天已完成的训练、评测和诊断。当前 A1/C1 范围的 Moment-DETR、QD-DETR、FlashVTG 50-epoch 对照，以及 Moment 分项消融和局部证据侧线均已完成。原五 split 的未完成部分已按用户要求退出当前任务范围。

## 实验共同设置

- 仅 seed3407；A1/C1 正式 split；各模型从对应 canonical checkpoint 初始化。
- 只用 Seen-val AUROC 选择最早最佳训练 epoch，阈值也只用 Seen-val；epoch0为诊断参照。
- 主比较为同预算 baseline（B0）与方法；匹配自然/轮换/编辑 forward 和采样，核验 exposure hashes。
- 50-epoch 对照使用 lr1e-5、batch16、原 weight decay、grad clip .1。
- Seen/U 为存在性 AUROC。Gap=Seen−Unseen；Gap缓解量=B0 Gap−方法Gap。增量单位pp，正Gap缓解量表示缩小。
- CF=coarse+fine，各权重1；B=额外 rotated BCE；P=existence pair。下表的B/P数字均为有效权重。

## 三模型50epoch：只加入 coarse/fine

| 模型 | Baseline Seen / U | CF Seen / U | ΔSeen(pp) | ΔU(pp) | Gap缓解(pp) | 判断 |
|---|---:|---:|---:|---:|---:|---|
| Moment | 0.7822 / 0.5326 | 0.7868 / 0.5353 | +0.46 | +0.27 | -0.19 | C1改善，A1下降；平均Gap扩大 |
| QD-DETR | 0.7904 / 0.5362 | 0.7934 / 0.5427 | +0.30 | +0.65 | +0.36 | 两fold正向点估计；U区间均跨0 |
| FlashVTG | 0.7973 / 0.5593 | 0.8032 / 0.5348 | +0.59 | -2.45 | -3.04 | 负向；C1 U区间完全低于0 |

各模型分项结果：

- [Moment 50-epoch 最终报告](records/saliency_v5/evaluation/RESULTS.md)
- [QD-DETR 50-epoch 最终报告](qd_detr_transfer/records/evaluation/RESULTS.md)
- [FlashVTG 50-epoch 最终报告](flash_vtg_transfer/records/evaluation/RESULTS.md)

QD此前的[阶段评测](qd_detr_transfer/records/interim_eval_v1/RESULTS.md)是历史snapshot；当前以最终50epoch报告为准。QD S1两fold最佳Seen-val均在epoch1，最终分数与阶段snapshot一致，不代表后续epoch提高了效果。

## Moment 10epoch：分项定位与低权重修复

v2拆开saliency与existence组合；v3拆开BCE/pair并测试BCE=.1；v4分别降低到BCE=.01和pair=.02。所有新checkpoint由Seen-val选择，最终正式U评测独立冻结。

| 阶段 | 实验 | 所检验的问题 | 结果入口 |
|---|---|---|---|
| v2 | B0、CF、B1+P.2、CF+B1+P.2 | 早期Seen损伤来自哪里 | [A1](records/ablation_v2/A1_RESULTS.md) / [C1](records/ablation_v2/C1_RESULTS.md) |
| v3 | CF+B1、CF+P.2、CF+B.1+P.2 | BCE/pair及其强度 | [Seen开发](records/ablation_v3/README.md) / [正式U](records/ablation_test_v2_v3/RESULTS.md) |
| v4 | CF+B.01、CF+P.02、CF+B.01+P.02 | 进一步减弱监督能否保住主任务 | [Seen开发](records/ablation_v4/README.md) / [正式U](records/ablation_test_v4/RESULTS.md) |

低权重方案的两fold平均正式结果（对照为匹配10epoch B0，不能与50epoch混作同预算增量）：

| 方法 | Seen | U | ΔSeen(pp) | ΔU(pp) | Gap缓解(pp) |
|---|---:|---:|---:|---:|---:|
| B0 | 0.7830 | 0.5313 | +0.00 | +0.00 | +0.00 |
| S1_saliency_only | 0.7856 | 0.5350 | +0.26 | +0.38 | +0.12 |
| Low_BCE_only | 0.7826 | 0.5387 | -0.04 | +0.75 | +0.78 |
| Low_Pair_only | 0.7815 | 0.5370 | -0.15 | +0.58 | +0.72 |
| Low_BCE_pair | 0.7801 | 0.5390 | -0.29 | +0.77 | +1.06 |

CF+BCE=.01、无pair是后续候选，但A1 Seen下降.42pp，C1 Seen上升抵消了平均值；两foldU区间均跨0。原高权重方案在Seen上的明显损伤，不能用较大Gap缩小掩盖。

完整逐fold/全部消融/历史完整50epoch方案见[全部baseline对照表](records/final_assessment/FULL_BASELINE_COMPARISON.md)。

## 侧线：saliency 局部证据直接用于 existence

本实验将query-fused视频token按saliency或uniform权重池化，经零初始化残差分支加入原existence logit；通过Local_CF/Uniform_CF控制saliency选择和额外容量，Local_noCF控制辅助监督。Moment、10epochs、同seed/同起点。

| 方法 | Macro Seen | Macro U | ΔSeen vs B0(pp) | ΔU vs B0(pp) | Gap缓解 vs B0(pp) |
|---|---:|---:|---:|---:|---:|
| B0 | 0.7830 | 0.5313 | +0.00 | +0.00 | +0.00 |
| S1_saliency_only | 0.7856 | 0.5350 | +0.26 | +0.38 | +0.12 |
| Local_CF | 0.7858 | 0.5348 | +0.28 | +0.36 | +0.07 |
| Uniform_CF | 0.7859 | 0.5349 | +0.29 | +0.36 | +0.07 |
| Local_noCF | 0.7830 | 0.5311 | +0.00 | -0.02 | -0.02 |

Local_CF与Uniform_CF接近，没有观察到saliency加权分支的额外收益；Local_CF的平均Gap缓解.07pp弱于原CF的.12pp。[侧线完整评测](local_evidence_transfer/records/evaluation_v1/RESULTS.md) / [退化报告](local_evidence_transfer/records/evaluation_v1/DEGRADATION_REPORT.md)。

## Pilot 与机制诊断

- [Inner pilot](records/PILOT_RESULTS.md)：两fold Novel-dev平均+3.88pp；这不是正式Unseen收益。
- [原完整50epoch方案](records/formal/A1_C1_FORMAL_RESULTS.md)：Moment macro U−.67pp、Seen−4.06pp，失败。
- [失败分析](records/formal/A1_C1_FAILURE_ANALYSIS.md)：训练曲线、初始梯度冲突、负例组成、score压低、条件排序、时间格检查。
- 正式评测包含视频聚类配对区间、source-pair/同query排序、定位和fresh shuffled-video诊断；相关JSON保留完整精度。

## 当前判断与后续

尚不能声称SHINE迁移是跨模型的稳定退化缓解方案。QD有小幅正向趋势；Moment纯CF不一致；Flash C1明确负向；低权重BCE是待验证候选；局部证据分支未带来额外收益。所有A1/C1 test已探索，paired bootstrap区间只反映固定seed模型的采样不确定性，不包含重训变异。

建议后续：匹配50epoch检验CF+BCE=.01无pair，并检查A1 Seen风险；在QD拆分coarse-only/fine-only，补开发集条件控制；先诊断Flash C1，不继续盲目扩展。以上尚未启动，不做多seed、不自动扩展五split。

## GitHub归档与重现边界

归档代码、上游MIT来源、实验计划/冻结hash、训练配置/状态/曲线、最终指标、逐样本预测、诊断与报告。模型权重、数据/视频文本特征、二进制依赖和缓存保留本地；冻结文件保存它们的来源路径和校验hash。这是实验与结果归档，下载后仍需原环境和特征/权重才能重现。

发布文件清单见 [PUBLICATION_MANIFEST](records/github_sync_2026_10_04/PUBLICATION_MANIFEST.json)。冻结JSON中的绝对路径对应原工作环境；历史记录中的running/queued是当时快照，当前状态以本页最终汇总为准。
