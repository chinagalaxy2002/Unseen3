# Idea10：SHINE → GMR，absence-aware transfer

2026-10-04 实验归档。正式目标是缓解 Seen→Unseen 退化，当前聚焦 A1/C1、seed3407。Moment、QD-DETR、FlashVTG 的50epoch比较，以及Moment分项消融和局部证据侧线均完成；原五split未完成部分退出本轮范围。

- **[当天实验与结果统一入口](EXPERIMENTS_README.md)**：配置、三模型最终结果、分项消融、局部证据侧线、结论与后续。
- **[全部baseline对照表](records/final_assessment/FULL_BASELINE_COMPARISON.md)**：完整绝对分数、增量和Gap；预算及历史baseline分开。
- [综合评估](records/final_assessment/RESULTS.md) / [结构化结果](records/final_assessment/RESULTS.json)。
- [QD-DETR](qd_detr_transfer/README.md) / [FlashVTG](flash_vtg_transfer/README.md) / [直接局部证据](local_evidence_transfer/README.md)。
- [交接](records/formal/WORK_HANDOFF.md) / [未来任务](records/formal/NEXT_ACTIONS.md) / [初始失败分析](records/formal/A1_C1_FAILURE_ANALYSIS.md)。

当前结论：QD coarse/fine有小幅正向趋势，Moment纯coarse/fine不一致，Flash C1明显负向；低BCE=.01无pair是待验证候选；直接saliency局部证据分支未观察到额外收益。不能宣称稳定、跨模型的退化缓解。

## 来源、代码与协议

上游 [zxccade/SHINE](https://github.com/zxccade/SHINE) 固定commit `dfbaab1cf8f6f88ca3c3a39bee2bbe8dd999f466`，MIT LICENSE，保留于vendor/shine。迁移保留各GMR架构；不是完整SHINE-native模型复现。Moment本地模型/训练复制位于code，其他迁移位于各独立子目录。代码与数据选择均有冻结hash，原共享环境未修改。

用户确认batch轮换query为有效absent、编辑层级距离链有效。只用Seen-val选checkpoint/阈值，正式U不用于epoch/权重选择。A1/C1已探索，结果不属于untouched确认；所有CI为固定单seed模型的配对视频bootstrap。

## 重现与归档

各阶段configs保存方案与冻结；code保存训练、评测及汇总入口；runs中公开训练配置/曲线/指标/状态，records保存正式预测与报告。模型权重、原始数据/特征和二进制依赖未上传；原路径与hash保留。这不是自包含数据包。环境见[ENVIRONMENT](records/ENVIRONMENT.md)。旧README和阶段进展保留于[历史快照](records/history/README_BEFORE_GITHUB_SYNC_2026_10_04.md)。
