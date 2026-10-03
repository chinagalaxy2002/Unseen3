# Moment-DETR-GMR Evidence V5

状态：四个随机初始化内层 baseline 均完成 100 epochs；P3 decoder 读取对照已完成 36/36 runs，每 run 50 epochs，两张 GPU 队列均成功结束。局部 ROI 分支仍受时间映射 gate 约束。属于受 V1–V4 结果启发的 exploratory follow-up。

- [完整工作方案](WORK_PLAN.md)：目标、阶段依赖、数据契约、内层协议、结构/监督对照、预算、判断分支和验收标准。
- [实施清单](IMPLEMENTATION_CHECKLIST.md)：按依赖逐项完成并关联实际产物。
- [设计依据](../auroc_degradation_audit/POST_V4_ADJUSTMENT_PLAN.md)：已确认项目问题与结论边界。
- [本轮运行记录](RUN_STATUS.md)：已完成准备、实际任务和后续依赖。
- [P3 执行记录](P3_RUN_STATUS.md)：baseline 完成审计、读取对照、验证和阶段边界。
- [P3 完整结果](P3_READOUT_RESULTS.md)：36 runs 的逐 fold 指标与 macro 汇总。

工作顺序：统一评测 → Seen source-pair/缓存 → 内层语义留出 → 读取对照 → 反事实监督 → 冻结五 split → 多 seed 与报告。

工作方案中的模块按阶段实现；具体已实现入口以相应执行记录为准。

查看当前进度：`python scripts/status_evidence_v5.py`。本轮队列已完成；R3 macro ΔNovel 为 +1.18 pp，但仅 2/4 folds 为正，未达到本轮推进条件。当前仍为 seed-3407 开发结果，三 seed G2 尚未通过。
