# Moment-DETR-GMR Evidence V5

状态：内层 baseline 实验已于 2026-10-03 启动，两张 GPU 各运行一个持久队列，共四个任务，每任务从随机初始化训练 100 epochs。verifier 对照尚未启动。属于受 V1–V4 结果启发的 exploratory follow-up。

- [完整工作方案](WORK_PLAN.md)：目标、阶段依赖、数据契约、内层协议、结构/监督对照、预算、判断分支和验收标准。
- [实施清单](IMPLEMENTATION_CHECKLIST.md)：按依赖逐项完成并关联实际产物。
- [设计依据](../auroc_degradation_audit/POST_V4_ADJUSTMENT_PLAN.md)：已确认项目问题与结论边界。
- [本轮运行记录](RUN_STATUS.md)：已完成准备、实际任务和后续依赖。

工作顺序：统一评测 → Seen source-pair/缓存 → 内层语义留出 → 读取对照 → 反事实监督 → 冻结五 split → 多 seed 与报告。

本目录的计划中出现的新增 Python 模块和运行入口均为待实现项；不能将它们视为已有可执行命令。

已经实现的入口以运行记录为准。查看当前进度：`python scripts/status_evidence_v5.py`。进程在会话结束后继续运行，队列在当前 fold 完成后自动开始同设备下一 fold；失败时停止该队列并保存失败状态。
