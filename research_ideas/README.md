# Semantic-novelty GMR：下一阶段研究目录

状态：`draft_not_executed`。更新时间：2026-10-04。本次交付仅撰写 Markdown，没有实施数据构造、模型修改、训练或推理，也没有提交或 push。

研究依据是用户整理的 [semantic_novelty_GMR_next_stage_research.md](../semantic_novelty_GMR_next_stage_research.md)，该文件保留原样。证据锚点为 main commit [`6cd96d723806e4b2b2474c68f289921eb543ffcc`](https://github.com/chinagalaxy2002/Unseen3/tree/6cd96d723806e4b2b2474c68f289921eb543ffcc)。目录编号表示建议的研究优先级，不是已启动的实验队列，也不意味着需要依次完成所有方向。

## 研究目标

解释并缓解三个 GMR backbone 在 semantic novelty 下共同出现的 Seen→Unseen existence AUROC 退化。核心任务为 `Y(V,Q)=1[视频中存在满足 query 的 moment]`，核心指标为 `AUROC(U+ vs U-)`。U- 视频可以包含其他事件。不能把“视频有事件”“定位更准”或“query 更像正例”直接当作目标。

三条优先排查的解释是：正负 query 的语言边际相关性；监督不足以识别条件兼容性；单个 existence score 混合了跨 query 的语义偏置和视觉证据。这些是待验证假设。V5 Cq 在四个 inner folds 的 pooled Novel AUROC 都提高，而 source-pair PairAcc 都降低，使区分语言先验与视频条件证据成为当前首要任务。

## 八个独立方向及优先级

| 优先级 / 原总结编号 | 目录与完整 idea | 定位 | 最小实验要回答什么 | 依赖与成本 |
|---|---|---|---|---|
| 01 / Idea 1 | [Crossed existence supervision](01_crossed_existence_supervision/IDEA.md) | A：最稳妥 | 同时固定 query 和 video 的监督，能否比 BCE / 单向配对更好迁移？ | 已标注四格支持；低至中 |
| 02 / Idea 2 | [Reference-centered interaction](02_reference_centered_interaction/IDEA.md) | B：研究价值优先 | 去除可加单模态分量并在该分数上训练，能否改善条件排序及 pooled AUROC？ | Seen-only reference bank；低至中 |
| 03 / Idea 5 | [Complete-event witness](03_complete_event_witness/IDEA.md) | C：风险更高、潜在贡献更大 | 共现的局部匹配能否替代完整事件绑定？如何让存在性由一致的事件证据支撑？ | 先用序列索引；精确时间监督另有 gate；中至高 |
| 04 / Idea 3 | [Query marginal matching](04_query_marginal_matching/IDEA.md) | 数据与监督贡献 | 同一自然 query 在不同视频上都有可靠正负标签后，语言边际预测性是否下降？ | 与 01 共用标签核验；中 |
| 05 / Idea 8 | [Independent compatibility adapter](05_independent_compatibility_adapter/IDEA.md) | 表示替代路线 | 冻结原始预计算特征，独立训练兼容性，能否避免 localization-conditioned 表示限制？ | 不依赖秒级 ROI；低至中 |
| 06 / Idea 6 | [Evidence-aware MIL aggregation](06_evidence_aware_mil_aggregation/IDEA.md) | 候选证据路线 | 监督局部证据及控制重复候选，是否比单纯增加 readout 容量有效？ | 候选及时间 provenance；中 |
| 07 / Idea 4 | [Compatibility domain generalization](07_compatibility_domain_generalization/IDEA.md) | 语义迁移路线 | source 语义环境变化时，关系损失而非语义分类表征能否稳定泛化？ | 先确认条件监督有信号；中至高 |
| 08 / Idea 7 | [Evidence intervention](08_evidence_intervention/IDEA.md) | 因果证据验证与方法扩展 | 移除/保留/替换目标证据后，分数变化是否符合 query 条件的预期？ | 时间对齐及扰动分布 gate；中至高 |

每个目录均含：`IDEA.md`（假设、机制、贡献与风险）、`EXPERIMENT_PLAN.md`（最小验证、对照、判据和失败解释）、`RELATED_WORK.md`（最近邻原文对照与 novelty argument）。共同约束见 [COMMON_PROTOCOL.md](COMMON_PROTOCOL.md)，研究分流见 [PRIORITY_AND_DECISIONS.md](PRIORITY_AND_DECISIONS.md)。

## 如何使用这组文档

将 01 的数据可识别性检查作为第一项未来工作，再决定先跑监督方向还是表示方向。01 与 04 共用经过核验的配对信息，但研究问题分别是监督结构和数据边际；02 可以独立验证打分机制；05 是 pooled 表示无效时的合理替代。03 不因风险高而取消，它应在可以区分完整证据与局部共现时推进。07、08 的启动条件分别是可靠关系损失和可靠证据干预。

所有数值收益门槛及超参数均是实验设计建议，尚未注册或执行。历史 V1–V5 结果与未来结果必须分栏，不能将本文的方案写成已验证方法。新方法若只提升 pooled AUROC，应报告其实际作用，不能直接宣称学习了视频条件存在性。新方法若只提升条件排序，也不能宣称已经解决主 benchmark 的 AUROC 退化。

## 单目录恢复与独立实验

八份 `IDEA.md` 的第 0 节均完整包含研究目标、三 baseline 路径与文件职责、四象限数据与特征/权重位置、原始基线与 V1–V5 的区别、数据和实验记录框架，以及该 idea 专属的代码复制/改动清单。它们可以随单个 idea 目录复制，理解背景不依赖本 README 或先前聊天。

原始三模型来源是 `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval`；Unseen3 有 Moment/Flash 副本，QD 代码需从原项目取得。未来每个 idea 内物理复制源码、隔离 imports 和全部输出，大特征/旧 checkpoint 只作可追踪只读输入，不修改其他模型、共享 release 或其他 idea。`vendor/code/data/configs/artifacts/runs/records` 是未来布局，本轮没有创建可执行实验或启动训练。

## 贡献判断原则

相关思想已有先例，并不使方向失去价值。先落实在本项目中新增的监督信息、分数约束、证据绑定或评价目标，再与最近邻比较。判断采用 A（基本已有）、B（已有核心思想且有实质扩展）、C（新组合解决未处理问题）、D（有明确新增机制）；当前文档通常给出 B/C 的候选定位。D 必须由完整数学定义、严格消融及相对最近邻的证据支撑。不存在“尚未搜到相同标题，所以绝对原创”的结论。


## 实验实施更新（2026-10-04）

上文“draft_not_executed”及计划性语句记录的是授权前文档阶段。之后按用户最新授权，在 `_orchestration/` 建立了可恢复任务队列并完成优先级 01–08 的最小筛查或必要 gate。方向最终 disposition、逐 fold 结果、隔离边界和失败原因以各目录 `records/DIRECTION_DISPOSITION.md` 为准；集成解释见 [`_orchestration/RESULT_SYNTHESIS.md`](_orchestration/RESULT_SYNTHESIS.md)。所有方向均未打开正式 U；当前没有满足证据要求的候选方法，未运行正式确认评测。

Idea 05 随后完成了预先冻结的 Seen-only query-prior score-correction 跟进，没有重训或访问正式 U。A1 校正后 Novel-dev pooled AUROC 下降 3.83 pp，且 same-video source-pair PairAcc 下降；C1 的 Seen-selected 系数为 0。方向 disposition 仍为 `insufficient_evidence`，详细 fold 结果见 [Idea 05 记录](05_independent_compatibility_adapter/records/DIRECTION_DISPOSITION.md)。当前队列没有运行任务，GPU worker/controller 均已退出。
