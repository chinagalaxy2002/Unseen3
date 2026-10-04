# Semantic-novelty GMR：下一阶段研究目录

状态：各方向包含设计、实施和实验结果，详见各目录 records。更新时间：2026-10-04。下文初始 01–08 方案保留文档阶段描述，之后的实施更新及方向 09/10 记录当前状态。

研究依据是用户整理的 [semantic_novelty_GMR_next_stage_research.md](../semantic_novelty_GMR_next_stage_research.md)，该文件保留原样。证据锚点为 main commit [`6cd96d723806e4b2b2474c68f289921eb543ffcc`](https://github.com/chinagalaxy2002/Unseen3/tree/6cd96d723806e4b2b2474c68f289921eb543ffcc)。目录编号表示建议的研究优先级，不是已启动的实验队列，也不意味着需要依次完成所有方向。

## 研究目标

解释并缓解三个 GMR backbone 在 semantic novelty 下共同出现的 Seen→Unseen existence AUROC 退化。核心任务为 `Y(V,Q)=1[视频中存在满足 query 的 moment]`，核心指标为 `AUROC(U+ vs U-)`。U- 视频可以包含其他事件。不能把“视频有事件”“定位更准”或“query 更像正例”直接当作目标。

三条优先排查的解释是：正负 query 的语言边际相关性；监督不足以识别条件兼容性；单个 existence score 混合了跨 query 的语义偏置和视觉证据。这些是待验证假设。V5 Cq 在四个 inner folds 的 pooled Novel AUROC 都提高，而 source-pair PairAcc 都降低，使区分语言先验与视频条件证据成为当前首要任务。

## 初始八个独立方向及优先级

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


## 方向 09：先验受控的证据增量审计（PEI-0）

[方向 09](09_prior_controlled_evidence_increment/README.md) 已实现 Stage 0 标签/特征审计、Stage 1 零训练相似度与双中心化、Stage 2 五 seed 视频分组交叉拟合 Cq、固定先验 offset 的 M2/M3、shuffle-refit null、共享视频 bootstrap 和条件功效模拟。13 项合成输入检查通过，尚未运行真实 PEI-0 实验。完整流程必须先通过有时间提取来源支持的 Stage 0 alignment gate；正式 U 不参与。冻结代码及判据见 [FREEZE](09_prior_controlled_evidence_increment/records/FREEZE.json)，验证记录见 [IMPLEMENTATION_VALIDATION](09_prior_controlled_evidence_increment/records/IMPLEMENTATION_VALIDATION.json)。


## 方向 10：SHINE → GMR 单 seed 迁移

[Idea 10](10_shine_absence_aware_transfer/README.md) 以用户提供的 [zxccade/SHINE](https://github.com/zxccade/SHINE) 为源码依据，已实现 coarse/fine saliency ranking 与最终 existence 监督的独立 Moment-DETR-GMR 迁移。按用户确认采用 batch 轮换 absent negative 和强制三层距离链；仅 seed 3407，先 A1/C1 各一折比较同起点十轮微调，优先检查 AUROC 与定位效果。已完成两个 folds × 两个 arms 的各十轮训练：Novel-dev AUROC A1 0.6122→0.6541（+4.18 pp）、C1 0.5455→0.5813（+3.58 pp），macro +3.88 pp；Seen macro +1.32 pp。该 pilot 未使用正式 U，未进行多 seed。定位和条件排序尚未一致改善，当前为 pooled 增益候选；[完整结果](10_shine_absence_aware_transfer/records/PILOT_RESULTS.md)包含配对区间和声明边界。

Idea 10 的用户目标已明确为 COMMON_PROTOCOL 的正式 Seen→Unseen 退化缓解。已冻结五 split 的 canonical / 同起点 baseline50 / SHINE50 比较，仍只用 seed 3407；A1/C1 两 arms 均已完成 50 epochs，按 Seen-val 选模冻结后已完成正式 U 评测；用户随后取消五 split 收尾优先级，当前聚焦 A1/C1 诊断与修复。[正式方案](10_shine_absence_aware_transfer/FORMAL_EXPERIMENT_PLAN.md)。此前两折 Novel-dev 增益不得作为正式 U 结论。

Idea 10 的 [A1/C1 正式评测](10_shine_absence_aware_transfer/records/formal/A1_C1_FORMAL_RESULTS.md) 未观察到退化缓解：SHINE 相对 baseline50 的 Unseen AUROC 分别 0.4858→0.4799、0.5747→0.5672，Seen 分别 0.8088→0.7517、0.7606→0.7365。两 split macro ΔUnseen −0.67 pp、ΔSeen −4.06 pp；Gap 缩小由 Seen 下降驱动。仅为两 split、单 seed 结果，不代表五 split 汇总。

Idea 10 清空上下文后的执行入口：[交接文档](10_shine_absence_aware_transfer/records/formal/WORK_HANDOFF.md)、[后续任务及恢复提示](10_shine_absence_aware_transfer/records/formal/NEXT_ACTIONS.md)、[诊断分析](10_shine_absence_aware_transfer/records/formal/A1_C1_FAILURE_ANALYSIS.md)。这些文档区分已完成结果、实时训练快照和待验证消融；v2/v3 消融已完成，最新阶段为 v4 修复开发；先读交接顶部更新和实时状态。

Idea 10 后续已完成 A1/C1 的 v2/v3 分项消融与[10-epoch 正式探索性评测](10_shine_absence_aware_transfer/records/ablation_test_v2_v3/RESULTS.md)。saliency-only 相对匹配 B0 的 macro ΔUnseen +.38 pp、ΔSeen +.26 pp，但 A1 −.40 pp/C1 +1.16 pp，尚无跨两折一致改善；新增 BCE/pair 的小幅 U 增量伴随 Seen 损伤。当前[低强度 v4 修复实验](10_shine_absence_aware_transfer/records/ablation_v4/README.md)已启动，继续单 seed、两卡各三任务、Seen-only 开发；权重在新 U 评测前固定。

Idea 10 最新进展：低权重v4已完成，用户要求扩大coarse/fine-only训练预算，已启动[匹配50epoch对照](10_shine_absence_aware_transfer/records/saliency_v5/README.md)，A1/C1各B0/S1两任务并行、seed3407，结束后自动评测。10epoch的Seen−Unseen Gap平均只缩小.12pp（相对.47%），其中A1扩大.40pp、C1缩小.64pp；暂不能称稳定退化缓解。

Idea 10 最新综合评估：A1/C1三模型50epoch与v2-v4均完成，详见[最终评估](10_shine_absence_aware_transfer/records/final_assessment/RESULTS.md)。coarse/fine-only相对同预算B0：Moment/QD/Flash macro ΔU +.27/+.65/−2.45pp，Gap缩小−.19/+.36/−3.04pp；不能认定跨模型普适缓解。Moment低BCE=.01无pair（10ep）U+.75pp、Seen−.04pp、Gap缩小.78pp，但A1 Seen−.42pp且两折U区间跨0，为待验证候选。当前没有新训练排程。

Idea 10 已整理[2026-10-04统一实验README](10_shine_absence_aware_transfer/EXPERIMENTS_README.md)，包含三模型50epoch、Moment全部分项消融、低权重修复和局部证据侧线。完整baseline表包含逐fold/两foldmacro，训练与正式评测均有独立记录。
