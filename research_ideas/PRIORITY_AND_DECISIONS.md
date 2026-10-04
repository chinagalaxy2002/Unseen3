# 优先级、分流与保留决策

状态：`draft_not_executed`。本文件用于未来实施阶段的决策，当前只交付文档。

## 如果只做一个新实验

选择 **01：同一冻结表示、相同训练覆盖下的 BCE / same-video ranking / 闭合四格监督对照**。先核验已有标签能闭合多少四格，再在现有四个 Seen-only inner folds 上训练小 head。它直接问：缺少控制 query 与 video 的条件监督，是否足以解释 source 判别不能迁移？它也能说明 V5 Cq 的 pooled 收益为何与 source PairAcc 冲突。首轮不联训 backbone，也不默认采用 full-slot architecture。

若已有四格独立支持不足，优先跑 exact-query column pairs，并把人工核验后的数据扩充归入 04；不要用随机 cross-video negatives 拼成假四格。数据审计本身不是方法收益证据。

## 研究分流

| 可识别观察（未来） | 下一项优先工作 | 理由 |
|---|---|---|
| 01 同时提升 pooled 与条件指标 | 复核种子、最近邻对照，再扩到其他 backbone | 最短路线形成监督方法贡献 |
| 01 提升条件指标，但 pooled 不升 | 02 分数分解 | 检查跨 query 的 offset/scale 是否仍阻碍主指标 |
| 01 无条件收益，train 条件损失也学不动 | 05 原始特征 adapter；检查标签支持 | 冻结 pooled state 可能缺信息，不能急于加更多 loss |
| 01 train 学会但 Novel 条件排序退化 | 07 关系层 domain generalization，或 03 完整事件绑定 | 区分 source relation 过拟合与部分语义共现 |
| Cq 在原分布强、自然 cross-video negatives 后弱 | 04 query marginal matching | 数据构造有可解释改进空间 |
| Candidate-level evidence 有信号，聚合易受重复/局部高分干扰 | 06 evidence-aware aggregation | 回到局部监督和候选依赖，不重复单纯扩大 readout |
| 精确 ROI / GT-clip provenance 已成立且 witness 有效 | 08 evidence intervention | 将证据一致性转成必要性/充分性的受控训练 |

此表允许方向并行探索，排序反映当前信息价值和预算，而不是先验认定低优先级方向无效。03 是最终三个推荐方向之一，但其最小无 ROI pilot 可以早于完整干预版本；完整训练应在最关键混杂有对照后推进。

## 不再原样重复的方向

- 单纯提高 localization / phrase correspondence 后期待 existence 自然提升。
- 用 generic eventness 或 video-only residual 替代 query-event compatibility。
- 重写主 sampler 为 semantic-group balanced pairing，并牺牲自然 S+ 覆盖。
- 再次只在现有 pooled state 上做相似 post-hoc residual correction。
- 把更大 MLP、更多 slots 或 normalized logsumexp 本身当作 pooling bottleneck 已成立的解决方案。
- 利用 text prior 的 pooled AUROC 收益作为视频理解证明。

这里放弃的是已有缺乏稳定支持的具体配置。Phrase structure、local evidence、paired supervision、分数分解均可以在新增机制和有效控制下保留。

## V1–V5 值得保留的资产

| 资产 / 发现 | 保留方式 |
|---|---|
| Canonical baseline 与固定版本 replay | 保留可比较 score 和 localization，对新分支建立明确 fallback |
| V5 四个严格 inner folds、feature banks、容量对照 | 作为小规模开发基础；复核 encoder 是否暴露相应 holdout |
| Cq / Cv 及 pooled/paired 分歧 | 固定为每个新方向的机制对照 |
| Source-pair provenance 与 sampler coverage 审计 | 复用配对支持，保持全部自然主训练行 |
| Phrase / candidate 表示 | 仅作为 03、06 可选输入，不强制继承 V1/V5 architecture |
| R4 time-grid gate | 没有精确抽取时间证据前，不宣称秒级 ROI 特征对齐 |
| Bootstrap、split 与定位保持的纪律 | 统一复用，同时修正 quartet / 多 split 的非独立性 |

最有价值的继承是可证伪的评价协议与失败约束，而不是版本名称。目录采用机制名称，不称 V6。


## 已执行研究阶段追加记录（2026-10-04）

历史段落描述的是原计划。按本次用户授权，方向 01–06 已在隔离目录内完成了可执行的最小筛查。方向 01、02、04、05、06 的具体结果与 `insufficient_evidence` disposition 见各自 `records/DIRECTION_DISPOSITION.md`；方向 03 的特定 binding run 因严格支持不足而 `blocked`。方向 07 先审计 inner-train 可信 pair environments：只有存在多个标签齐全、视频支持足够且不将标签本身编码为环境的 unit strata，才冻结 D0/D1/D2/D3 小筛查。方向 06 没有可用正 candidate-slot 标签或时间映射，因此不扩展 GT witness 监督。

方向 08 的 time/label gate 已在 2026-10-04 实际审计并因必要时间映射与变换后标签缺失而 blocked；详情见 `08_evidence_intervention/records/DIRECTION_DISPOSITION.md`。现有非时间配对替代不作为证据干预。

方向 07 的最小 source-pair ERM gate 已完成，但条件排序的 fold 方向不一致，pooled 平均提升未达冻结参考且配对区间宽。故 disposition 为 `insufficient_evidence`，不进入环境元优化；详情见 `07_compatibility_domain_generalization/records/DIRECTION_DISPOSITION.md`。
