# Idea 04：最小实验与升级顺序

> 状态：`draft_not_executed`。以下步骤均为未来计划；本次未构造 manifest、导出 feature、训练或评测。  
> 前置说明：[IDEA.md](IDEA.md)；统一数据边界与统计规则：[COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 本轮唯一自变量

检验相同 query 的辅助标签边际配平是否优于**同样数据、同样计算**的原比例辅助训练。先不加入 pair ranking、DG、reference centering、新语言模型或额外负例生成，避免多个机制同时改变。

## 2. 数据启动条件与未来产物

| 未来产物 | 必须记录的字段/检查 | 不满足时如何处理 |
|---|---|---|
| `query_support_manifest.jsonl` | fold、qid、video_id、raw/canonical query、y、标签来源、source linkage、review status | 无双标签支持的 q 只在主 stream 保留 |
| `support_audit.md` | query 数、独立视频数、每 q 各标签覆盖、跨 q 视频复用、construction strata | 少数模板占主导则先补可信支持，不进入大规模训练 |
| `aux_sampling_spec.json` | 均匀 q→均匀 y→类内视频的概率、预算、seed、重复上限记录 | 不用当前预测动态改权重 |
| `data_boundary_audit.md` | inner_train_seen 与 inner_val_seen/Novel-dev 的视频/文本/来源边界 | 发现泄漏需重建边界，不能以更高分为通过 |

这些只是命名建议，没有在本次生成。相同 q 关联的第一轮规范化只处理保守大小写/空白规则，保留人工确认；不得因为 q embedding 相似就把标签传播到未标注视频。

## 3. 第一轮实验组

| 组别 | 主 stream | 辅助 stream | 回答的问题 |
|---|---|---|---|
| M0 | 完整自然 BCE | 无 | canonical 小 head 对照 |
| M1 | 与 M0 相同 | `Q_*` 同支持集，按预定原标签比例 | 额外支持曝光是否有用 |
| M2 | 与 M0 相同 | `Q_*` 内每 q 各标签 1/2，q 均匀 | 精确 query 配平是否新增收益 |
| Cq/Cv controls | 相同训练数据边界 | 明确与多模态对应的 sampling | 纯语言/视频的边际预测能力 |

M1 与 M2 使用同一支持全集、同 auxiliary example/forward 数、同 head、相同 optimizer、相同总 steps。M1 的 q 分布也固定为与 M2 相同，区别仅在该 q 内的 y 权重；否则 q 频率变化会混杂效果。M0 与有辅助组比较时增加 compute-matched 自然 BCE duplicate-stream 控制，不能只比较 wall-clock 不同的两个训练。

Frozen backbone 的缓存只用于原视频/原 query 对；已有 query-conditioned representations 不可交换后假装完成 shuffled-video control。若控制需要新视频与 q 的对应输入，未来应重新运行 frozen backbone。

## 4. 预算与选模

先选择一个 action 与一个 composition inner fold、一个固定 seed 做 feasibility pilot。辅助权重初始固定 `λ=0.25`；仅在预定义 Seen validation 安全条件不满足时比较 `λ=0.10`，记录为第二个 run。此数值是计划，不是经验最优值。两组使用相同小预算，避免在 Novel-dev 上寻找各组最优 epoch 或大网格。

学习率与 head 结构优先沿用已核验 readout 设置；若需变更，所有组同时变更。训练用 inner_train_seen；每个 run checkpoint 根据 inner_val_seen 的同一规则选定，冻结后一次性评价 Novel-dev。Novel-dev 只在有限跨 run 开发中使用，属于开发集，不能后来写成独立确认测试。

通过 pilot 再完成四个既有 inner folds；最终方法确认至少三 seed，再考虑三 backbone 与正式五 split。历史正式五 split 已影响研究方向，汇报时标记 exploratory。

## 5. 固定报告表

每个 fold/seed 同时报告以下指标，不能根据哪项最好临时替换主目标。

| 报告项 | 分布/子集 | 核心解释 |
|---|---|---|
| 原始 pooled Seen/Novel AUROC | 原 benchmark 全体，分布不重采样 | 主目标与 Seen 代价 |
| 支持/非支持 pooled AUROC | 保持各子集原分布，记录覆盖 | 是否只记住支持模板 |
| Same-query PairAcc/AUROC | 有可信 present/absent 的同文本跨视频 | query prior 无法直接区分标签 |
| Same-video PairAcc | 源关联正负 q | 与已有 V5 诊断衔接，但不是视频证据充分条件 |
| 四格 group 与行/列排序 | 标签齐全 crossed subset，等权 | 加性 query/video 边际控制 |
| Cq/Cv 与 shuffled controls | 原分布与辅助分布分别报告 | 分数依赖与边际能力 |
| Edit-type、长度、场景/source strata | 预定义且有样本支持 | matching 后的 video nuisance |
| Raw localization identity | frozen 输入与输出对应 | 确认非定位训练代价 |
| Unique coverage/repeat count | 每 q、每视频 | 防止重复曝光冒充数据多样性 |

随机 shuffle 保留原标签的结果仅是 input-dependence control；不使用未经重新标注的新 pair 评价真实存在性。Cq 在精确平衡辅助分布上的 chance 性质要求按实际辅助权重评价；原始 benchmark 上 Cq 不必降到 0.5。

## 6. 推进与停止判据

采用共同协议预先固定的建议门槛：四 fold 原始 Novel macro AUROC 至少 +2 pp、至少 3/4 正向、Seen 下降 ≤1 pp，并有 same-query/same-video/crossed 的一致支持。CI 与 seed dispersion 共同报告；少量共享视频的 quartet 不满足稳定确认要求时，只作 pilot。

- M1≈M2 且都超过 M0：先归因于支持数据/额外优化，不主张 query 配平有效。
- M2 只在辅助平衡 challenge 变好：停止扩全 backbone，优先解释 coverage 和迁移缺口。
- M2 pooled 变好而条件指标变差：不作为 compatibility 方法推进；可转 score comparability 分析。
- M2 Seen 显著损失：下调预定 λ 或放弃该配平强度，不能靠 gap 变小声称成功。
- 可信支持与实现核验通过后多 seed 全无收益：保留负结果，转 independent feature adapter 或 complete-event witness；不重复堆 readout 容量。

## 7. 第二轮只在首轮通过后开展

比较 `M2` 与 `M2 + same-query pair ranking`，所有组使用相同 pairs，以判断 conditional objective 是否提供超出配平的作用。另做背景/时长 matched subset 的 sensitivity analysis；matching 使用训练侧预先固定 nuisance，不按待评价的模型分数筛样本。不要同时引入 hard-negative 生成与 DG 后仍沿用首轮因果归因。

实施成本低；1–2 GPU 是拟定设备规模，准确运行时长与支持规模需未来测量。文档完成不代表任何实验已授权启动或已完成。
