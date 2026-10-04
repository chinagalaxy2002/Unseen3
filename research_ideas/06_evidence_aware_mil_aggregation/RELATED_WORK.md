# Idea 06 候选证据与聚合：近邻及 novelty 边界

状态：`draft_not_executed`。2026-10-04 阅读原论文方法正文；下面按七轴比较，不根据题目相似性否决方向。

## 1. QVHighlights / Moment-DETR

[原论文](https://arxiv.org/html/2107.09609)，核对方法 §4。这是本项目候选 set 的直接结构先例。

| 轴 | 该工作 |
|---|---|
| 输入 | 视频/text 特征与 learnable moment queries |
| 监督 | GT spans、foreground/background 与 saliency |
| Score | decoder每候选类别置信度及 span，另有 clip saliency |
| Negatives | unmatched proposals 的 background labels |
| Objective | bipartite matching、分类/回归、saliency ranking |
| Semantic novelty | 常规 moment/highlight benchmark，无本项目 holdout |
| Evaluation | moment mAP/Recall、highlight mAP/Hit |

unmatched proposal background不等于全视频 query absence；同视频存在某 moment 时，其他 proposals不匹配并不能训练“整个视频不存在”的拒绝判断。因此本方向保留候选 set，新增 full-query negative evidence 与视频级 conditional evaluation，不能把原 foreground confidence 默认当正确存在性分数。

## 2. BCANet / Generalized Video Moment Retrieval

[ICLR 原文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf)，核对 §4 的 region evidence 与 no-target 判空。它已把候选证据与 absence 连接；“用最高候选 evidence 拒绝”不是新机制。

| 轴 | 该工作 |
|---|---|
| 输入 | 视频/text 序列与 region queries |
| 监督 | GT boundary与region support |
| Score | 各候选 evidence，全部低于阈值判空 |
| Negatives | relation edits/cross-video no-target |
| Objective | boundary contrast、region proxy、matching |
| Semantic novelty | 未设 action/composition semantic holdout |
| Evaluation | N/T accuracy、single/multi moment定位 |

本方向推进候选冗余与背景响应在语义变化下的稳定性，而不是再提出 no-target过滤。没有同等 temporal supervision 时，不能使用 BCANet式 oracle支持来美化对照。

## 3. NA-VMR / Moment of Untruth

[原论文](https://arxiv.org/html/2502.08544v1)，核对 §3.4 与 Appendix C。

| 轴 | 该工作 |
|---|---|
| 输入 | indicator/saliency temporal scores |
| 监督 | query existence；负例局部 score置零 |
| Score | RNN汇总后MLP二分类 |
| Negatives | 过滤 ID mismatch 与场景外 OOD queries |
| Objective | binary/foreground/saliency losses，负例不回归时间 |
| Semantic novelty | ID/OOD负例，非 unseen-positive semantic protocol |
| Evaluation | rejection accuracy、positive recall |

所以本方案 candidate-negative supervision 本身已有先例。值得保留的是 full-query absent下压低局部响应；新增空间是明确区分 supervision与aggregation、处理重复候选相关性，并用 unseen-positive/negative与两向条件测试检验是否可迁移。

## 4. SHINE

[原论文](https://arxiv.org/html/2407.05118)，核对 §3.2–3.4。

| 轴 | 该工作 |
|---|---|
| 输入 | video/query及层次hard queries |
| 监督 | GT interval与pseudo saliency |
| Score | clip saliency；GT内top-k mean与GT外max用于ranking |
| Negatives | 五类primitives的LLM编辑和batch queries |
| Objective | coarse/fine saliency ranking与base localization |
| Semantic novelty | Novel-Composition、Novel-Word |
| Evaluation | R@IoU/mIoU |

它已讨论区间尺度与不同negative语义的saliency响应，不能把 length-aware top-k/细粒度负例当无人研究。本方向关注 full-video query absence AUROC、冗余候选与video条件控制；而语义编辑不自动保证整个视频absence，仍需独立核验。

## 5. 与项目 V5 的直接边界

[V5 QueryConditionedReadout](../../models/moment_detr_gmr_evidence_v5/readouts.py) 已有 candidate-wise MLP和 `logsumexp−logK`。固定K的减logK只是统一offset；不可能单独改善AUROC。若重跑相同公式有收益，原因必在候选scores、监督、温度或优化，而非“新增length normalization”。

## 6. 可发展的 novelty argument

已有候选式grounding、GVMR与NA-VMR说明了局部置信度和负例监督的重要性，SHINE进一步研究语义编辑下的saliency结构。本方向保留这些基础，针对semantic novelty中尚需要受控解释的failure：相关proposals与部分匹配背景能否在存在性score中形成不当证据累积。拟新增完整query证据监督与明确的冗余group aggregation，并把二者作用分开验证，同时要求自然pooled AUROC、same-query/crossed compatibility与真实重复事件保留成立。其意义不是创造另一个pooling名称，而是把“至少一条完整事件证据成立”与“许多重复高响应”在监督及推理中区分，并检验该区别能否跨语义迁移。

当前分类更接近**新组合针对明确failure（C）**，其中candidate-negative与MIL核心已有（B）。只有当相关性机制在相同scores/监督下有独立、跨语义收益，才有依据提高为明确新增机制。已知duplicate invariance与普通NMS本身不能支撑此判断。

## 7. 最强反对意见与回应边界

“普通MIL或NMS，加GT支持训练，改善并非semantic novelty机制。”应对是固定candidate-score比较、supervision×aggregation因子实验、无label grouping、provenance gate、U+保留、条件控制与真实重复事件审计。若aggregation无独立价值，应保留局部existence supervision方向，而不是为保住结构声称所有改动都必要。
