# Idea 05 近邻方法与可发展的 novelty

状态：`draft_not_executed`。核验日期 2026-10-04；来源为原论文方法正文。以下不是穷尽文献，也不是“未检索到就等于绝对原创”。相似工作用于确定可保留的核心思想与推进空间。

## 1. 近邻核验与七轴比较

### 1.1 Soccer-GMR：Retrieving Any Relevant Moments

[原文](https://arxiv.org/html/2605.02623)，重点核对 task、§3.3、existence adapter 方法。它已研究 null/single/multi moment，并显式报告 AUROC，所以“新增 existence head”或“首次 AUROC rejection”不是本方向贡献。

| 轴 | 原工作 |
|---|---|
| 输入 | 视频窗口、事件 query、VMR 表示 |
| 监督 | 窗口内外事件构造的 existence 与 moments |
| Score | pooled cross-modal state 上 MLP existence score |
| Negatives | 同事件在其他窗口中的 naturally null pairs |
| Objective | existence BCE 与 VMR 学习 |
| Semantic novelty | 非本项目的 action/composition holdout |
| Evaluation | rejection/AUROC、定位、整体 GMR |

可推进点是存在性表示路径与语义迁移：用原特征旁路隔离 source localization adaptation，检验相同 query 换真实视频后的标签翻转。没有证据前，不能反向声称原 GMR 完全靠文本。

### 1.2 Moment of Untruth / NA-VMR

[原文](https://arxiv.org/html/2502.08544v1)，核对 §3.3–3.4、§4。该工作已有附加二分类分支，保留 query-rejection 思想；区别应落在输入证据路径和迁移协议。

| 轴 | 原工作 |
|---|---|
| 输入 | 视频、query；indicator 与 saliency 序列 |
| 监督 | existence 标签、正例 span；负例无 boundary loss |
| Score | indicator/saliency 合并，RNN→MLP→sigmoid |
| Negatives | 筛选 cross-video ID 与 LLM OOD scenarios |
| Objective | BCE；负例 foreground/saliency 压低 |
| Semantic novelty | ID/OOD 负 query，不等于 unseen positive semantics |
| Evaluation | rejection accuracy 与 positive R@IoU |

本方案不声称轻量 head 原创；尝试保留冻结原始 evidence，避免仅从已经 source-trained 的 scalar foreground/saliency 推断存在。关键评价同时包含 unseen positives，不能通过“新词都拒绝”成功。

### 1.3 BCANet：Generalized Video Moment Retrieval

[ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf)，核对 §3.4、§4。该方法已经直接处理 CLIP/SlowFast 与 text tokens，并学习投影，故“原始序列+cross attention”不是空白。

| 轴 | 原工作 |
|---|---|
| 输入 | CLIP/SlowFast 序列、CLIP text tokens |
| 监督 | GT moments、boundary/region 支持 |
| Score | decoder evidence scores；均低于 threshold 时判空 |
| Negatives | S/P/O edits 与 cross-video queries |
| Objective | boundary contrast、query-region proxy、matching |
| Semantic novelty | 未设置本项目式语义 holdout |
| Evaluation | no-target 与 multi-target accuracy、定位 |

本方向采用独立 existence 分支、冻结既有定位结果，以表示路径的受控比较研究 semantic novelty；不继承 BCANet 的时间 oracle。输入类似是合理起点，不必因此放弃独立 compatibility 问题。

### 1.4 MVMR / CroCs

[原文](https://arxiv.org/html/2309.16701v3)，核对 §6 与 Appendix B.1。它的重要启示是：视频条件判别需要控制潜在 false negatives，不能默认所有 in-batch mismatch absent。

| 轴 | 原工作 |
|---|---|
| 输入 | query 与多视频候选 moment |
| 监督 | 正 moments、过滤 potential negatives |
| Score | 双 joint spaces 的 IoU 与 mutual matching 分数相乘 |
| Negatives | 语义过滤、两向 informative hard negatives |
| Objective | BCE 与 cross-directional contrastive loss |
| Semantic novelty | 非 unseen semantic existence protocol |
| Evaluation | distractor pool 中 moment R@IoU |

本方案保留可信关系监督，不把跨视频检索排名当跨 query calibrated existence；比对必须使用 full-video present/absent 真值，而非模型相似度即可生成的负标签。

## 2. 本项目已有结构的边界

[V5 readouts](../../models/moment_detr_gmr_evidence_v5/readouts.py) 已有 query projection、cross-attention、FFN 和 normalized logsumexp。新方案的首轮核心变化是原始序列路径，非交互头新颖性。把 R3 的 slots 换 raw-sequence 若只得到 pooled 提升，不足以支撑新论文方法贡献。

## 3. 积极但可验证的 novelty argument

已有 GMR/NA-VMR 证明 existence adapter 可行，BCANet/CroCs 展示了局部跨模态证据与关系监督的价值。本方向保留这些思想，研究其尚未由上述设定直接回答的问题：当语义类别在训练与测试间变化时，定位导向的表示是否仍适合完整视频的事件存在性判别，以及无需修改定位输出的轻量旁路是否可恢复可迁移 compatibility。新增空间不是另一个 cross-attention 名称，而是受控的 pooled/slots/raw-sequence 比较、冻结定位的模块化适配、以及排除语言边际收益的两向条件验证。若这些条件成立，方法可以形成已有核心思想在新 failure mode 下的实质扩展；若只是输入更多带来性能变化，应诚实定位为诊断与工程结果。

当前分类：**已有核心思想，setting/验证/模块化使用有扩展空间（B）**；不预先宣称明确新机制（D）。

## 4. Reviewer objection 与必要应对

“普通 adapter，容量和输入量不同，实验不能证明 source representation 丢信息。”因此需要 head 家族/预算披露、raw mean、Cq/Cv、同 query/crossed controls、训练 competence 检查与跨模型验证。论文宜主张“该路径在这些受控条件下更利于语义泛化”，不要在未做信息论或严格干预证据时主张已经确定信息损失的唯一原因。
