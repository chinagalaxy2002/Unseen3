# 01 · 最近邻方法与 novelty argument

状态：`draft_not_executed`。检索核验截至 2026-10-04；下面是对 primary full text 的方法边界比较，并不声称穷尽所有相关工作。研究者判断与论文事实分开。本文新机制未被实验验证。

## 核验材料

| 工作 | Primary 原文与已核读位置 | 最近之处 |
|---|---|---|
| Liu, Qu, Hu，D-TSG，MM 2022 | [Reducing the Vision and Language Bias for Temporal Sentence Grounding](https://arxiv.org/html/2207.13457)，§3.3–3.4 | 特征去偏，以及固定 video / query 的两轴对比 |
| Yang 等，MVMR / CroCs，CIKM 2024 | [MVMR: A New Framework for Evaluating Faithfulness of Video Moment Retrieval against Multiple Distractors](https://arxiv.org/html/2309.16701v3)，§3–5、CroCs 方法部分 | 每 query 多视频检索、双向匹配与负例过滤 |
| Thrush 等，Winoground，CVPR 2022 | [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/html/2204.03162)，数据与评价定义 | 两图两句交叉匹配、text/image/group 判据 |
| Cheng 等，SHINE，ECCV 2024 | [SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](https://arxiv.org/html/2407.05118)，primitive dictionary、hierarchical contrast、split/evaluation | Novel composition/word 下的语义 hard negatives |
| Ding 等，SoccerGMR，arXiv 2026 | [SoccerGMR](https://arxiv.org/html/2605.02623)，任务定义、数据与 existence head | 明确的 present/absent GMR 与 AUROC |

出处名称以 primary 页面为准；2026 arXiv 工作不凭时间或仓库文件名推断已正式接收。方法重叠不能据此取消探索，但必须成为对照。

## 七个维度比较

| 工作 | 输入 | Supervision | Score | Negative construction | Objective | Semantic novelty | Evaluation target |
|---|---|---|---|---|---|---|---|
| D-TSG | video + sentence | Moment 边界与相关对比 | Grounding / cross-modal matching | 改变视频或 query，并使用名词/动词信息 | 边界、distillation、双轴 contrast | 不以本项目 semantic-held-out existence 为核心 | Temporal localization |
| MVMR/CroCs | query + 多 candidate videos | Positive moments、匹配关系 | Moment quality 与跨模态匹配 | 文本/视觉语义过滤 distractors | Localization、cross-modal positive/negative learning | 非本项目 S/U existence 分层 | 多视频正确 moment 排序 |
| Winoground | 2 images + 2 captions | 交叉正确/错误匹配标签 | Image-text compatibility | 控制组成及匹配反转 | 本身主要是评价集 | Compositional challenge，非 temporal holdout | Row/column/group correctness |
| SHINE | video + sentence / primitives | Positive grounding、hierarchical semantics | Saliency / grounding scores | Primitive 替换的 hierarchical negatives | 多粒度对比及定位 | Novel compositions / words | Positive query grounding |
| SoccerGMR | soccer video windows + query | Existence + positive localization | Scalar existence 和 intervals | 事件内外窗口并核验 | BCE 与 grounding | 不以新 semantic 的 U+/U- 为核心 | Existence AUROC 与定位 |
| 本方向 | 2 videos × 2 exact natural queries | 四格完整存在/不存在标签，正格可有 interval | 共享 scalar existence logit | 只闭合可靠标签，保留 natural main | Natural BCE + row/column ranking；joint 是单独可选扩展 | Seen-only 开发、task-level semantic holdout | 原始 pooled U AUC + 双条件/单模态审计 |

## 真正的重叠与实质扩展

**D-TSG 是监督机制的强最近邻。** 它已有固定 video / query 的对比，不可声称首次双轴训练。我们的可检验扩展是完整四格的 whole-video absence 与 semantic-held-out existence，而不是更换 contrastive loss 的名称。基础 `Lcross` 等于该矩阵上的双向 pairwise loss，已在 IDEA 中明确；对照必须固定支持和预算。

**MVMR/CroCs 是问题组织的强最近邻。** 它检验多视频 distractors 下的定位忠实性，使用语义过滤降低 false negatives。我们的标签要求更明确：没有 target moment 的全视频 absence，以及自然 pooled existence 与条件排序同时考核。现有过滤可以帮助生成候选，但不能单独认证 absence。

**Winoground 是评价结构的强最近邻。** 四格形式、行列和 group 指标均已有，因此不是原创评价形状。本项目把其可识别性用于 temporal existence 的监督单元，同时保持原始分布 AUROC。视频可以包含多个相反动作，不能直接把静态 caption matching 的互斥假定搬入。

**SHINE 说明语义分解与 hard negatives 已有成功路线。** 我们关注新语义下的存在/不存在，而非只对确定 present 的句子定位；采用同一自然 query 的跨视频标签翻转，是与编辑句子辨别不同的监督信息。应以其思想的 existence 适配为 baseline，而非因它已有关键词就否定新方向。

**SoccerGMR 已覆盖存在性 AUROC。** 本方向不能把 GMR 或 BCE-existence head 当新任务；新增问题是 task-semantic holdout 的跨 query AUROC 退化与纯单模态解释之间的分离，需要保持主指标并增加条件验证。

## 可用于论文的 novelty argument

已有工作分别发展了 temporal grounding 的双轴对比、多视频 distractor 评价、组成性交叉匹配和 present/absent GMR。我们的推进在于将这些思想组织为 semantic-novelty existence 的可识别监督：两段视频与两条完全相同编码的自然 query 构成已核验的闭合标签矩阵，使两类单模态边际解释都无法满足所有条件翻转；同时保留自然分布的 pooled AUROC，检验条件兼容性是否真正缓解跨语义退化。新增贡献需要由相同样本曝光的 BCE、单向及双向 ranking 对照证明，而不是依靠新的 loss 命名。若最差约束耦合带来额外稳定收益，再将其作为监督机制扩展单独报告。

## Novelty 判定与 reviewer objection

当前定位 **B/C**：核心对比和四格形状已有，semantic-novelty whole-video existence 的可靠监督与联合评价构成待验证扩展。若最终只是复现已有关联排序，则方法部分降为 A，仍可保留经验证的数据/评价贡献。若 coupling loss 严格优于同数据的双向排序，才讨论 D 层的具体机制。

最强异议：“额外 hard negatives 加双向 ranking 就能解释全部收益。”最有说服力的回应是 exposure-matched BCE 和 nearest-neighbor ranking controls，以及自然 negative / 同场景 matrix / 三 backbone 的一致结果。不能用“相关论文没报本 benchmark”作为唯一 novelty argument。
