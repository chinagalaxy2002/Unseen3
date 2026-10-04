# 02 · 参考归一化、交互分解与 novelty 边界

状态：`draft_not_executed`。检索截至 2026-10-04，使用 primary papers 的 score / objective 定义核验。没有以标题相似性否定 idea，也不因 setting 不同而自动主张新机制。

## 已核验的强最近邻

| 工作 | Primary 原文与核读位置 | 为什么必须比较 |
|---|---|---|
| Bogolin 等，QB-Norm，CVPR 2022 | [Cross Modal Retrieval with Querybank Normalisation](https://arxiv.org/abs/2112.12777)，正文 normalization / Dynamic Inverted Softmax；[正式入口](https://openaccess.thecvf.com/content/CVPR2022/html/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.html) | Query bank 在无 test-query pool 条件下修正 gallery hubness |
| Wang, Jian, Xue，DBNorm，EMNLP 2023 | [Balance Act: Mitigating Hubness in Cross-Modal Retrieval with Query and Gallery Banks](https://arxiv.org/html/2310.11612)，§2.3；[正式入口](https://aclanthology.org/2023.emnlp-main.652/) | 两类 bank 及双重 similarity normalization |
| Conneau 等，CSLS，ICLR 2018 | [Word Translation Without Parallel Data](https://arxiv.org/html/1710.04087)，§2.3 Eq.6 | 两侧局部密度相减的经典分数 |
| Joulin 等，RCSLS，EMNLP 2018 | [Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion](https://aclanthology.org/D18-1330.pdf)，§3 Eq.4 | 将检索校正纳入训练，直接反驳“首次 train-time normalization” |
| Wang, Chen, Shen，CausalVTG，NeurIPS 2025 | [CausalVTG: Towards Robust Video Temporal Grounding via Causal Inference](https://openreview.net/pdf/795816bb4e9201f21b3ea3ab10ba87000e65eba6.pdf)，counterfactual learning / relevance estimator | 错配 V-Q relevance learning，与条件存在性相邻 |

QB-Norm arXiv 首发于 2021、正式 CVPR 2022；日期与正式发表年份分开。CausalVTG 的题名及作者以原文为准，不将其 relevance prediction 等同本项目 semantic-held-out whole-video absence。

## 七轴对照

| 工作 | 输入 | Supervision | Score formulation | Negative construction | Training objective | Semantic novelty | Evaluation target |
|---|---|---|---|---|---|---|---|
| QB-Norm | Query embedding、gallery、train query bank | 冻结既有embedding | Inverted softmax / dynamic correction | 不新增existence negatives | 无需重新训练 | 不要求task semantic holdout | Cross-modal retrieval recall |
| DBNorm | 当前query、gallery、train query/gallery banks | 冻结embedding | Dual inverted softmax类 | 不新增whole-video absence | Post-processing | 非本项目S+/S-/U+/U- | Text-image/video/audio retrieval |
| CSLS | 两语言word embeddings | 无监督alignment或词典适配 | `2cos−rT−rS`，local NN means | 邻域竞争，不是event absence | Alignment后做retrieval scaling | 未配对词可检索，非event novelty | Bilingual lexicon retrieval |
| RCSLS | Word vectors与seed lexicon | Translation pairs | Relaxed CSLS criterion | 邻域dictionary candidates | 直接优化retrieval criterion | 非video-event semantic split | Word translation accuracy |
| CausalVTG | Video-query pairs | Grounding、counterfactual relevance | Relevant-pair estimator及grounding | 错配video/query | Causal/debiasing与relevance learning | 非本项目Novel存在/不存在划分 | Temporal grounding及relevance |
| 本方向 | 当前V/Q + fixed Seen-only reference banks | Whole-video existence BCE，条件监督单独factor | `f−ErefVf−ErefQf+ErefV,Qf+b0` | 仅可靠absence支持；reference不是certified negative | Centered score上训练，raw/posthoc/trained拆开 | Task-levelheld-outaction/composition | NaturalpooledU AUC + same-query/四格/单模态controls |

## 精确区别，不靠“我们也做去偏”

QB-Norm 的典型 inverted-softmax 用 reference queries 对每个 gallery item 的可匹配性归一化；它已证明 train bank 可避免并发 test query 需求。我们新增的是两个固定边际期望相减及 existence训练，不是 bank 概念本身。

DBNorm 已有 query/gallery banks，不能声称首次双 bank。其 gallery侧考虑 gallery与reference gallery 的相似度，不能不加说明就等同“固定当前 Q、换一组 reference videos 的 cross-modal mean”。我们应分别比较，不能只把双侧归一化重命名。

CSLS 的 `2cos−localmean−localmean` 与本方向形似，核心系数和 query-dependent nearest-neighbor sets 不同。这里采用 fixed product reference measure、原分数系数为1，以获得任意可加单模态项严格相消的代数性质；双线性情形就是 centered features，不声称该代数操作首次出现。

RCSLS 已直接在校正检索准则上学习映射，所以“我们在训练时归一化而已有工作都只后处理”是错误叙事。可推进的差异在任务监督、fixed-bank interaction constraint、semantic-held-out存在性与机制controls，需实证证明不只是经典retrieval-loss迁移。

CausalVTG 已训练错配 V-Q relevance，不可声称首次反事实兼容性。我们的分解是统计参考中心化，并无 front-door / 因果 effect 的识别保证；其 relevance learning 可作为辅助监督基线。若只通过 scene-query 共现改善，不应改称因果去偏。

## 可用于论文的 novelty argument

相似度归一化、双侧密度校正及训练检索准则均有充分先例。我们的研究推进是将可加单模态项的消除明确为 GMR existence score 的约束，并在该交互分数上训练 whole-video presence/absence，检验这种约束能否缓解 task-semantic novelty 下的跨 query AUROC 退化。固定 Seen-only reference 的四项公式允许验证纯 query/video 分量的相消，又通过 raw、post-hoc、centered-trained 三组拆开打分修正与交互学习；same-query、闭合四格、错配视频和自然主分布评价共同限定“收益来自何种证据”的解释。贡献是可验证的 existence formulation 与训练/机制分离，而非声称首次归一化或首次交互建模。

## Novelty 判定与最低证据要求

当前 **B/C**。若只有双线性中心化与post-hoc收益，算法核心接近 A，应用/诊断可有价值；若 centered training、条件监督或新reference机制相对最近邻产生稳定且可归因的提升，则可形成实质 B/C 扩展。D 需要更明确的新机制与同数据/预算消融，而非仅更长公式。

最强异议：“这是 CSLS/QB-Norm/RCSLS 的应用，reference 重排让AUC好看而已。”回应需要同时满足：同 scorer posthoc不能解释全部收益；训练后的四格D或其他条件证据改善；多banks稳定；自然U+保留及主AUC提高；单模态/错配对照不能复现收益。只报告pooled提升不够。
