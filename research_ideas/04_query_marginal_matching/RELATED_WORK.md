# Idea 04：近邻文献与 novelty 边界

> 文献核对日期：2026-10-04。状态：`draft_not_executed`。本文件比较已检索 primary full text 与本项目的拟议方法，不把计划写成已有结果。  
> 研究问题：正负 query 语言边际干预的先例是什么；本项目是否新增明确的监督与评价目标；怎样证明收益是 video-conditioned existence？

## 1. 最接近工作的七轴对照

下表不以标题或“去偏”关键词判断等价。先比较输入、监督、score、负例，再比较 objective、semantic novelty 与实际评价目标。

| 工作 | 输入 | Supervision | Score formulation | Negative construction |
|---|---|---|---|---|
| SugarCrepe，Hsieh 等，NeurIPS 2023 D&B | image + candidate captions | 人工核验正/负 captions，作为 benchmark | VLM 相似度；盲语言分数用于审计/筛选 | LLM 可读/合理 edits，human validation，语言 gap 对称筛选 |
| NegCLIP/ARO，Yuksekgonul 等，ICLR 2023 | image + text | image-caption matching | CLIP 图文相似度 | linguistic swaps/shuffles 与相似 image negatives |
| D-TSG，Liu、Qu、Hu，ACM MM 2022 | video + sentence | GT boundaries 与训练 matching | grounding boundary scores，feature debiasing | 同 q 换视频、同视频换 q，noun/verb 近邻 |
| SHINE，Cheng 等，ECCV 2024 | video + original/hierarchical q | GT moment 与 saliency 训练 | clip saliency ranking + DETR grounding | LLM 分层替换 primitives，batch negatives |
| CroCs/MVMR，Yang 等，CIKM 2024 | q + 多视频候选池 | moment/matching labels | mutual matching 与 temporal proposal evidence | 双向 hard negatives 与相似度筛选 |
| 本方向 | video + exact matched q | 可信 whole-video existence y；完整自然主流 | 与对照相同 scalar existence head | 只从已标注同文本双标签支持构造辅助 stream |

| 工作 | Training objective / 是否训练贡献 | Semantic novelty setting | Evaluation target |
|---|---|---|---|
| SugarCrepe | benchmark 数据去偏；非存在性训练目标 | 组合理解 challenge；非本项目 semantic holdout | image-to-text forced-choice accuracy、blind controls |
| NegCLIP/ARO | composition-aware contrastive fine-tuning | order/attribute/relation challenge | caption discrimination + retrieval |
| D-TSG | localization + contrastive/distillation | rare-query/常规 grounding | R@IoU；不是 U+ vs U− existence AUROC |
| SHINE | coarse/fine saliency ranking + grounding losses | Novel-Composition / Novel-Word | localization R@IoU、mIoU |
| CroCs/MVMR | matching BCE 与 cross-directional contrast | 多视频 distractor；不是严格 semantic-holdout existence | multi-video moment retrieval |
| 本方向 | natural BCE + exact-query-balanced auxiliary BCE | Seen-only inner development，之后 unseen action/composition | 原始 pooled AUROC + 同 q/同视频/四格条件指标 |

## 2. 阅读到的方法细节与真正差异

**SugarCrepe** 的 §3–4 解释 plausibility/fluency 为什么使 blind models 有预测性，并用人工验证与两类语言分数 gap 的对称 subsampling 改善 benchmark。它证明语言负例质量不能仅靠“改一个词”保证。本方向借鉴语言控制，但不删除自然 benchmark，不依据当前语言模型自适应筛掉难例；在训练中对**同一完整 q** 的 present/absent 视频配平，建立不同的约束。[原文](https://arxiv.org/html/2306.14610)

**NegCLIP** 的 §3–4 用 caption hard negatives 使 contrastive objective 必须关注组合差异，并保留检索能力。它不是 exact-query 双标签配平。其先例要求本项目加入普通 hard-negative/支持数据曝光对照，避免把更多 hard examples 的收益误归因于语言边际干预；同时不由“NegCLIP 已做 hard negatives”否定训练分布研究。[原文](https://arxiv.org/html/2210.01936)

**D-TSG** 的 §3 通过 video-only、noun-only、verb-only biased branches 捕获偏置，再识别要去掉的部分；也构造两向负例。它明确保留有用 nouns/verbs，提醒本项目不要消除全部语言内容。区别在于这里不增加偏置 distillation branches，干预的是相同文本下的存在性监督边际；最终目标含真正 absent query 的 semantic novelty AUROC。[原文](https://arxiv.org/html/2207.13457)

**SHINE** 的 §3 使用分层可行 primitive replacements 和 coarse-to-fine saliency ranking，目标是 unseen composition 的 temporal localization。它已覆盖“合理语义负例促泛化”的核心思想。本方向不能把合理 edits 本身当新贡献；区别应落在 exact-q 配平、保留主分布、whole-video absence 与独立 video intervention 的机制验证。[原文](https://arxiv.org/html/2407.05118)

**CroCs/MVMR** 用双向 hard-negative matching 在多个视频中寻找 query 的 moment，说明同 q 跨视频监督已有强先例。本方向的潜在增量是专门控制 query-label marginal，并测试 pooled 与 conditional existence 排序的分离；单纯把 cross-video negatives 改名没有新颖性。[原文](https://arxiv.org/html/2309.16701v3)

## 3. 可防守的 novelty argument

已有工作分别通过更合理的负文本、双向 matching、feature distillation 或 saliency ranking 改善图文/视频 grounding；这些思想对本方向是可用基础。我们拟议的扩展是：在具有可信 whole-video present/absent 标签的 exact-query 支持上，构造语言标签边际被精确控制的辅助监督，同时保留自然训练主流的完整覆盖；把这种训练干预的贡献与额外曝光、hardness 和 video nuisance 分开，在 unseen semantic existence 的 pooled AUROC、同 query 跨视频判别、同视频 query 判别和四格控制上共同验证。其意义在于回答当前 Cq 能提高 pooled AUROC 却降低 source-pair 判断时，训练是否还能得到可迁移的视频条件存在性，而不是仅把 benchmark 过滤得更“难”。

当前判定属于**已有核心思想，setting 与监督/验证机制的实质扩展**，不是绝对原创声明。若实验只能证明重采样有用而不能给出迁移和机制证据，方法贡献会偏弱；仍可发展为有价值的数据监督诊断。若 exact-query 支持与 conditional gains 跨三 backbone 成立，则更有机会形成“query-conditioned supervision repairs semantic-novelty existence”的论文主线。

## 4. Reviewer objection 与必要对照

| 最强质疑 | 应提供的具体证据 |
|---|---|
| 只是 class balancing | 说明 q 内边际而非总体比例，M1/M2 同 q distribution 与曝光 |
| 只是更好的 hard negatives | 同支持数据与难度，只有 label weights 改变 |
| 支持子集模板化，无法推广 | 支持/非支持与 action/composition 分层；覆盖和复用公开 |
| 配平后仍靠 video prior | Cv、可信 crossed quartets、背景/时长匹配 |
| 只改分数，没改 grounding | pooled 与两向 conditional 同时改善，shuffle/真实视频换位响应 |

## 5. 核验入口

1. SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality，NeurIPS 2023 D&B：[全文](https://arxiv.org/html/2306.14610)，[作者实现](https://github.com/RAIVNLab/sugar-crepe)。
2. When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It?，ICLR 2023：[全文](https://arxiv.org/html/2210.01936)。
3. Reducing the Vision and Language Bias for Temporal Sentence Grounding，ACM MM 2022：[全文](https://arxiv.org/html/2207.13457)。
4. SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding，ECCV 2024：[全文](https://arxiv.org/html/2407.05118)。
5. Yang 等，MVMR: A New Framework for Evaluating Faithfulness of Video Moment Retrieval against Multiple Distractors，CIKM 2024，arXiv v3：[作者全文](https://arxiv.org/html/2309.16701v3)，[作者实现与接收信息](https://github.com/yny0506/Massive-Videos-Moment-Retrieval)。CroCs 是文中的方法名，不能替代论文标题。
