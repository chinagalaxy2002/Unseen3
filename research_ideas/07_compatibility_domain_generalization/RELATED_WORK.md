# Idea 07：Domain Generalization 与 temporal debiasing 的实质比较

> 核对日期：2026-10-04；状态：`draft_not_executed`。检索并阅读以下 primary 方法正文。环境、训练机制与收益均是拟议设计。  
> 问题：哪些泛化思想已存在；哪种不变性适合 query-conditioned existence；本方向怎样超出普通 DG 套用？

## 1. 七轴方法对照

| 工作 | 输入 | Supervision | Score formulation | Negative construction |
|---|---|---|---|---|
| IRM，Arjovsky 等，2019/2020 arXiv | 多环境 x | y + training environment | 共享 predictor w∘Φ | 非特定 video negatives |
| Group DRO，Sagawa 等，ICLR 2020 | x + train group | y + group | task classifier score | 按已知 spurious attribute×label groups |
| MLDG，Li 等，AAAI 2018 | 多 source-domain x | supervised y / RL rewards + domain | 单共享 task model | 非 video-specific negatives |
| DomainBed，Gulrajani、Lopez-Paz，ICLR 2021 | multi-domain image benchmarks | class labels + domain | ERM/DG classifiers | 非 existence negatives |
| D-TSG，Liu、Qu、Hu，MM 2022 | video + sentence | moment boundaries、matching/distillation | boundary prediction + debiased features | 两向换 video/query、noun/verb 近邻 |
| SHINE，Cheng 等，ECCV 2024 | video + layered queries | moment/saliency | clip saliency 与 grounding score | plausible primitive edits + batch negatives |
| CVA，Moon 等，CVPR 2026 | video + query | GT spans + training alignment | query-aware alignment/grounding | query-aware background replacement、boundary-related contrast |
| 本方向 | video/query + trusted pair units | whole-video existence + train-side environments | 同结构 existence score；条件差值参与训练 | 固定且有标签的 same-q/crossed units，不任意标跨视频负例 |

| 工作 | Training objective | Semantic novelty setting | Evaluation target |
|---|---|---|---|
| IRM | 环境内同一 optimal classifier；IRMv1 gradient penalty | 多环境 OOD 理论/实验 | OOD task risk |
| Group DRO | worst-group risk + regularization | known training group mixture shift | worst-group / average accuracy |
| MLDG | meta-train update 同时改善 meta-test domain | unseen domains、共享任务标签空间 | classification / RL generalization |
| DomainBed | 公平算法与选模比较 | held-out domains | out-of-domain accuracy |
| D-TSG | grounding + bias identification/distillation + contrast | conventional/rare queries | R@IoU |
| SHINE | hierarchical saliency ranking + grounding | Novel-Word / Novel-Composition | R@IoU、mIoU |
| CVA | context-aware alignment、boundary contrast、temporal modeling | temporal grounding 泛化/鲁棒性 | localization / highlight metrics |
| 本方向 | 全量 natural BCE + environment transfer of pair risks | 任务级 inner semantic holdout，再正式 semantic novelty | 原始 U+ vs U− pooled AUROC + 两向与四格条件排序 |

## 2. DG 的真实机制边界

**IRM** §3 的目标是找到让同一 classifier 在各环境最优的表示，IRMv1 用固定 dummy scalar 的 risk gradient penalty 近似约束；它不是任意 feature distribution matching。原文也讨论 feature distribution alignment 可能要求错误的不变性。本方向继承“稳定预测规则”思想，但动作/对象身份不应被抹掉，泛化监督单位改为可信 compatibility differences。[全文](https://arxiv.org/pdf/1907.02893)

**Group DRO** §2–3 使用已知 groups 的最差风险，强调低 worst-group training loss 不等于低 test loss，正则/early stopping 重要。因此它支持强对照和支持规模审计，不支持再次使用 V3 的含噪 group-max + coverage 丢失。现阶段拟议主方案不是新的 DRO 算法。[全文](https://arxiv.org/pdf/1911.08731)

**MLDG** 的 Methodology/Algorithm 1 把 source domains 分成 meta-train 与 meta-test，经一次虚拟梯度更新后计算另一个环境风险；目标要求 source 改善可迁移。Idea 07 的虚拟更新公式是它的应用。可能的新增贡献来自关系风险与环境有效性，而不是宣称发明 episodic training。[全文](https://arxiv.org/pdf/1710.03463)，[AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/11596)

**DomainBed** §3–5 强调模型选择是算法一部分，公平 tuned ERM 是必要基线，target-domain oracle selection 不能与 source-only selection 混比。本方向因此固定 inner_val_seen checkpoint，Novel-dev 只作为跨 run 开发，并用相同 pairs/compute 的 ERM，而不是只比较原 canonical checkpoint。[全文](https://arxiv.org/pdf/2007.01434)，[作者实现](https://github.com/facebookresearch/DomainBed)

IRM 的保证也不应泛化到任意新语义。**The Risks of Invariant Risk Minimization** 分析了识别与 OOD 失败条件，是本方向“环境风险一致不等于因果兼容性”的反证参照；不能以 DG 名称替代真实 intervention controls。[全文](https://arxiv.org/pdf/2010.05761)

## 3. Temporal grounding 邻域的真正重合

**D-TSG** 已把视觉 salient activity、noun-only、verb-only 信号作为偏置候选，并识别哪些有害，以保留有用语义。它与本方向共同反对无差别删除语言信息；区别是 D-TSG 用 feature distillation 和 localization supervision，这里在 complete video existence pairs 上测试跨 construction/semantic-family 风险迁移，不需要多条 biased backbone branches。[原文 §3](https://arxiv.org/html/2207.13457)

**SHINE** 已针对 novel words/compositions，使用 LLM 分层可行负例和 saliency ranking。它是很强的 semantic-grounding 先例，不能因评价不同就忽略。新增边界是：genuine absent-event 标签、同 q 跨视频、加性 prior 受控的四格评价，以及 environment-aware optimization 是否超过相同 pairs 的 ERM；单纯换一个 negative-ranking loss 不够。[原文 §3–4](https://arxiv.org/html/2407.05118)

**CVA** 用 query-aware context/background manipulation 与 boundary-aware alignment 研究 grounding 的背景与时间鲁棒性。它提示背景干预已有，而 event identity 应保留。本方向第一轮不用人工剪接/背景替换，只从可信 pair environments 检验存在性关系；如果以后引入 CVA 类增强，须单列 augmentation 对照，不把所有提升归因于 DG。[原文](https://arxiv.org/html/2603.24934)

三者分别覆盖 feature 去偏、语义负例与 context intervention，但并未因这些相似思想让本方向失去价值。研究空间在于对当前 failure 的识别：**究竟要迁移哪个关系、哪类环境变化、以及是否真正改善 unseen present/absent 排序。**

## 4. 可发展的 novelty argument

现有 DG 方法通常在固定标签分类任务中稳定 predictor，temporal debiasing 则主要通过 feature distillation、负文本 saliency ranking 或背景增强改善定位。我们拟将泛化对象明确为 query-conditioned event-existence 的条件兼容性差值：用标签齐全 pairs 建立 train-side semantic-family/negative-construction 环境，保留决定事件身份的语义及自然主训练覆盖，要求跨环境更新改善另一环境的关系风险；随后在严格任务级 semantic holdout 的 pooled AUROC、同 query/同视频判别与四格控制上验证收益。新增空间不是再命名 MLDG，而是可学习关系的监督单位、有效环境的构造条件，以及从 source robustness 到真实视频存在性迁移的完整证据链。

当前 novelty 判断为**已有核心思想，GMR setting 与关系监督/评价机制的实质扩展**。若提出了有明确性质、不同于普通 pair MLDG 的环境识别或 compatibility-preserving 约束，才可能进一步成为新的机制；本文件不提前宣称已达到该层次。优先发展并验证，而不是因 IRM/MLDG 已有就否定。

## 5. Reviewer objection 与验证要求

| 质疑 | 必须提供的对照/边界 |
|---|---|
| 只是 MLDG 换任务 | 同 pairs 的 ERM、MLDG、IRM/Group DRO，解释 relational units 的新增价值 |
| 环境是人为划分，收益来自随机正则 | random-environment control、支持与相关性审计 |
| 把新 action 当 nuisance 删除，损失完整语义 | task-family 与 nuisance 区分，hard-positive 与 action/composition 分层 |
| teacher 已见过 episode semantics | exposure manifest，不能把 training-side episode 叫真正 unseen |
| novelty gain 由更多计算 | example/update/forward/backward 与实际设备 budget 对照 |
| 更稳 group risk 不是真 grounding | conditional/crossed controls 与原始 U AUROC 同时改善 |

## 6. 核验书目

1. Arjovsky、Bottou、Gulrajani、Lopez-Paz，Invariant Risk Minimization，arXiv:1907.02893（2019，本文读取 PDF 版本）：[全文](https://arxiv.org/pdf/1907.02893)。
2. Sagawa、Koh、Hashimoto、Liang，Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization，ICLR 2020：[全文](https://arxiv.org/pdf/1911.08731)。
3. Li、Yang、Song、Hospedales，Learning to Generalize: Meta-Learning for Domain Generalization，AAAI 2018：[全文](https://arxiv.org/pdf/1710.03463)。
4. Gulrajani、Lopez-Paz，In Search of Lost Domain Generalization，ICLR 2021：[全文](https://arxiv.org/pdf/2007.01434)。
5. Rosenfeld、Ravikumar、Risteski，The Risks of Invariant Risk Minimization，ICLR 2021：[全文](https://arxiv.org/pdf/2010.05761)。
6. Liu、Qu、Hu，Reducing the Vision and Language Bias for Temporal Sentence Grounding，MM 2022：[全文](https://arxiv.org/html/2207.13457)。
7. Cheng 等，SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding，ECCV 2024：[全文](https://arxiv.org/html/2407.05118)。
8. Moon、Lee、Seo、Im，CVA: Context-aware Video-text Alignment for Video Temporal Grounding，CVPR 2026：[全文](https://arxiv.org/html/2603.24934)，[作者项目页与正式书目信息](https://byeol3325.github.io/projects/CVA/)。
