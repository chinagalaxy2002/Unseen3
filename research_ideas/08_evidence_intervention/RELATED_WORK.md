# 事件证据干预：已有机制、贡献边界与推进空间

状态：`draft_not_executed`。

> 核对日期 2026-10-04。研究问题：如何区分 query-specific evidence removal 与任意输入损伤？已有方法是否保证 whole-video absence？是否在 semantic novelty 下用 natural existence AUROC 验证？以下以原论文为依据，不按“counterfactual”关键词判定重复。机制见 [IDEA](IDEA.md)。

## 1. 最接近四项工作

| 工作与原文 | Input / supervision | Score / negative construction / training | Setting / evaluation | 与本方向的关系 |
|---|---|---|---|---|
| [CCL，NeurIPS2020，§3.1–3.4](https://proceedings.neurips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf) | video/image、sentence；弱监督 bag 关系，无 proposal alignment | gradient 挑 critical proposals；feature/interaction/relation 三层 robust / destructive 变换；Agg(proposal scores) ranking 和 distribution consistency | weakly-supervised grounding；video R@IoU/image 定位 | 目标证据破坏、背景保留与 ranking 核心已存在。本方向应明确新增 validated whole-video labels、matched damage、query specificity 与 semantic-novelty existence |
| [CVA，§3.1–3.3](https://arxiv.org/html/2603.24934) | untrimmed video / query、GT spans/saliency | GT 及附近 context 保留、CLIP relevance 筛 background 来源；context encoder 与 boundary contrast，MR/HD losses | 常规 MR/HD；R/mAP、HIT@1 | query-aware 背景不变性已有；不是简单 random mixing。本方向同时约束正确 label 翻转和保留，并不以非 GT clips 天然 absent |
| [HRVTG，作者论文§3.2、§4、§5](https://github.com/CVL-hub/HRVTG/blob/main/ECCV2026_HRVTG.pdf) | grounding MLLM、test stream queries、teacher pseudo intervals | text edits、Pseudo-GT 删除、hybrid queries；online LoRA GRPO、拒绝/positive consistency rewards 和不对称 KL；区间/∅ | TVGBench-RF；conditional IoU、balanced decision F1 | visual pruning 生成 absence 已有强先例。这里只在 Seen 侧监督；不将单次 teacher 支持删除当穷尽 absence，并用 same-operator damage 控制归因 |
| [Counterfactual Cross-modality Reasoning / CCR，MM2023，§3](https://arxiv.org/html/2308.05648) | video / query、弱监督 pairs、masked query reconstruction | 引入 counterfactual cross-modal 知识，抑制 unmasked query 对 reconstruction 的单模态作用，修正 proposal contrast | weakly-supervised localization；R@IoU | 通过 counterfactual 减语言 prior 已有；它操作 reconstruction 路径，本方向检验真实 video 证据改变时的 existence 响应，而非只校正 masked word 预测 |

## 2. 深入比较：相似思想应如何继承

CCL 的关键不是仅“删除高分片段”：它用 gradients 选 critical / inessential proposals，分别构造 robust / destructive 变换，比较 aggregate alignment 并约束 proposal 分布。已有机制可作为我们的强对照。与之相比，本设计强调 absence 监督的语义有效性、重复事件和 matched input 损伤；如果最终只实现 feature mask 加 ranking，novelty 应定位为 setting/评价扩展，不能声称新干预思想。[CCL 原文](https://proceedings.neurips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf)

CVA 保留目标及邻近 context，再按 query relevance 选 replacement，说明背景替换的 false-negative 风险已有认真处理。因此本方向不能用“我们考虑 query”作为区别；需要证明 target drop 与 background damage 的比较具有独立 label 支持，并检验 partial drop 仍有 witness 时不错误拒绝。[CVA 原文](https://arxiv.org/html/2603.24934)

HRVTG 最接近 visual pruning；它依赖 frozen grounding teacher 生成 Pseudo-GT，删除后产生 counterfactual 拒绝信号，再进行 test-time 适配。我们可继承文本/视觉互补 probe，但应新增更严格的 label gate 与干预 control，并保持 formal U 完全不适配，检验训练侧机制是否泛化到 natural unseen 事件。区别不是 GRPO 改 BCE，而是监督有效性和验证目标。[HRVTG 原文](https://github.com/CVL-hub/HRVTG/blob/main/ECCV2026_HRVTG.pdf)

CCR 的单模态效应分析与本项目 Cq 现象相连：文本 prior 可以使跨模态目标看似成功。但 reconstruction 效应减去后不能自动证明实际 event absence；本设计需要同 query 换 video 及 query-specific 干预双差来检验。[CCR 原文](https://arxiv.org/html/2308.05648)

## 3. 文献来源和能力边界

CCL 原 PDF 已读取§3 公式；HRVTG 作者仓库内 PDF 已读取 probe construction、reward 与评价，另核对[作者实现说明](https://github.com/CVL-hub/HRVTG)。HRVTG 的 ECCV2026 身份来自作者 PDF/仓库，不另外宣称已独立核验会议录。CVA/CCR 使用 arXiv 全文。本文没有复现这些方法，不能断言它们在本 benchmark 必然失败，也不能用其未报告本项目 AUROC 证明它们无法解决问题。

## 4. 积极发展的 novelty 论证

Counterfactual grounding 和 context invariance 已有扎实基础，本方向拟进一步回答“干预收益是否来自目标事件证据，且标签确实是 whole-video absence”。具体推进是建立 all-occurrence / 必要 context 核验，使用相同算子、长度、边界与 damage 强度的 background control，区分 partial 证据减少与所有 witness 消失，并把训练中的 response constraint 连接到 semantic novelty 下的 natural pooled AUROC、same-query 排序与对称四格。这些新增约束有意义，因为 language prior、input damage 和重复事件都可伪造一个看似成功的 necessity 实验。若 I2 在同样 verified 干预数据上超过 label-only I1，并在 natural videos 与第二算子上改善 conditional existence，方法贡献是可迁移且更可识别的 query-specific evidence 监督；若没有这个消融收益，仍可保留有效负例构造与诊断贡献，而不否定整个方向。

当前分类为 **B：核心思想已有，监督有效性、机制控制与 setting 有实质扩展空间**。若最终形成新 response 机制并证明超出 CCL-style / label-only 监督，可进一步论证 D；在结果前不预授新机制结论。

## 5. 最强 reviewer objection

“这是 CCL/CVA/HRVTG 的拼接，只在合成数据上学会拒绝。”所需回答是同结构 I0/I1/I2、same-operator target/background matched control、cross-operator 验证与未干预 natural AUROC，同时证明 partial repeat positive 保留。若只提高 synthetic refusal 或只改变 threshold，objection 成立，应收窄结论。

另一 objection 是“必要性不等于唯一因果证据”：本文接受这一点。输入变换是模型机制诊断与受控监督，不是从观察数据识别事件发生的因果效应；双差只能排除指定加性 prior 形式，不能证明所有 shortcut 消失。
