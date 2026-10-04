# 完整事件 Witness：相关工作与可发展贡献

状态：`draft_not_executed`。

> 文献核对日期：2026-10-04。问题是“已有工作是否已经学习同一完整事件的满足性，并在 semantic novelty 下验证 present/absent ranking”，而非关键词是否重合。以下来自原论文方法与评价章节；HRVTG 同时核对作者 PDF 和实现说明。没有新实验结果。机制见 [IDEA](IDEA.md)。

## 1. 最接近四项工作的逐维比较

| 工作与已读位置 | 输入 / supervision | Score / negative construction / objective | Semantic setting / evaluation | 与本方向的实质关系 |
|---|---|---|---|---|
| [SHINE，ECCV 2024，§3.2–3.4、§4](https://arxiv.org/html/2407.05118) | 视频、整句、GT interval 及由其构造的 pseudo saliency | GPT-3.5 对 verb/noun/adjective/preposition/adverb 分层替换；coarse saliency 排序与分层距离 margin，联合定位 loss | Novel-Composition、Novel-Word；R@IoU/mIoU | 已有多 primitive hard queries 与 compositional 泛化；saliency 分层距离并非同一实例 action–argument witness。保留文本可行性过滤，新增完整事件 joint score 与 whole-video absence 验证 |
| [Generalized Video Moment Retrieval / BCANet，ICLR 2025，§3.4、§4、§5](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf) | 视频、query、subject–predicate–object relations/timestamps，零/多 moments | 修改 S/P/O 和跨视频 query；boundary-aware region contrast、query-region attention、matching/proxy loss；候选 evidence 判空 | NExT-VMR；no-target/multi-target、N/T-acc、定位 R/mAP | 已有关系组合、零目标与局部 evidence，不能宣称首次将关系用于 GMR；需新增可证伪的绑定约束与 semantic-holdout AUROC/条件排序 |
| [Learning to Refuse / RA-RFT，§3–4](https://arxiv.org/html/2511.23151v1) | grounding LVLM、relevant 时间回答、hard-irrelevant 拒绝与解释 | LLM 编辑语义类别；GRPO 奖励 format、refuse-IoU、explain、query correction；输出区间或拒绝 | HI-VTG 与相关 relevance scenarios；relevance F1、RA-IoU 等 | 细粒度编辑、拒绝解释、查询纠正已有。我们的最小模型不以解释文本证明证据绑定，而以同样 negative 下联合与独立证据的对照证明 |
| [HRVTG，作者原论文，§3.2、§4–5](https://github.com/CVL-hub/HRVTG/blob/main/ECCV2026_HRVTG.pdf) | 冻结 grounding MLLM、测试流、teacher pseudo intervals | 文本编辑、Pseudo-GT pruning、hybrid probes；online LoRA/GRPO、positive consistency 与 counterfactual 拒绝奖励；输出区间/∅ | TVGBench-RF；conditional IoU、balanced decision F1 | query/event 不存在的反事实验证已有。本方向只用训练侧可信标签，不适配 U；candidate joint binding 不等于删一个 teacher 区间后默认 absence |

这些方法各有优势。SHINE 的明确 compositional setting 最接近我们的 semantic novelty，BCANet 的 no-target/multi-target 定义最接近存在性，RA-RFT 与 HRVTG 则最直接处理拒绝与微小语义差异。它们不能被一句“不是我们的 AUROC”否定；需要在同数据与相同 feature 上建立可比基线，明确哪些机制可低成本复用。

## 2. 方法细节带来的设计约束

SHINE 并非简单随机换一个词：它构造 progressively altered negatives，并约束 temporal saliency distributions 的层次差异。所以把“LLM 生成多级负例+ranking”当新机制不可防守。可发展的不同点是，语义编辑数目不必与视觉 absence 的难度单调对应；一个必要关系被改动就可使完整事件不成立。Witness 应用核验的 full-query labels，而不按编辑数量假设 evidence 强度层级。[SHINE 原文](https://arxiv.org/html/2407.05118)

BCANet 的数据包含关系元组时间交集形成组合 query，说明 event composition 和局部 evidence 并非空白。我们应沿用“query 可有零/多个 witnesses”的表述，同时避免将单个 local window 中 atom 同时出现当作关系成立。需要能区分论元或事件身份的 joint negative，而非只把独立分数取 min。[BCANet 原文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf)

RA-RFT 与 HRVTG 已经把细粒度拒绝作为训练/适配问题。我们的贡献不能是“新增 refusal head”。可以推进的是监督可识别性：负 query 未变部分不误标负，多个重复事件不因删除一次变成 absence，hard positives 保持可接受；这些约束必须与模型机制共同验证。[RA-RFT](https://arxiv.org/html/2511.23151v1)、[HRVTG 作者原文](https://github.com/CVL-hub/HRVTG/blob/main/ECCV2026_HRVTG.pdf)

## 3. 邻域工作与不可忽略的反证

[The Hard Positive Truth about Vision-Language Compositionality](https://arxiv.org/html/2409.17958) 直接提醒只优化 hard negatives 可能损害保持意义的正改写。因此 witness 的 success 不能只有 U− 拒绝与编辑 challenge；要检查 paraphrase positive 和未见但真实存在的 U+。当前小型 visual feature 很可能缺少角色或动作方向信息，此时增加逻辑约束不会产生新的视觉证据。这个竞争解释应保留。

## 4. 积极发展的 novelty argument

已有工作已经证明细粒度负例、语义组合和拒绝学习有价值；我们拟推进的是它们在 semantic-novelty GMR 中尚未充分识别的“完整事件满足性”问题：当查询 primitives 各自在视频中成立、但指定的 action–argument 或事件关系不成立时，存在性分数应由联合 witness 而非独立共现决定。方法将同候选的有序联合 verifier、编辑条件监督和 hard-positive 保留结合，并用 same-query cross-video 与对称四格控制排除语言/视频边际收益。若在相同 feature、同样 negatives、同预算下，联合 witness 同时改善自然 pooled AUROC 与受控兼容性，新增贡献就是可迁移的事件满足性机制；若只增强 shared-window evidence，则应诚实收窄贡献，但仍可形成对现有细粒度 ranking 的有意义扩展。

当前 novelty 分类为 **B：已有核心思想，setting/mechanism 有实质扩展空间**；“明确新机制”只有在真实实例 binding 与受控消融成立后才可主张。不存在类似论文不是可证明的前提。

## 5. 最强 reviewer objection 与所需证据

Objection：“这是 SHINE/RA-RFT 的 hard negatives 加一个更大的 verifier，提升只是数据质量。”回答应是实验，而非命名：所有 arms 共用 negatives/paraphrases、候选、特征、曝光与容量，比较 full-query joint head、独立 atom maxima、joint witness；没有 binding 优势就不声称绑定贡献。另一 objection 是“同槽不能证明同事件”：应保留多主体/错角色子集，逐步引入可观测 identity 证据，不把 soft-min 数学形式包装成已识别因果关系。
