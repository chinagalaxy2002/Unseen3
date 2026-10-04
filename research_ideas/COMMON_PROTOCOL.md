# 共用证据、实验与评价协议

状态：`draft_not_executed`。本文件定义未来实验的设计，不执行实验。证据版本：main `6cd96d723806e4b2b2474c68f289921eb543ffcc`。八个方向遵循同一套目标、数据边界和评价定义，避免通过更换训练分布或指标解释产生表面提升。

## 1. 任务与证据强度

标签 `Y(V,Q)` 判断的是整个视频中是否存在满足 query 的事件。S+/S- 使用训练期 Seen semantics，U+/U- 使用 task-level held-out semantics。U- 不表示视频没有动作。预训练 CLIP 或其他编码器可能见过相关语义，因此只能声称 task-level semantic holdout，不能声称对任何预训练均完全未见。

| 历史 baseline | Seen AUROC | Unseen AUROC | Gap |
|---|---:|---:|---:|
| Moment-DETR-GMR | 0.7518 | 0.5287 | 0.2231 |
| FlashVTG-GMR | 0.7550 | 0.5479 | 0.2071 |
| QD-DETR-GMR | 0.7476 | 0.5144 | 0.2332 |

这组数值来自用户总结及项目此前审计，五个 split 均 Seen > Unseen；仍属单 seed、共享 benchmark 的观察。三个 backbone 共享数据/标签、source-distribution existence 判别及跨 query scalar ranking，不能假设共享完全相同的 pooling 或 decoder。

### V1–V5 可沿用的约束

| 实验与来源 | 已有证据 | 对新方案的约束 |
|---|---|---|
| [V1](../experiments/trm_gmr_joint_v1/MULTI_SPLIT_RESULT.md) | localization / phrase correspondence 增强没有稳定 existence 收益 | 不把定位提升作为存在性提升的充分条件 |
| [V2](../experiments/trm_gmr_joint_v2/CURRENT_AUROC_RESULTS.md) | generic visual residual 收益不稳定；文档为完成 2/5 的阶段快照 | 需要 query-conditioned evidence；不写成五个 split 的终局反证 |
| [V3](../experiments/trm_gmr_joint_v3/MULTI_SPLIT_RESULT.md)、[审计](../experiments/auroc_degradation_audit/AUDIT_AND_ADJUSTMENT_PLAN.md) | sampler 丢失约 14.71%–45.09% S+，全部 split localization floor 失败 | 保留自然主训练流；within-semantic-group 与 exact-query conditional ranking 分开 |
| [V4](../experiments/moment_detr_gmr_auc_v4/MULTI_SPLIT_RESULT.md) | pooled-state residual correction macro ΔU≈0.00039，各 CI 含 0 | 不能继续将同类 post-hoc residual 当主要创新，也不能推断所有冻结表示无信息 |
| [V5](../experiments/moment_detr_gmr_evidence_v5/P3_READOUT_RESULTS.md)、[JSON](../experiments/moment_detr_gmr_evidence_v5/P3_READOUT_RESULTS.json) | R1≈−1.42 pp、等容量≈−1.49 pp、R2≈−0.11 pp、R3≈+1.18 pp；均无稳定 Novel 提升 | pooling bottleneck 未获支持；全 slots 不是默认解法 |
| V5 Cq，同上 | pooled Novel Δ≈+3.16 pp，4/4；source-pair PairAcc 四折均下降，macro≈−12.89 pp | 首先区分 language predictive signal 与真正条件兼容性；这仍不能直接证明 benchmark shortcut |

V5 上述 Novel 是 Seen-only inner holdout 的 Novel-dev，不是正式 U。V5 Cv 的 same-video PairAcc 必为 0.5，但个别 fold 的 pooled Novel score 可较好，提示 video marginal 也需要控制。所有历史现象只构成假设依据。

### 根因假设与最低成本证伪路线

下面区分支持、反对和缺失证据；最低成本实验均为未来计划，尚未执行。优先级依据可识别性而非复杂度。

| 假设 | 已有支持 | 反对或限制 | 缺失证据 / 最低成本证伪 |
|---|---|---|---|
| H1：语言边际与标签相关 | Cq pooled Novel 四折获益 | 没证明三个 multimodal baseline 主要用它 | 同编码 query 双标签支持 + Cq 平衡评价；04 同支持配平对照 |
| H2：negative construction 与 shift 的难度组成影响退化 | source 编辑来源、负例类型组成不同 | 类型差异本身不证明捷径 | 预固定 edit-type、自然语言和同场景 strata；同 q 换视频核验 |
| H3：absence/编辑标签存在噪声 | 未完全标注不能推出事件不存在 | 已有 review provenance；不能预设标签普遍不可靠 | 盲抽样重新核验全视频，包括重复事件，按来源报一致率 |
| H4：绝对 BCE 缺少双条件识别约束 | Cq 与 PairAcc 分离、三模型共享判别设置 | V3 ranking 组合失败，但 sampler 混杂 | 01 同 exposure 的 BCE / row / row+column 对照 |
| H5：完整事件绑定不足 | V1 correspondence 不保证 existence、V2 eventness 不稳 | 无稳定 pooling 证据也无现成 binding 反证 | 03 同 negatives 的普通整句 / 独立 atoms / 联合 witness |
| H6：existence 表示或定位耦合限制信息 | V4/V5 readout 失败；共同 checkpoint/优化可能影响 | Flash 的路径不同，不能归因统一 decoder pooling | 05 pooled/slots/raw-sequence 路径比较，raw 与 gated localization 分报 |
| H7：语义/场景关系 source-specific | 三模型五 split 都有 gap | 单 seed、共享 benchmark；可能是 feature 本身不足 | 07 同 pairs/compute 的 ERM 与跨环境风险训练 |
| H8：query/video-dependent offset 破坏 pooled ranking | Cq 稳定 pooled 增益与条件下降 | V4 该 pooled residual 无收益；不能排除合法 priors | 02 raw/posthoc/centered-trained；同 scorer D 不变性控制 |
| H9：候选信号存在但聚合受冗余/长度影响 | 候选与视频 scalar 的目标不同 | V5 R3 对 pooling 的稳定支持不足 | 06 固定 logits 的 copy/length sensitivity，再分离监督×聚合 |
| H10：分数需要真正 relational / counterfactual evidence | pooled/paired 冲突要求更强机制验证 | PairAcc 和一般 interaction 都仍可走捷径 | 01 闭合矩阵，08 标签可信的目标/背景损伤双差；后者先过时间 gate |

Calibration 是贯穿 H8 的对照：global monotonic mapping 不能改善 AUC；conditional normalization 可改变 pooled 排序，却不能自动证明 grounding。Benchmark 语言捷径是 H1/H2 的待验证解释，counterfactual 监督是 H4/H10 的候选机制，不把前者预先当成已证实缺陷。

## 2. 数据权限与 split 边界

1. 首轮只使用现有四个 inner folds：`A1_action_01`、`A1_action_02`、`C1_composition_01`、`C1_composition_02`。定义见 [inner_fold_index.json](../experiments/moment_detr_gmr_evidence_v5/inner_fold_index.json)。不新建与结果最相符的 holdout。
2. 梯度、reference bank、negative mining 和人工新增训练标签均来自该 fold 的 inner train。主数据流保留原始自然比例及全部可训练行；auxiliary pairs/quartets 独立采样，不把不能组成 pair 的行删除。
3. Epoch selection 默认沿用 inner Seen-val pooled AUROC，平局选最早；结构、损失和超参数可以根据已完成的 inner Novel-dev 结果进行研究开发，必须计入开发次数。Novel-dev 不进梯度、不选 epoch，且不能再包装成 untouched test。
4. 已有 V5 inner baseline 从排除相应语义的数据上训练，适用于 inner Novel 分析。不能加载见过 inner holdout 的 canonical baseline，再把 readout 当作 pseudo-Unseen 验证。三个 canonical backbone 可以做 Seen-val 诊断；Flash/QD 的严格 inner Novel pilot 若要做，需另行建立相应模型。
5. 正式 U 仅在方案、数据、代码版本、超参数和比较表冻结后评价。历史 U 已被研究人员读过，正式复测应明确其 exploratory 属性；若要强 confirmatory 声称，增加未参与开发的预先固定 split 或独立数据。
6. `source_qid`、编辑类型、semantic-group ID、审查 provenance 只用于配对、审计和分组，不作为模型输入。过滤 auxiliary 数据时同时检查自然 query、反事实 query 及编辑引入的语义，不能从 formal U 获得额外监督。

### 配对有效性

“没有 GT caption”“没有标注这个动作”“视频/句子相似度低”均不能证明事件 absent。新 cross-video negative 需要已有可靠 absence label 或另行人工核验；uncertain 单独保存，不强制转为 0。保留 exact text、编码输入与特征 hash，不能把 paraphrase 当 exact query。大小写和空白规范化能支持初筛，但最终同 query 必须确认实际编码输入等价。

已有 [source_pair_audit.json](../experiments/moment_detr_gmr_evidence_v5/source_pair_audit.json) 记录大多数 formal train source pairs 可配对；缺失来源的行继续进入主 BCE，不从其他 split 补入。发布信息中的 `user_attested_video_review` 是已有审核来源记录，不等于本轮独立重新观看视频，也不能无证据认定其无效。

### 未来 phase 0：只在实施阶段做的最低成本可识别性审计

建立 exact-query / video / label incidence graph，统计每 query 两类标签支持、每 source pair 的来源、可闭合 2×2 矩阵、独立视频数、semantic/negative-type 覆盖。去重时一个 quartet 的交换行列不产生新样本。查明 train/val video 重复及 provenance，报告对估计的影响。四格不足时先用已核验 column pairs 或少量人工扩充，不能自动补齐交叉格。该审计的操作与结果目前均未实施。

## 3. 必须同时报告的六类评价

### 3.1 Pooled AUROC

主指标在原始自然分布全部样本上计算；inner development 分别报 Seen / Novel，正式评价分别报 Seen / Unseen、每 split 与 macro。不能用 balanced subset 的 AUC 替换主指标。记录正负数、唯一 video/query 数及 full-precision score。AUROC 衡量排序，单个共享正温度或全局单调 calibration 不能改变它。

### 3.2 Same-video PairAcc

对可靠 source pairs，`P[s(V,Q+) > s(V,Q-)]`，平局计 0.5；按视频聚类。同视频固定了 video，却没有固定 query。因此 Cq 可以高于 0.5，此指标单独不能证明使用视觉。

### 3.3 Same-query ranking

使用完全相同的 Q，在确认 present 的 V+ 与 absent 的 V- 上比较：`P[s(V+,Q)>s(V-,Q)]`。报 query-macro PairAcc、支持 query 数及按 query 的 AUC（仅双类支持者）；它们不同于全局 pooled AUROC。Query-only 确定性模型在相同编码输入下应为 0.5。仍需控制 video-only，它可能利用场景/演员先验。

### 3.4 闭合四格评价

核验 `Y11=Y22=1, Y12=Y21=0`。四个有向 margin 分别固定行或列：

`d = (s11−s12, s22−s21, s11−s21, s22−s12)`。

报 row PairAcc、column PairAcc、四 margin 全部正确的 strict group accuracy，以及 `D=s11+s22−s12−s21` 的分布。`D>0` 不等价于四个 margin 都正确；strict group accuracy 没有通用的 0.5 随机基线。任何 `a(Q)+b(V)` 满足 D=0，也无法使四个严格 margin 同时为正。对完全对称加权的 quartet 集，纯 query 或纯 video 的正负 score 多重集相同，pooled AUC=0.5；这一结论不直接推广到任意不对称子集或所有可加函数的 pooled AUC。

### 3.5 单模态对照

使用相同 split、epoch rule 和相近有效参数量分别训练 Cq、Cv，不只删除输入后测试一个从未见过缺失输入的模型。对照可复用已明确版本的 V5 Cq/Cv，同时为新 scorer 提供配套单模态版本。历史 score replay 对齐到同一支持子集，不能把样本组成变化当成性能变化。

### 3.6 Shuffled-video 与反事实 query controls

在 encoder/fusion **之前**替换视频并重新计算 score；不能复用原 query 条件下的 decoder slots 假装换 query/video。报告错配前后 score、排名稳定性及 shuffled AUC。对原标签评价错配输入只是依赖性诊断，不是替换后事件标签的准确率；不能强求其 AUC=0.5，因为 video/question 分布仍可能有标签相关性。

用已核验的自然 query / 编辑 query，以及同 query 在不同视频上的标签翻转形成反事实条件。尽量固定 actor、scene、object，并引入自然表达 negative；不能仅靠改变词序或病句识别完成任务。即使满足四格，也可能仍利用 scene×query interaction，所以机制声明必须与输入干预、hard natural negatives 对照一致。

## 4. 公平比较与选择规则

Frozen-backbone pilot 优先：主 comparator 使用完全相同的 feature bank、head family、训练行、epoch 上限和选择规则。Auxiliary arms 对齐额外 forward 数、四格覆盖与参数；必要时加相同数据仅 BCE 的 exposure control。不要把新增样本、额外预算和机制一起变化。

保留旧 baseline score，另报新 head 真实结果与 fallback 后 policy 结果。独立随机初始化 head 的 epoch 0 不是 canonical baseline；不能用其参数位置假装 identity fallback。增量模型若要 epoch 0 等于 baseline，需显式证明 residual 初始化及 forward 公式。

Canonical foreground 使用项目定义的正确 softmax 类别；不要混用 logit 的某个位置与概率。历史四位小数 score 可产生更多 ties，新实验用未舍入 float score；报告由于精度重算产生的差异。

Frozen proposal head 的 raw localization 可保持相同，但 existence threshold 会改变 gated localization，二者分别报告。Threshold 仅由 Seen-val 决定，不能在 U 上调；joint fine-tuning 时独立报 Seen MR mAP / R@1 等项目既有指标，不以 Seen AUROC 下跌换 gap 缩小。

## 5. 统计与建议推进门槛

一至两张 GPU 指并发资源需求，不承诺单次总时长；时间/显存由未来 pilot 测量。首轮用 seed 3407 便于与 V5 对齐，候选确认至少追加两个 seed；重复 seed 不能当作新增视频样本。

Pooled AUC 用 video-cluster paired bootstrap，比较方法时同步重采样相同样本。Source-pair 按共同视频聚类。Quartet 可能共享两个视频和 query，不能独立 bootstrap 每个 quartet；先报告 incidence graph 连通分量和有效支持，再采用可成立的分量聚类或显式 two-way/multiway sensitivity。若形成一个巨大分量，不能给出虚假的独立 quartet CI，应报告描述统计及限制。五个 splits 共享数据时 macro CI 也不能假设它们完全独立。

建议的下一阶段推进标准（尚未冻结）：

- 主指标：inner Novel macro ΔAUROC≥2 pp，至少 3/4 folds 正向，action / composition 两类 macro 均正向。
- Source 保持：Seen macro 下降不超过 1 pp；backbone 联训还需满足预先固定 localization floor。
- 机制支持：same-query 与 quartet/strict group 指标同向改善，并优于配套 Cq/Cv；可把条件 PairAcc +3 pp 作为优先推进信号，但小样本时先看 CI、有效支持与方向一致性。
- 确认：多 seed 与成对 CI 支撑方向；再将最优一至两个方法移植三个 backbone，不对八个 ideas 全部先跑全规模。

这些数值是资源分配建议，不是统计显著性定义。小 subset 的 CI 宽时结论应是证据不足，不能宣告根因成立或已被排除。

## 6. 不同结果意味着什么

| 观察 | 可以提出的结论 | 仍不能推出 |
|---|---|---|
| Pooled 提升，Cq 相似提升，条件指标不升 | 边际判别/分数可比性可能改善 | 真正事件证据增强 |
| 条件指标升、pooled 不升 | 条件兼容性更好，但跨 query scale/offset 或剩余数据组成仍妨碍主 AUROC | 已解决主 benchmark |
| 对称四格 query-only 仍高于 0.5 | 检查不对称权重、不同编码、漏标签或划分泄漏 | 四格监督理论失效 |
| Frozen pooled head 无效、原始序列 adapter 有效 | 结果支持表示或 backbone-conditioned readout 限制 | 任意 pooling 都是根因 |
| 所有小 head 无效 | 当前 features、标签支持或优化可能不足 | 所有 video-query 方法都无效 |
| Seen 明显下降、gap 变小 | Source discrimination 被破坏 | Novelty robustness 提升 |

统一将“机制效果”“数据效果”“预算效果”拆开讨论。若方法失败，排除的是对应版本及实验支持范围内的假设，而非整个研究思想。
