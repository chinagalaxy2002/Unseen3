# 事件证据干预：Gate 与受控实验计划

状态：`draft_not_executed`。

> 尚未实现、运行或生成 intervention 数据。机制见 [IDEA](IDEA.md)，共享协议见 [COMMON_PROTOCOL](../COMMON_PROTOCOL.md)。本方向优先级 08，低于无需恢复秒级来源的 readout/配对实验。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. Gate I0：时间来源

未来首先核对 sampling time axis、CLIP/SlowFast stride/offset、padding、末端取整、truncate 与真实可见范围。当前 V5 记录没有逐 clip timestamps；`clip_length=1`不能证明 support。通过条件是原始日志可独立核对，或者另行冻结有记录的近似 index mapping 与误差范围。后者只能主张近似特征层实验，必须做边界扩张/平移敏感性；不得沿用 pending R4 称 gate 通过。

若重抽固定小子集，所有 arms 共享新 features，记录 extractor 版本、timestamps 与 checksum；原 canonical 结果另列不可完全同条件。本文不重抽、不改缓存。

## 2. Gate I1：标签有效性与损伤匹配

未来从 inner_train 正例固定抽样，不看 Novel 成绩挑“删除很有效”的样本。独立检查 query 所有必要条件、事件重复次数、完整 witness 与必要前后 context。一个 GT 不代表唯一事件；删一个 span 不自动 label=0。verified 全删除子集、partial deletion 子集、unknown 分别保存。

为每 target intervention 准备 background intervention，保持算子、长度/clip 数、mask 数、replacement 来源类型、拼接边界数相同，并尽量匹配 motion/feature 能量与上下文伤害。background 是否无关须针对完整 query 验证。语言/场景中含有 query 必要条件的背景不能作为 invariant control。构造 matching 只用 inner_train 统计，不读 formal U。

替换 clip 需检查是否引入事件，captioning/CLIP 相似度可筛候选而不能证实 absence。保留 GT 必要事件只证 positive，没标注的剩余 video 不证 negative。若 label 无法可靠核验，I1 不通过；可保存为 diagnostic stimulus 但不用训练负 BCE 或 necessity。

## 3. 四 innerfold 和选模

A1_action_01、A1_action_02、C1_composition_01、C1_composition_02。干预 source 仅 inner_train；所有原样本保留 main BCE，不以可干预比例重写 loader。验证/Noveldev 的干预 challenge 在训练前独立冻结，不参与梯度。每 fold 只按 inner_seen_val 选 checkpoint，Noveldev 仅在选定 checkpoint 后做有限配置开发，不每 epoch 选模；formal U 不调参不适配。

Inner pilot 使用各 fold 对应的 inner baseline；不存在相应 strict inner checkpoint 的 backbone 不直接借正式 canonical checkpoint 充当未见语义模型。正式 canonical checkpoints 只做 Seen 诊断，Flash/QD 严格 inner 验证需后续另建。干预 teacher 或支持挖掘若用任务训练模型，也遵守同一 semantic boundary。

canonical localization 冻结，original input 的 raw spans/class outputs 保持；干预输入的 localization 可作为分析但不是部署结果。不要通过 negative 来源、GT mask、edit tag、是否 intervention 作为 scorer 输入泄漏 labels。若网络必须带 valid mask，所有 conditions 以相同方式处理，不给 target 操作专用标识。

## 4. Gate I2 之后的三臂最小实验

| Arm | 同一 scorer | 辅助监督 | 目的 |
|---|---|---|---|
| I0 natural BCE | 固定小 temporal video-query head | 仅完整 natural main BCE | 无 intervention 基线 |
| I1 label-only interventions | 同 I0 | 相同数量的 verified keep/background positive、all-support-drop negative BCE | 控制额外监督和合成样本 |
| I2 evidence response | 同 I0/I1 | I1 + matched necessity + preservation | 检验响应约束超出增加负例 |

三臂 main 曝光、总 optimizer steps、容量一致；I0 以重复 natural forwards 匹配 auxiliary 计算预算，另报告唯一样本数。I1 与 I2 必须使用完全相同的变换和 labels。初始不加入新的 quartet 训练、reference centering 或 witness architecture，避免把多机制 gain 都归因于干预。

起始 head 可 hidden128、1 交互 block、4heads，auxiliary 数量和 loss weights 用有限候选；这些是未来建议，参数尚未冻结。先 1seed 四 fold，过 gate 后 3seeds 和其他 backbone。估计 1–2GPU 可训练小 head 不等于保证耗时；label preparation 另计。

## 5. 评价矩阵

主结果必须是在未干预原始 video 上的 natural pooled Seen/Novel AUROC。补充 original 同 video PairAcc、真实同 query 换 video、严格对称 2×2、counterfactual queries、query-only、video-only、shuffled-video、hard positive 及 U+false rejection。shuffled 控制重算交互，旧 labels 仅做 input dependence；只有核验新对应关系才计算准确率。

干预机制指标另列：

1. \(\delta_t-\delta_b\) 在已核验全删除与保留背景条件上的分布。
2. 对已核验不受操作影响的 control query，双差\(\eta\)；防止 video-only 损伤解释。
3. partial 删除但另有事件时的保留率，阻止“一删即拒”。
4. 保留完整事件、换背景的 positive 保留率，以及 hard-positive text paraphrase。
5. 相同数量/能量不同位置 damage、边界数 matchedcontrol 和第二算子泛化。
6. 时间 mapping±index、support 扩张及 context 扩张敏感性。

任何“删掉 GT 导致 score 降低”都不能单独证明真实存在性提高。为 target/control 分别报告复核数、coverage、unknown、重复事件和 query 必要 context，防止只展示成功 subset。统计按原始 source video 聚类并配对抽样；多个变换与 query 共享视频不会形成更多独立 n。

## 6. 决策标准

建议整体 gate 为 Novel macro≥+2pp、至少 3/4 fold 为正、Seen 代价≤1pp，且 conditional 与四格改善。另要求 I2 超过 I1 的 evidence-response 效应与 natural 指标，第二算子上方向保持，partial-repeat 不发生大幅错误拒绝。数值在启动前冻结，不是结果。

Action/composition 两类 macro 均需正向；小样本与共享干预的区间不支持结论时，先记录证据不足。

| 观察 | 允许的解释 | 下一步 |
|---|---|---|
| I1 有效、I2 无额外收益 | 更可靠负例/同 query 监督有效，response objective 未增益 | 优先简化到 label-only，不能包装为新 necessity 机制 |
| I2 只改善 synthetic rejection | 学会算子或损伤检测的可能性高 | 停止 natural grounding 贡献主张，换算子核验 |
| target/background 都降、\(\eta\)弱 | 可能是 generic damage / eventness | 不把这个当 query-conditioned evidence |
| partial 删除导致拒绝 | occurrence 覆盖或 label 策略有问题 | 回查重复事件，限制全视频 negative 监督 |
| natural 与 conditional 均好、I2>I1 | 支持 matched evidence-response 监督 | 冻结后多 seed/多 backbone 确认 |
| mapping 敏感、扩 context 后翻转 | 精确必要证据结论不可信 | 修复时间来源或收窄结论 |

最强 confound 是标签真伪，不能通过更多 GPU 解决。gate 未通过时应保留 pending，不制造可运行命令或声称完成实验。

## 7. 后续可交付物

未来的 time provenance manifest、transformation/label audit、all-occurrence 核验、matched damage 审计、configuration freeze、完整 logits、原 video 指标、intervention 指标和失败案例。当前仅本目录三份 Markdown。
