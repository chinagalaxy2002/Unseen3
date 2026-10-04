# 完整事件 Witness：最小验证计划

状态：`draft_not_executed`。

> 仅计划，尚未实现或运行。机制见 [IDEA](IDEA.md)，共用限制见 [COMMON_PROTOCOL](../COMMON_PROTOCOL.md)。优先级 03 是候选研究顺序；先前 crossed audit 若没有发现细粒度错绑定失败，不应无条件启动大模型。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 先判断这个实验是否有可识别的目标

在未来准备阶段，固定从 Seen 训练语义中抽样、独立复核的 challenge：action/object 都出现但关系错误；同动作不同参与者；同参与者不同对象；保持意义的 hard positives。先做 action–object，后两类只在视觉可观测时加入。模型不得读取 reviewer 文字、编辑记录或 labels。对全视频复核后仍不确定的 negatives 标为 unknown，仅进入诊断说明，不进入负监督。

核验目标不是“这句没有 GT”，而是“这个视频里没有满足整句的实例”。一个 query 可能在多处成立；所有成立区间不完整时，不能借缺少某个 span 宣称 absence。原始 natural evaluation 不因 challenge 清理而重建或删除难例；新 challenge 是补充机制证据。

## 2. 数据和选模协议

四个开发 folds 为 A1_action_01、A1_action_02、C1_composition_01、C1_composition_02。训练只使用各自 inner_train；所有 S+/S− 保留 main BCE。source-linked pairs、hard positives、binding challenge 是 auxiliary stream。语义划分及规范化规则继承已有 freeze；新 paraphrases 不能跨 inner semantic boundary，引入训练集中未出现的新动作作为“正例改写”。

Inner pilot 的冻结 task checkpoint 使用对应 fold 的已有 inner baseline，不能把已见过 inner Novel 语义的正式 canonical checkpoint 当作严格 holdout teacher。三个正式 canonical backbone 只用于 Seen 诊断；Flash/QD 的严格 inner Novel 扩展需要另建相应 inner baseline。原始预训练 feature 的 task-level 未见边界与预训练暴露边界分开说明。

checkpoint 仅按预先固定的 inner_seen_val existence AUROC 选取；Seen/localization guards 同时记录，tie-breaking 预先固定。Novel-dev 在每个完整 run 选定 checkpoint 后评价，仅用于有限配置开发；不得每 epoch 评价后选最大 Novel checkpoint。开发次数和历史可见性登记。正式 U 只在方案冻结后一次性评测，不调 loss、parser、阈值、bank 或窗口。历史五 split 用于提出方向，应如实声明 exploratory。

canonical localization 完全冻结，dropout/eval mode 和 raw span outputs 保持一致；existence readout 的改变可影响拒绝门控后的 end-to-end 指标，应与 raw localization 分开报告。

## 3. 最小三臂：先隔离完整事件绑定

| Arm | 输入/候选 | 打分 | 目的 |
|---|---|---|---|
| W0 full-query head | 同一视频序列、多尺度 index windows、query tokens | 普通整句 compatibility + MIL，无显式 primitive/argument 路由 | 控制读原始序列和容量变化 |
| W1 independent atoms | 同 W0 | 各 atom 在全候选独立取高值后组合 | 直接检验跨时间共现 shortcut |
| W2 joint witness | 同 W0 | 同候选内 full-query 与 action–argument 联合验证，再 MIL | 检验联合满足性，避免把同槽当成功证据 |

三臂完全相同 BCE、source pairs、hard positives、binding negatives、曝光次数和 optimizer budget。若 W2 有额外 parameters，W0 添加 capacity-matched 中性层，不改输入。报告 trainable parameters、forward FLOPs、候选数和显存，不能仅用大致 hidden size 声称容量相等。

初始 W2 可只有 \(j_k\) 联合 score，不加 atom soft conjunction；只有存在 gain 才单独比较加 conjunction。所有臂均使用共同无秒级 GT 的 index windows，窗口内有时间顺序；全视频长候选比例单独报告，防止宽候选退化为同视频 atom 共现。候选提取器/采样策略在臂间共享且不依赖正负 source identity。

## 4. 建议的有限开发预算

小 head 的起始建议为 hidden 128、1 temporal/cross-attention block、4 heads、dropout 0.1，窗口采用固定 feature-index 尺度。学习率、epoch 上限先参照 V5 readout 的可运行预算，启动前再根据输入长度冻结；本文不给未经核验的运行命令。候选温度/辅助权重最多预设两个值，不能每 fold 独立大规模搜索。这些是未来参数建议，不是已有配置或结果。

先 1 seed 完成四 folds × 三 arms；资源不足可先 A1/C1 各一 fold 只检查训练可行性，但不能据此宣布泛化成功。通过机制与指标 gate 后再 3 seeds、五正式 split 和其余两 backbone。三 backbone 复现需要相同 feature/候选协议，另列 canonical baseline 差异，不把 backbone-specific feature gain 归给 witness。

## 5. 必报评价

- 原始 natural pooled Seen/Novel AUROC、每 fold Δ 与 macro Δ。
- Matched same-video source PairAcc，按 pair coverage、edit type、语言 plausibility 分层。
- Same-query present/absent video PairAcc/AUROC；只用核验交叉标签，不假设任意换视频就 absent。
- 完整 2×2 quartets：行、列、全部正确率与 interaction contrast；严格对称加权，报告独立视频数量和 pair 重用。
- Cq query-only、Cv video-only；query-only 在 same-video PairAcc 高于 0.5 不视为视频能力。
- Shuffled-video：重新跑视频查询交互，不换已经融合的 hidden states；保留旧 labels 的结果仅叫 input-dependence control。可信换视频标签单独测。
- Counterfactual queries、hard positives、完整绑定 challenge；真实 U+ 被拒比例及其 score 分布。
- Candidate coverage、长窗口比例、解析成功率、unknown negatives、重复事件和缺失特征。
- Raw localization identity 与拒绝门控后的指标；不得用 Seen 降低导致 gap 缩小当主要收益。

视频 bootstrap 应配对抽样、保持同视频相关性；quartets 共享视频时用连通分量/聚类敏感性分析，不把所有四格当独立样本。低覆盖 fold 只报估计与区间，不能过度解释；raw score 不截断、不只存概率小数。

## 6. 推进与停止规则

建议门槛：Novel macro ≥ +2 pp、3/4 folds 为正、Seen ≥ −1 pp；same-query 与 complete quartet 均有一致改善，绑定 challenge 上 W2 超过 W0/W1，hard positives 不明显下降。这些门槛须在未来启动前冻结，并报告 CI 是否支持结论，而不是只报超过门槛。

Action 与 composition 两类 macro 也应均为正；小 quartet 子集支持不足时报告证据不足，不用大量共享组合制造精确 CI。

| 结果 | 能得到的结论 | 后续选择 |
|---|---|---|
| W2 > W1，但 W2 ≈ W0 | 跨候选独立 atom 聚合有害；未证实显式绑定价值 | 保留普通 joint scorer，避免过度建模 |
| W2 > W0/W1，natural 和 crossed 都改善 | 支持完整 witness 学习与 Novel existence 的联系 | 做同负例/参数严格复现，扩 backbone |
| 仅 challenge 改善 | 解决局部错绑定，主 score/覆盖仍不足 | 分开研究 aggregation 或 score comparability |
| 仅 pooled 改善，换视频不改善 | 无法归因为真实视频条件证据 | 停止 grounding 贡献主张，回到 prior controls |
| U− 好但 U+ 明显坏 | 过度 conjunction / novelty refusal | 降低结构强制性，研究 false rejection |
| 所有臂无法识别核验简单事件 | 特征能力/可见性是竞争解释 | 不继续扩大 slots 或堆 loss |

## 7. 未来产物，而非当前已生成产物

未来应生成冻结配置、split manifest、解析和 witness 标签审计、模型输入白名单、完整 logits、条件指标、定位 identity checks、失败案例与 novelty 消融。当前目录只有研究文档；无代码、checkpoint、推理缓存或实验效果。
