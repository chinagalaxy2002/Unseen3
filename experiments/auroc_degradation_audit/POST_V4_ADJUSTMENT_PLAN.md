# V4 后的项目审查与调整建议

日期：2026-10-03。范围：当前 main 工作区的实现、实验记录、release annotations 和已有 V4 predictions。工作区存在未跟踪文件；V3 五 split 汇总也是本地未跟踪文件，本文不将其描述成已提交结果。本次只增加本报告，没有修改训练或评测代码，没有启动训练。

## 判断

下一阶段优先研究“局部 query-event evidence 如何被读取和监督”。现有实验尚不能区分：信息已经在 max pooling 时丢失、存在于冻结中间表示但未被读出、以及表示本身不能泛化。建议依次推进：评测口径修复 → Seen 内语义留出开发协议 → 冻结表示的读取对照 → 同视频反事实局部 verifier。暂不启动又一轮大规模联合训练。

## 已确认的发现

### 1. V4 的结论受输入瓶颈限制

`training/moment_detr_gmr_auc_v4/common.py:115` 捕获最终 decoder states，然后逐维 max pool。`models/moment_detr_gmr_auc_v4/residual_adapter.py:7` 只读取 pooled vector 与 baseline logit，使用 hidden=64、bound=2 的 residual MLP。它不是只读取 scalar 的校准器，但也无法读取未池化 slots、query tokens 或候选区间的局部视频证据。

因此，V4 支持“这套受约束 pooled-state adapter 收益极小”，不支持“所有冻结表示读取方法均无效”。逐维 max pool 会丢失哪个 slot 提供哪个维度的信息；这是一项明确的结构限制，其对泛化失败的因果贡献仍需实验。

### 2. 数据已有对应关系，V4 没有利用

逐 split 读取 release `train.jsonl`，按 `source_qid` 关联，五个 split 各有 1500 条 S−，其中各 1499 条能对应到训练内、同视频、exist_label=1 的源正例。剩余一条源 qid 是 `train5276`；不要为了补配对直接导入未检查语义资格的样本。该负例仍可进入全量 BCE。

V4 `rows_to_bank` 没有保存 vid/source_qid/construction_type/source_windows；`PairSampler` 只按 exact query → composition → action 回退。其主要约束是跨样本语义分组排序，没有直接使用源正例与最小编辑负例的对应关系。

负例的 release 状态是 `user_attested_video_review`；本次仅核查元数据关联，没有独立观看视频核验不存在性。

训练 S− 构成：

| Splits | object_counterfactual | action_counterfactual | composition_counterfactual |
|---|---:|---:|---:|
| A1/A2_alt/A3（各自） | 1023 | 450 | 27 |
| C1/C2_alt（各自） | 584 | 796 | 120 |

历史 U− 的类型也随 split 明显变化，例如 A1 有 964/1119 条动作替换负例。这是已检查测试集上的探索性发现，不能按这些比例反向定制训练。未来应固定报告负例类型分层结果，避免把构造类型差异完全解释成语义新颖性效应。

### 3. 旧 phrase loss 有未经验证的跨样本负例假设

`models/moment_detr_trm_gmr_joint/joint_loss.py:175` 用 batch 循环移位构造负视频/负 phrase，没有确认另一视频不含该 phrase 描述的事件。代码确认了这一假设存在，尚未量化 false-negative 比例及其影响。新 verifier 不应直接继承这条路径；优先使用已有可信整句负例，且不能把整句不存在解释成每个未修改 phrase 均不存在。

### 4. V3 已完成五 split

当前 `experiments/trm_gmr_joint_v3/MULTI_SPLIT_RESULT.md` 记录 5/5 完成，各训练 50 epochs；全部采用 fallback_seen_mAP，全部未通过 localization floor。五 split macro Seen AUROC=0.712389，Unseen AUROC=0.528708。此前四 split 的摘要应标记为阶段快照。

### 5. V4 localization 选 top-1 的口径不一致

原 evaluator `training/moment_detr_gmr/evaluate.py:65` 用 `softmax(class_logits)[...,0]` 排序。V4 的 `training/moment_detr_gmr_auc_v4/metrics.py:39` 用 `class_logits[:,0]` 排序。二分类 softmax 的排序等价于前景与背景 logit 差值排序，不等价于仅按前景 logit 排序。

用已保存的 V4 predictions 重算，仅替换排序、保持区间不变，得到以下 raw R1@0.5（百分比）：

| Split | S+：当前 logit0 → softmax 排序 | U+：当前 logit0 → softmax 排序 |
|---|---:|---:|
| A1 | 36.65 → 36.97 | 23.01 → 23.01 |
| A2_alt | 35.63 → 35.42 | 26.79 → 26.79 |
| A3 | 33.98 → 34.27 | 38.02 → 39.58 |
| C1 | 34.69 → 35.73 | 48.77 → 48.77 |
| C2_alt | 32.89 → 36.89 | 25.22 → 35.65 |

这些是排序修正诊断值，并非完整官方后处理指标；仍需统一裁剪、取整与窗口处理。这不改变 existence AUROC，不改变 baseline 与 V4 原始定位输出相同的事实。当前 infer 将同一次 baseline forward 的 spans/logits 同时写入 base/v4 字段，相同属于结构保证；这也不等价于独立复现了历史 evaluator 的所有指标。

### 6. 统计区间需要考虑共享视频

V4 `metrics.py:21` 对正负 rows 分层重采样，两模型共用 row 索引。它是模型比较的 paired bootstrap，但不是按视频/源样本聚类的 bootstrap。同一视频多个 query、同一 source 的派生负例具有依赖关系。下一轮以 vid 为 cluster，重采样时保留该视频全部相关 rows，并为两模型使用相同抽样；单类别 replicate 的处理应预先固定。五 split 若共享视频，macro interval 也需保留跨 split 的视频依赖。训练 seed 的不确定性需另外估计。

## 开发协议

从每个正式 split 的 Seen semantics 中构造 action 留出与 composition 留出的内部开发任务；按视频/源样本组防止衍生样本穿越分区。

关键：如果内层留出语义已经被 canonical baseline 用于任务训练，就不能称为 pseudo-unseen。用于这类开发的 baseline/teacher 必须从任务微调前初始化，重新在内层训练子集训练，排除留出语义的任务标签。通用预训练暴露与任务训练暴露应分开说明。若暂不付出重训成本，只能将其称为 Seen subgroup validation。

仅用内部开发任务确定容量、正则、配对比例、目标权重与选模规则；统一应用于五正式 split。已查看的 U 保持 exploratory，未来确认性结论需新的未参与开发的语义划分/数据。各 split 的任务模型独立，避免跨 split 共享会暴露正式留出语义的任务监督。

## 实验顺序与判断分支

### 阶段一：表示读取诊断

保留 canonical baseline，先比较以下 Seen 训练的 readout，尽量控制训练预算和参数量，额外报告容量差异：

| 对照 | 输入 | 要回答的问题 |
|---|---|---|
| V4 reference | pooled state + s0 | 已有受约束修正能做到什么 |
| pooled readout | pooled state，较少约束的 head | bound/anchor/readout 容量是否限制了 V4 |
| slot readout | 全部 frozen decoder slots + query tokens | pooling 前是否有更有用的信息 |
| local verifier | frozen 原始视频序列 + query tokens + 固定候选区间 | 局部视觉证据能否补足 decoder 读取 |

使用相同的基础监督先比较输入，再单独加入 source-linked 配对。不要一次改变输入、采样、loss 和训练骨干。

额外控制：query-only、video-only；同视频正负 query 的 score 差；已验证同 query 不同视频的 score 差；source GT 区间作为诊断用固定区间，与预测候选区间比较。GT 区间必须同时用于该 source 正 query 与对应负 query，不能只给正例提供 oracle 信息。推理不使用 GT 或源正 query。

若只有 slot readout 有收益，优先改 pooling/读取；若原始局部特征 verifier 有收益，优先训练证据分支；若两者在内部语义留出均无稳定收益，才考虑更强时序特征或更大范围表示更新。GT 支持的局部 verifier 仍失败时，检查标签、动作/状态信息和预计算特征的能力边界。

### 阶段二：V5 候选——固定定位的局部证据 verifier

冻结原定位模型并维持 eval，按其固定候选区间从 CLIP/SlowFast 序列中抽取带时间顺序的局部特征，结合 query tokens 训练小型交互模块。候选生成保持不变，候选内证据变换可学习。这可保持原始定位输出，同时允许新分支学习表示。

对每个候选先计算完整 query 的兼容性，再固定规则汇总候选证据。不能分别在不同候选寻找 action 和 object 的最大值后直接拼成事件存在；二者需要在同一事件支持区间内具有联合证据。动作不必在单帧可见，应保留区间内时序。

使用已关联且元数据检查通过的 `(video, source_positive_query, counterfactual_negative_query)`，对同一组候选施加正负兼容性差异监督。源 GT moment 用于训练中的局部证据约束，负 query 的全视频不存在性仍依赖原 release 标签。对象替换不应导致所有共享动作 phrase 都被当成负例。

初始目标保持简单：全量自然分布 BCE + 一个 source-linked pair loss。主 loader 每 epoch 保留全部 Seen rows；独立 pair stream 不决定主数据覆盖。global AUC、anchor 和其他正则逐项消融，只在内层开发集有证据时加入，不默认继承 V4 的四项组合。辅助采样可在 Seen 内按 edit type 诊断/平衡，不能根据历史 U 类型比例定权重。

新 evidence score 与 baseline residual 融合应是对照选项；不要先把新分支限定为 bound=2 的小修正。若更新共享 transformer，即使冻结 span/class heads，定位输出也可能变化；需重新验证，不能继续声明 localization invariant。

### 阶段三：进入五 split 的条件

先在内部语义留出任务证明改进可重复，再冻结配置跑正式五 split。通过条件在查看正式 U 前写明：主 AUROC 最小有意义改进、允许的 Seen 退化、定位约束及计算预算。阈值数值由研究用途和内层实验确定，不能事后围绕 U 点估计设置。

最终至少报告逐 split pooled AUROC、macro、source-matched PairAcc 及其样本量、Seen AUROC、统一口径 raw localization、官方 soft gate 指标、负例类型分层、视频聚类区间。对最终有希望的方案进行多个独立训练 seed 复核，并明确是否只变化 verifier seed，还是重训 baseline seed。

## 文献定位边界

SHINE（ECCV 2024）已研究语义合理 hard negative queries 和 coarse-to-fine saliency ranking；因此“加入细粒度负例”本身不足以支持新颖性。可核验来源：[论文摘要与作者页面](https://arxiv.org/abs/2407.05118)。这是定向检索，不是完整新颖性审查。

本项目可进一步验证的特定问题是：在语义新颖性下，局部证据的读取与监督能否同时改善 null-set discrimination 和跨 query pooled ranking，并保持定位能力。当前尚无 V5 实验结果，不应预先宣称方法有效或具有新颖性。

## 首批实际工作

1. 统一 localization scorer、单位与后处理；单独保留历史结果，并更新根 README 的版本索引。
2. 导出 Seen-only source-pair manifest，包含 vid/source_qid/edit type、语义资格和源样本缺失报告；扩展 feature bank 保留未池化 slots、query/局部视频特征及 masks。
3. 冻结内层语义留出协议，执行 pooled → slots → local evidence 的读取对照；依据结果选择 V5，而非立即重复五 split 联合训练。
