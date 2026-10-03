# V5 工作方案：语义新颖性下的局部事件证据学习

版本：draft-1，2026-10-03。

这是根据当前项目审查制定的实施方案。阶段任务、数据契约、对照和判断规则已明确；尚未实现新模型、生成内层划分或运行新实验。本方案不构成已冻结的实验记录。正式 freeze 必须补充实际数据清单、代码与 checkpoint 哈希、最终开发配置和计算预算。

审查依据：[V4 后调整报告](../auroc_degradation_audit/POST_V4_ADJUSTMENT_PLAN.md)。V1–V4 的既有 U 结果已影响设计，下一轮五 split 仍为 exploratory。

## 1. 目标与假设

研究目标：在保持原始定位输出的前提下，检验更完整的候选证据读取与同视频反事实监督，能否改善未见语义下的 query-event existence ranking。

| 假设 | 所需对照 | 可以支持的解释 | 不能直接推导的解释 |
|---|---|---|---|
| H1：V4 主要受修正方式限制 | 固定 pooled 输入，逐项去掉 bound/anchor，比较独立 head | 当前约束或 readout 限制了效果 | backbone 必然足够 |
| H2：pooling 损失了有用候选信息 | 完整 slots 与 pooled 的受控对照 | 候选身份/信息读取有价值 | 所有未见动作都能从 slots 识别 |
| H3：decoder evidence 不足，原始局部视频仍有信息 | 局部视频 verifier 对比 slot readout | 旁路局部证据值得学习 | 通用 visual eventness 足够 |
| H4：监督未充分约束最小语义差异 | 同一结构 BCE vs BCE+source-pair | 对应反事实监督改善迁移 | 任何新造负例都可靠 |
| H5：定位误差限制了 verifier | 对正负 query 同时使用 source GT ROI 的诊断 | 减少候选误差后证据可辨性改变 | GT 支持下的结果是可部署性能 |

贡献候选需由实验支持：证据瓶颈诊断、保持定位的局部兼容性学习、语义新颖性与不存在性联合评测。方案阶段不承诺效果或新颖性。

## 2. 协议边界

1. 正式模型训练只读该 split 的 S+/S−；模型输入不包含 semantic IDs、construction_type、source_qid、exist_label 或 source positive query。
2. 正式选模和阈值只用 Seen validation。U 数据读取只能由独立最终评测入口执行；训练、缓存提取、配对和内层开发入口默认拒绝 U。
3. 五正式 split 共用架构、loss、超参数及选模规则；分别训练。跨 split 的 Seen 标签不能共享给其他 split 的任务模型。
4. 全量主 loader 保留原始训练分布；每 epoch 覆盖所有有特征的 Seen qid。无配对资格只影响辅助 stream，不能导致主 loader 丢样本。
5. canonical baseline 固定来源、feature order CLIP→SlowFast、slot 数和预处理。读取配置以 checkpoint 中的 opt 为准，不能凭目录名称推断。
6. 原定位模块冻结参数并持续 eval。冻结 head 但修改共享 transformer 不能宣称定位不变。
7. 历史结果原样保留；修订指标写入独立文件，明确 evaluator version。不得静默覆盖旧 summary。

## 3. 阶段与依赖

| 阶段 | 内容 | 交付物 | 通过条件 | 依赖 |
|---|---|---|---|---|
| P0 | 统一 evaluator 与 baseline 审计 | 修订评测、全精度 baseline、审计报告 | 排序/后处理/单位一致，qid 完整 | 无 |
| P1 | source-pair manifest 与 feature bank | Seen 配对、缓存 schema、覆盖报告 | 无越界标签；主流覆盖完整 | P0 |
| P2 | 内层语义留出开发任务 | fold manifests、内层 baseline、协议哈希 | 任务训练未暴露留出语义 | P0/P1 |
| P3 | 表示读取对照 | pooled/slots/local 对照与机制诊断 | 可解释的读取差异；决定主结构 | P1/P2 |
| P4 | 反事实监督和必要消融 | BCE、source-pair、ROI 辅助对照 | 内层收益可重复，Seen 代价可接受 | P3 |
| P5 | 冻结并跑正式五 split | checkpoint、完整预测、统计报告 | 全部完成固定预算，完整报告失败 | P4 |
| P6 | 多 seed 与确认性研究 | seed 报告、新数据协议、研究结论 | 结论与证据强度一致 | P5 |

P0/P1 的独立 CPU 审计可以批量执行；GPU 训练按实际可用设备排队，默认每设备一个任务。实验不因历史 U 点估计修改剩余 split 的设置。

## 4. P0：评测和结果记录

### 4.1 统一定位口径

- canonical slot score 为 `softmax(raw_class_logits, dim=-1)[...,0]`；采用等价 margin 排序时需声明实现和浮点 ties 的处理。
- 固定稳定排序，ties 按 slot index；二分类前景/背景索引从 criterion 核实。
- 分别记录 `raw_unprocessed` 与 `raw_postprocessed`。官方 soft-gated 输出走同一窗口后处理，不与 diagnostic hard gate 混用。
- 统一 span 类型、duration 缩放、clip_length、裁剪、取整、候选截断及 R1/mIoU 的百分比/比例单位。
- 区分不存在性分类 AUROC、raw localization 和官方 GMR 指标，不能互相替代。

修改位置：`training/moment_detr_gmr_auc_v4/metrics.py` 和可复用 evaluator；历史 V4 修订另存 `metrics_revision_01.json`，汇总报告标注修订版本。对 V1–V3 能从已有完整字段重算的项统一重算；缺少原始字段的项保留并标为非直接可比。

### 4.2 必要验证

- 用二分类 logits 的反例确认只按前景 logit 会产生错误 top-1，验证 softmax/margin 的预期排序。
- 相同 span/class logits 输入统一 evaluator 得到相同定位指标；raw 和 postprocessed 的差异可解释。
- 单独运行 baseline 与 verifier 包装后的 forward，比较 spans/class logits，记录最大误差；不能仅把一次 forward 的同一数组复制到两列当作独立一致性验证。
- AUROC 使用全精度 logits；probability 的饱和 ties 另外报告，不对四位小数历史值逆变换恢复排序。
- qid 唯一、覆盖和 GT 集合一致；NaN/Inf、特征缺失、label count 全部进入报告。

### 4.3 时间轴审计

核查预计算 clip grid 与视频 duration、clip_length、特征截断、TEF 的实际映射。局部 ROI 的 index 必须来自被记录的时间映射，不能未经检查就把 span 秒数乘 `T/duration`。若原特征存在截断，记录可见时间范围；GT 超出范围的样本仍进入全视频 BCE，无法支持的 ROI 辅助项跳过并统计。

P0 通过之后才允许对“局部证据”进行因果解释。

## 5. P1：数据契约与缓存

### 5.1 source-pair 清单

输出每 split 的 `train_source_pairs.jsonl` 和 Seen-only `val_source_pairs.jsonl`，每条包含：

`pair_id, split, positive_qid, negative_qid, vid, source_qid, construction_type, positive_query_hash, negative_query_hash, source_windows, verification_status, pair_eligible, exclusion_reason`。

资格检查：两 qid 真实存在于当前允许的数据分区、同视频、正负标签正确、source 对应一致、特征可用、query 非相同文本、两个 query 都满足当前训练语义资格。metadata 矛盾时阻断相关 pair，不擅自改标签。

当前预计各正式训练 split 为 1499/1500 个可关联 pair；以新导出的审计结果为准。缺源的负例继续进入主 BCE，不从其他 split 或 U 导入源正例补齐。

测试 source-pair 构建只由最终评测入口执行。已有 `matched_u_pairs.jsonl` 的 benchmark PairAcc 保留独立名称，不能与新 source-pair 指标混为一个数。

### 5.2 feature bank v5

| 字段 | 内容 | 存储要求 |
|---|---|---|
| qid/vid | 样本、视频标识 | 索引字段，不进入预测 head |
| base_logit/base_repr | 原 logit、pooled decoder state | 兼容 V4 |
| decoder_slots | final slots，通常 Q=10、D=256 | query-specific，禁止拿正 query 的 slots 当负 query 的表示 |
| raw_query_tokens/query_mask | checkpoint 所用 CLIP token features | 保持原截断与归一化；冻结基础 features |
| raw_video/video_mask | CLIP 与 SlowFast 视频序列 | 按 vid 去重存储，模态与 TEF 分开记录 |
| spans/class_logits | 原始候选输出 | query-specific，完整精度 |
| time_mapping | clip 支持范围、duration、截断信息 | P0 审计后固定 |
| labels/source metadata | 标签与配对索引 | 仅供 sampler/loss/诊断，不能进入模型输入 |

使用按 vid 去重的 sharded bank 或视频路径索引，避免每个 query 重复写入完整视频。抽样估算磁盘、RAM 和 GPU 使用后确定 shard 大小。初始保存 fp32；若改用低精度，须先证明 baseline replay 和 readout 指标不受实质影响。

缓存 identity 包含 schema_version、checkpoint hash、annotation hash、feature order、extractor code hash、normalization 与时间映射版本。任何不匹配均重新提取；存在文件不等于缓存有效。

交付 `coverage_audit.json`：主流唯一 qid 覆盖、pair 覆盖/缺源原因、每类 label/edit type、可用 ROI 比例、特征缺失、NaN/Inf。

## 6. P2：真正的内层语义留出

### 6.1 三个数据角色

在正式 Seen-only 范围内定义 `inner_train`、`inner_seen_val`、`inner_novel_dev`。前者训练；Seen val 选每次训练的 checkpoint/阈值；novel dev 只用于选择研究结构和共同超参数，不用于梯度或 epoch selection。正式 val/test 的 U 不进入这些角色。

### 6.2 划分规则

- 用 released action_base/composition 定义留出，semantic metadata 只参与协议与审计。
- 按 vid 分组，源样本及派生负例始终在同一视频组；同视频其他 query 也不能穿越训练/验证。
- 语义留出资格覆盖负例的 source query：若源 query 暴露留出动作/组合，即使编辑后的负 query 是 Seen，也不能进入 inner_train。
- 为 composition 留出检查动作和对象 primitive 在训练中分别有支持；否则标成复合缺失，不声称纯组合留出。
- fold 候选只根据 Seen 元数据支持量筛选，不能参考任何模型 score。初始支持下限为每个 fold novel dev 正/负各至少 50、distinct vid 至少 30；action 必须有训练可比组。阈值是开发可行性规则，不是统计保证。
- 目标为 action 与 composition 各至少 2 个非相同语义 fold。生成器不足时报告不可行，按预设层级扩大数据池/减少组数；不得静默挑选效果好的 fold。

初始使用 A1 与 C1 的 Seen-only 数据分别开发 action/composition 协议；正式 held-out semantics 仍禁止进入各自任务训练。更换开发 parent 只能因元数据支持不足、特征或运行问题，并记录理由，不能因性能换 parent。

### 6.3 内层 baseline

从任务训练前的初始化重新训练每个 fold 的 baseline，沿用 canonical 模型与基线训练规则：100 epochs、seed 3407、Seen-val MR-full-mAP 选模。所有研究变体共享该 fold baseline。

canonical best.ckpt 已用过内层留出语义，不能直接作为 pseudo-unseen teacher；它仅用于正式 split 固定 baseline 或 Seen subgroup 快速调试。快速调试不作为进入正式实验的泛化证据。

## 7. P3：读取对照，逐项隔离

### 7.1 V4 限制诊断

先在内层 baseline 的 pooled bank 上检查 V4 reference，然后保留其训练设置分别只去掉 bound 或只去掉 anchor。此阶段用于判断约束影响，不与后续不同 loss 的 readout 直接做因果归因。保存 delta 分布、bound 饱和率及训练/验证 AUROC。

### 7.2 统一 BCE 的读取实验

| ID | 输入 | 变化 | 作用 |
|---|---|---|---|
| R0 | 原 baseline | 无更新 | 对照与合法 fallback |
| R1 | pooled representation | standalone MLP，BCE | 重新读取 pooled 信息 |
| R2 | pooled representation + query tokens | 加显式 query 输入 | 区分 query 输入与 slots 的作用 |
| R3 | 完整 slots + query tokens | query-conditioned set readout | 保留候选身份 |
| R4 | 固定 ROI 局部视频 + query tokens | 局部交互 verifier | 读取 decoder 外的证据 |
| Cq/Cv | query-only / video-only | 去掉另一模态 | 检查文本或视频 shortcut |

R1–R4 使用相同 BCE、主 loader、训练 seed、epoch budget 和 checkpoint rule。hidden size/容量从有限开发候选选择，报告 trainable parameters/FLOPs；参数量差异超过 20% 时加一个容量匹配对照，再解释输入效应。R2→R3 是 pooling 对照，不能拿 R1→R3 的全部差异归因于 pooling。

R3 是 permutation-invariant set readout；不靠人为 slot 顺序编码。R4 保留区间内时间顺序，模型只能读取当前 query 自己的候选及原始视频。

### 7.3 GT 支持诊断

对已有 source 正负 pair，使用同一 source GT ROI 分别输入正/负 query；固定其他条件，比较预测 ROI 与 GT ROI 的 PairAcc/score gap。仅用于诊断，不是正式推理结果。禁止只对正例使用 GT，禁止把正 query 的 query-conditioned hidden state 传给负 query。

同视频配对的判别并不能代替 pooled AUROC；两者及 edit type 分层都报告。不同视频同 query 的检查只使用存在真实已标注正负的组合，不把随机交换视为已知 absent。

## 8. P4：V5 方法候选及目标函数

### 8.1 结构

候选来自固定 canonical baseline。对第 j 个区间抽取时间有序的视频 tokens，经独立视频投影与 query token cross-attention 生成联合表示，输出完整 query 兼容性 `e_j`。原 video/text features 冻结，新增投影/交互/head 可训练。

初始规格：hidden=128、4 attention heads、1 层交互、每 ROI 8 个时间采样点、dropout=0.1；所有 Q 个原始 slots 保留，attention masks 处理无效 ROI。时间采样采用审计后的 grid，不能把零填充当有效证据。候选 class score 默认只作诊断，不预先充当存在性乘子。

默认全视频分数 `s = logsumexp(e_j) - log(Q_valid)`，温度 1；每候选先融合完整 query，再汇总候选，避免跨候选分别拼接动作/对象。该 pooling 是可检验的工作默认，不是无偏 existence estimator。max pooling 或 baseline residual fusion 只在预定开发对照中选择。

推理输入只有当前 query、视频特征与该 query 的预测区间；不读取 GT、source_qid 或源正 query。

### 8.2 损失与辅助流

第一项自然分布 BCE；第二项 source pair：

`L_pair = mean softplus(1 - s(v,q+) + s(v,q-))`。

主 pair loss 中，每个 query 使用自己的预测候选及 query-specific 表示。视频相同不意味着两次 forward 的候选必然相同。若要检验纯语义证据差异，用另行标记的共享 source GT ROI 诊断/辅助项。

GT ROI 辅助项为正 query 的局部兼容性正标签、同 ROI 上反事实负 query 的负标签；先作为单独消融，权重固定候选 0.25。它不承担全视频 absence 标签的替代角色。共享未修改 phrases 不施加不存在性标签。

| 对照 | 结构 | 目标 |
|---|---|---|
| S0 | P3 选中的读出结构 | BCE |
| S1 | 相同结构 | BCE + λ_pair L_pair |
| S2 | S1 | + 0.25 ROI 辅助 BCE |
| S3（有条件） | S1 或 S2 | + 固定 global ranking 项 |

S3 只在内部“PairAcc 提升而 pooled AUROC 无提升”的预定义现象出现时实施；它是新增开发实验，不能直接在正式 U 上选择。global AUC margin=1、λ=0.25。anchor、group worst loss、phrase roll negatives 不默认加入。

配对 stream 独立 uniform-over-valid-pairs，配对 batch 初始 32。主流 batch 32，不足显存时使用 gradient accumulation 达到有效 batch；最后不足 batch 保留。每 step 记录来自主流和 pair 流的样本数与梯度范数，pair 类型平衡作为可选消融，不能重排主 loader。

## 9. 训练预算和有限开发规则

| 项目 | 工作默认 | 修改边界 |
|---|---|---|
| verifier epochs | 50，关闭早停 | OOM/NaN 修复须记录并一致应用 |
| optimizer | AdamW，lr=1e-3，wd=1e-4 | 内层 lr 候选仅 1e-3 / 3e-4 |
| hidden | 128 | 128 / 256；完成基础读出对照后才比较容量 |
| dropout | 0.1 | 初始固定 |
| pair weight | 0.25 | 仅 0.25 / 1.0；不做全排列大 sweep |
| margin | 1.0 | 初始固定 |
| gradient clipping | norm 1.0 | 记录裁剪比例；更改需独立开发记录 |
| seeds | 3407、3408、3409 | 多 seed 是 verifier seed；baseline seed 单独记录 |
| checkpoint | Seen-val pooled AUROC | 同分优先早 epoch；pair/novel dev 不选 epoch |

阶段筛选先 seed 3407；只有前两名结构和最终监督方案扩展至三个 seed。任何 lr、容量、pooling/fusion 开发候选均先在内层完成，不在正式 split 单独优化。

baseline 是独立 fallback 候选。standalone verifier 的 epoch 0 随机输出不等于 baseline；只有明确 zero-init residual 路径才可以声称 epoch 0 复现 baseline。

正式运行中若 best verifier 的 Seen-val pooled AUROC 不超过 baseline，deployment checkpoint 回退 baseline，同时保留 verifier best checkpoint 和未通过记录。baseline 不参与 verifier 梯度。最终同时报告 learned verifier 与采用 fallback 的策略结果，避免失败被隐藏。

正式模型沿用内层选择出的共同超参数；每 split 仍按 Seen validation 选 checkpoint。Seen val 达标不保证 Seen test 或 U 达标。

## 10. 阶段门槛与停止规则

以下数值是本方案建议的工程筛选标准，不是已观察到的收益或统计定律。正式 freeze 前可依据仅 Seen 内层数据的方差/用途调整一次并记录依据；读取新正式 U 结果后不能追改。

| 门槛 | 条件 | 未通过的处理 |
|---|---|---|
| G0 评测可信 | qid/特征覆盖完整；排序和时间映射明确；无 NaN/Inf | 修复评测/输入，停止训练解释 |
| G1 内层资格 | 至少 action、composition 各 2 个有效 novel folds；teacher 无任务语义暴露 | 修复划分/重训 teacher；不能改叫成功泛化 |
| G2 值得推进 | 三 seed 平均 inner-novel macro AUROC 提升 ≥0.01，至少 3/4 folds 为正，Seen macro 降幅 ≤0.005 | 保留为负结果；按机制分支处理 |
| G3 定位保持 | 独立比较定位 tensors 一致，同输入指标一致 | 修复 eval/freeze；若有意变更则新设联合训练方案 |
| G4 正式完成 | 五 split 每个 50 epochs；无因 U 变化而改配置；全部报告 | 失败 split 保留失败状态，不能只汇总成功项 |

G2 是投入五 split 的筛选规则，不要求将开发集 bootstrap 置信区间当作确认性证据。正式结果判断同时关注效果大小、视频 cluster CI、seed 稳定性和逐 split 退化；macro +0.01 可作为实用效果目标，但最终声明必须由实际区间与复现支持。

| 观察 | 后续动作 |
|---|---|
| R2 优于 R1，R3 未优于 R2 | 优先显式 query readout；不宣称 pooling 是核心原因 |
| R3 稳定优于 R2 | 采用 slots/set readout；局部视频分支视新增收益决定 |
| R4 优于 R3 | 采用独立局部证据 verifier |
| GT ROI 明显好于预测 ROI | 记录候选支持限制；后续定位调整作为独立研究 |
| query-only 已接近完整模型 | 审查构造偏差/文本 shortcut；不能用完整模型收益宣称视觉证据迁移 |
| pair 提升、pooled 不提升 | 分解组内/跨组 ranking；按预定义规则开发 S3 |
| 所有 readout 均失败 | 检查时序信息与标签；再立项更强特征/共享表示更新，结束当前 V5 假设 |

## 11. P5：正式评测与统计

### 11.1 指标

主指标：逐 split U pooled AUROC 与其相对 canonical baseline 差值、五 split macro。辅助：Seen pooled AUROC、benchmark matched U PairAcc/n、source-pair PairAcc/n、统一 raw localization、官方 soft gate GMR、FRR/RR、edit type 与 action/composition 分层。

分层没有双标签时报告不可估计及样本量；不静默丢组。gap 不能作为替代成功标准；不对 macro 正值隐藏退化 split。

### 11.2 bootstrap 与 seed

- 每 split 按 vid 聚类，draw 数为其 distinct vid 数，有放回抽样，抽中的视频带入全部相应 rows，两模型共用抽样。
- 5000 个有效 replicate，seed=3407；单类别 replicate 丢弃并计数，最多尝试 50000 次。不足有效数量时报告不可估计，不静默改成 row bootstrap。
- macro CI 从五 split 的联合 vid 集合进行同一视频权重重采样，然后每 split 计算 AUROC delta 再取均值，保留跨 split 同视频依赖。每 split 有效类别均存在才接纳 replicate。
- PairAcc CI 按 vid cluster 保留 pair 对应关系。小 matched subset 明确样本量，不外推整个 U。
- verifier 三 seed 报告均值、std、逐 seed delta；固定 canonical baseline 不包含 baseline 重训不确定性。若论文结论涉及完整训练程序，另安排 baseline 重训 seed。
- 非零 delta 的方向、大小、CI 和多 seed 分开叙述；CI 包含 0 表示不确定，不表示等效无效。

### 11.3 冻结与运行

采用 `development → ready_to_freeze → frozen → trained → selected → evaluated → aggregated` 状态。freeze 必须记录 code hash/git 状态、配置 hash、各 split annotation/feature/checkpoint hashes、内层 fold manifests、被尝试的开发变体、最终选模/阈值规则及所有 gate。

正式 U 入口要求 selected checkpoint 与 Seen threshold 已落盘且 checksum 固定；test inference 完成后不得用 U 选择其他 epoch。同一次 protocol 下的新 bug 修复要产生 revision，记录受影响指标，不能静默重写历史。

保存运行命令、device、software versions、start/end、epoch 数、loss/gradient、训练流覆盖、最佳 epoch、fallback、完整 logits/候选 score、cache identity、统计 seed。

## 12. 实施文件与责任边界

下列均为计划新增路径，当前不保证存在；实现时采用 module entrypoint 和显式路径参数，避免训练脚本绑定个人目录。

| 文件/模块 | 工作 |
|---|---|
| `scripts/recompute_canonical_metrics.py` | 同口径历史/当前重算，输出 revision |
| `scripts/prepare_source_pairs_v5.py` | Seen source manifest 与资格审计 |
| `scripts/prepare_inner_semantic_folds_v5.py` | 元数据驱动 folds、video grouping、语义暴露检查 |
| `training/moment_detr_gmr_evidence_v5/feature_bank.py` | 带 identity 的分片缓存与去重视频读取 |
| `models/moment_detr_gmr_evidence_v5/readouts.py` | pooled、set readout、local verifier 与控制模型 |
| `training/moment_detr_gmr_evidence_v5/losses.py` | BCE、source pair、可选 ROI/global 项 |
| `training/moment_detr_gmr_evidence_v5/train.py` | 全量主流、独立 pair stream、选模与状态 |
| `training/moment_detr_gmr_evidence_v5/infer.py` | Seen 校准与单独最终 test 入口 |
| `training/moment_detr_gmr_evidence_v5/metrics.py` | 分层、视频 cluster bootstrap、统一输出 |
| `scripts/freeze_evidence_v5.py` | 检查 gate 并生成 freeze |
| `scripts/run_evidence_v5.sh` | 指定 split/seed/device/stage；缓存和 checkpoint 校验 |
| `scripts/aggregate_evidence_v5.py` | 五 split、多 seed、失败/fallback 完整报告 |

采用四类重要测试：evaluator 排序/时间映射；分区/source pair 资格；冻结定位前向与参数无梯度；统计抽样保持 cluster 与 pair 对应。再做小批次端到端训练验证 loss finite、覆盖计数、checkpoint replay。测试用于上述实际风险，不为每层实现重复写镜像测试。

## 13. 资源和排期

现阶段没有测量 V5 每 epoch 时间、GPU 峰值显存或可用磁盘。以下是依赖安排，不是保证完成日期。

| 工作批次 | 内容 | 估计工程时间，不含排队/训练 |
|---|---|---|
| 批次 1 | P0 指标、时间轴、P1 pair 与缓存 schema | 2–3 个工作日 |
| 批次 2 | P2 fold 生成、内层 teacher 管线；P3 模块实现 | 3–5 个工作日 |
| 批次 3 | P3 读出筛选、P4 监督消融和三 seed | 随运行测量更新 |
| 批次 4 | P5 freeze、正式运行、统计与报告 | 随运行测量更新 |

初始最小运行预算：4 个内层 baseline ×100 epochs；4 folds ×4 个研究 readouts ×50 epochs。Cq/Cv、V4 限制诊断和三 seed 分开计入；逐阶段计时后在 `resource_budget.json` 写明 GPU-hour 上限和停止点。不能为节省算力跳过内层 teacher 的语义排除。

正式筛选一轮为 5 splits ×50 verifier epochs；最终三 seed 为总计 15 runs，首轮 seed 计入其中。必要 baseline 多 seed 是额外预算。GPU 无空闲时排队，默认不终止现有进程。

## 14. 完成定义

实施完成需要 P0–P5 所有必需交付物和 gate 记录，不以训练进程结束为完成。研究结论完成还需要对 H1–H5 作支持/反驳/未能区分的逐项判断。

若达到研究目标，报告具体机制收益与适用边界；若未达到，保留完整对照并指出下一步需要改变的假设。未来确认性研究必须另用未参与开发的新语义留出/数据，并在看结果前制定协议。

当前可立即实施的第一批任务：P0 evaluator 修正和历史指标 revision、P1 Seen source manifest、P2 fold 元数据可行性报告。新 V5 的五 split 训练在 G0–G3 通过后执行。
