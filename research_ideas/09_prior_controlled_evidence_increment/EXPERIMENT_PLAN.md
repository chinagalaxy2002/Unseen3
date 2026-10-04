# PEI-0 冻结规范

本轮已实现 Stage 0/1/2，freeze revision 2 固定实现与条件功效模拟。当前没有真实实验新数值结果；13 项合成输入检查通过。方向 01–08 和 V5 的历史结果只用于提出假设。四折是开发集，已经读过 Novel-dev，不声称其为 untouched test。先验匹配配对监督仍须等 E 结果后再实施。

## 假设与识别范围

H1：语义留出 pooled gap 可大量由 query 先验变化解释。H2：冻结的视频文本特征仍含先验之外可迁移的增量。Cq 与 baseline 的 recipe/选模不同，gap 接近不能分解出因果占比，也不能推出 baseline 没用视频。

本实验控制的是已指定的 b、μq、μv，不是全部语言或视频先验。增量可以来自 scene×query；条件兼容性措辞还要求 same-query 与 video-only 控制。N 结果只表示该通道及协议下未检出可用增量。

## Stage 0：标签与特征门控

读取固定四折的三个角色文件，先检查 manifest 和 annotation SHA256，拒绝 U partition、重复 qid、标签与 S+/S- 不一致。inner Novel-dev 在源 release 中仍是 S partition，角色由 strict-inner 文件定义，不能按 novelty_type 重分。

每个角色及 train+Seen-val 联集报告：原始正负行数、唯一 exact query/video、双类 query 数、same-query 行配对数、video-query 二部图连通分量、冲突 cell、重复 cell、现有标签闭合四格、construction_type、审核 provenance、语义组。报告角色之间的 video/query/qid 重叠和跨折视频共享。四格只计已有可靠标签格，冲突格不参与；不补造负例。Exact text 相同仅提供候选分组，特征模式还会检查实际编码 token 是否相同。

原始 CLIP 视频使用全部 clip，不混入 SlowFast，也不使用经 query 条件化的 decoder 表示。文本先逐 token L2 规范化，取前 32 个有效 token 的均值，再 L2 规范化，和 V5 bank 的 masked-mean 通道保持对应。512 维一致不等于处于同一联合空间。

只用 train 正例，按每行 GT 内外 clip 相似度计算 AUC，再报告 row-macro 和视频簇 bootstrap CI。缺少真实时间来源时只能用名义中心作描述，alignment gate 保持 pending。没有内外两组 clip 的行不计入，报告支持数；缺失、非法或非有限特征会阻止后续评分。维度、时间来源或 gate 不通过时，重抽 CLIP projected sentence embedding 必须单独注册模型版本、输入文本、提取设置和 hash。

按唯一视频构造无固定点的置换，一个视频的所有 query 共享 donor；置换仅在 train 内。先算 shuffled-video 的 max/top3/mean pooled AUC 分布，再算真实 train AUC。该 null 保留文本先验和视频长度等混淆，不能强求 AUC=0.5。它既不是 M3−M2 null，也不能产出该增量的 MDE。

## Stage 1：零训练证据与双中心化

`z(V,Q)` 在 max、top-3 均值、全时间均值中按 train pooled AUC 选择，平局按 FREEZE 中的顺序。另两个聚合器是辅助对照，不能根据 Novel-dev 选择。

Reference bank 仅包含该折 train 唯一 video 和唯一 exact query。定义 `μq(Q)=mean_{Vref} z(Vref,Q)`，`μv(V)=mean_{Qref} z(V,Qref)`，`μ=mean_{Vref,Qref} z`，`z̃=z−μq−μv+μ`。所有 reference 均不使用标签。评价点若包含于 reference，删除该视频和该 query；μ 用同时删除二者的 Cartesian reference。评估新 query/video 不需要把它们加入 bank。若同 query 的实际 token 不同，先查明并修复，不能任取一行平均掩盖差异。

Seen train+val 的纯零训练分数可以作描述性评价；由于聚合器用 train 标签选择，其 Seen pooled AUC 不是独立泛化结果。另报 Seen-val、Novel-dev，以及 same-query query-macro AUC、已有审核 source-pair 的 same-video PairAcc。same-query 支持仅限双类 query，CI 用 incidence graph 分量；支持分量少于 5 个或最大分量覆盖超过 50% 的双类 query 时仅作描述，CI 留空。source-pair CI 按视频聚类。

## Stage 2：先验控制后的增量

固定 b 系数为 1，用相同输入、类别平衡和训练 recipe 拟合：

- M0 = b。
- M2 = b + c + aq·standardize(μq) + av·standardize(μv)。
- M3 = M2 + w·standardize(z̃)，其 c、aq、av 独立重拟合。

M2 实际拟合 3 个参数，M3 拟合 4 个参数；不能沿用“2–3 个参数”的粗略说法。标准化统计只来自 train，零方差映射为零。逻辑回归使用固定 b offset、train balanced BCE、斜率 ridge 1e-4，截距不惩罚。零拟合版本把 b 和三个特征按 train 标准化后等权相加，独立报告。

b 为五 seed、按视频五折的 train OOF logit 均值。每个 held-out 视频的所有 query 都必须排除在训练和 epoch selection 之外；选模 validation 从 held-in 视频再划出 20%，外层视频分组由 seed 3407 固定，内部验证由 seed+1009×outer_fold 的随机顺序固定；不因标签或结果重新划分，缺类则停止。评价 Seen-val/Novel-dev 用各 seed 的五个模型 logit 均值。数据量缩减可能影响 prior 能力，报告其 OOF/eval AUROC 和 log loss。Cv 可复用已核验历史版本，但要记录版本及 recipe 差别，不能冒充公平重训对照。

主估计目标为四折宏平均 Novel-dev `AUROC(M3)−AUROC(M2)`。打乱视频对照必须重算 z、μv、μq、μ、重新标准化并重拟合 M2/M3；使用同一组 b。多个 fold 共享视频，bootstrap 同一次 draw 对相同视频使用同一 multiplicity；种子不是独立数据样本。单侧中心化、mean/max、Cv 是辅助解释，不进入主指标选择。

## Null、功效与判据

Stage 2 代码和功效模拟已在 revision 2 冻结。完整流程在真实 Novel 增量计算前完成 shuffle-refit 增量 null 和功效模拟。FREEZE 把“null ≤0.5 pp”明确定义为相同宏增量统计量的 null 95% 分位数 ≤0.005。重复次数不足时应注明分位数精度。

MDE 的 alternative 在 M2 产生的标签模型上注入正 z̃ 系数，固定网格 `[0,.125,.25,.5,.75,1,1.5,2]`。每个设置模拟 200 次，按视频共享正态随机截距（SD 0、0.5、1 三种敏感性），相同 video/query 在所有 fold/role 共享 uniform draw；每角色将期望正例率校准到原始标签比例。每次重新拟合 M2/M3，系数 0 的宏增量 95% 分位数确定单侧 α=0.05 门槛。效应大小为模拟标签下 oracle `base+w·z̃` 对 `base` 的平均宏 AUROC 增量。

MDE 为达到 80% power、且后续更大网格点都达到该功效的最小正 oracle 增量；取三种随机效应设置的最大值。找不到、重复出现缺类、模拟失败均不给 MDE。模拟固定已拟合 Cq 和已观察特征，依赖有限网格与生成模型，不包含重新训练 Cq 的不确定性；不能宣称无条件精确功效。shuffle-refit null 是另一个诊断，不直接用其标准差代替 MDE。

E：宏增量 ≥0.02，至少 3/4 折为正，paired shared-video macro bootstrap 95% 下界 >0，shuffle-refit null 95% 分位数 ≤0.005，且 MDE ≤0.02。只有 E 才进入先验匹配的配对监督方法阶段。

N：四折 same-query AUC 的有效 CI 均包含 0.5，宏增量 <0.01，至多 1/4 折为正，且功效足以支持该判断。措辞为“未检出该通道的可用增量”；CI 包含 0.5 本身不是等效性证明。若无有效 same-query CI，不能套用 N。

MDE >0.02 或无法确定、CI 无效、其他结果均为 inconclusive；保持原阈值，扩大预先指定的语义留出支持。只有 same-query CI 下界 >0.5 且 video-only 对照支持时，才使用“条件兼容性”措辞；E 单独只支持“指定先验控制后的证据增量”。

所有输出保存在方向 09，源标注、特征、banks、checkpoint 保持只读。1–3 小时仅为原提案估计，真实时长和内存必须随执行记录。

代码入口为 `code/run_pipeline.py`，完整使用说明见 README。Cq 优化器使用 AdamW，与实际 V5 训练代码一致。Seen/Novel 的确定性 query-only 推理先按唯一输入向量计算并广播；train OOF 同 query 跨外层模型仍可能不同，不能把其微小条件排序当作视频使用证据。
