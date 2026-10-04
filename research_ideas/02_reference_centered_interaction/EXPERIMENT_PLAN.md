# 02 · 最小验证与 reference sensitivity 设计

状态：`draft_not_executed`。方法见 [IDEA.md](IDEA.md)，数据和选模规则见 [COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。当前没有 feature extraction、bank 构造、训练或新增 inference。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 实验问题

分别检验 P1：固定 scorer 的单模态 offset correction 是否有用；P2：在 centered score 上训练是否比 post-hoc 更好；P3：收益是否来自真实视频条件关系；P4：收益是否仅由 reference bank 选择或额外容量解释。P1 与 P2 不混为一个结论。

## 2. 输入、reference 与边界

仅使用既有四个 inner folds 的 train / Seen-val / Novel-dev。第一轮使用冻结预计算视频序列和文本特征的 masked mean，不使用秒级 GT ROI；禁止假定不同模态的原始向量已在可直接点乘的共同空间。当前语义过滤遵循 fold manifests。

每 fold 从 inner train 固定两个 reference banks，保存成员、feature/encoding hash、采样 seed 与 weights。首轮建议每 bank 128 个唯一 video/query，另用 32 与 512 作仅对最优候选的 sensitivity；这些是待冻结建议，实际规模受有效支持限制，不重复抽样伪装独立样本。默认均匀按唯一实体采样；频率加权、semantic-stratified 是后续显式 factors，不从 evaluation labels 选权重。

Reference videos 无需、也不能假称对所有 Q 均 absent。RQ 里的 query 保持自然表达，不添加 holdout 语义。开发选择不能用正式 U 的 bank performance。

## 3. 最小 arms

| Arm | 训练分数 | 测试分数 | 固定比较 |
|---|---|---|---|
| B0 | Raw `f` + global intercept | Raw | 同 rank 双线性 source BCE |
| B1 | 与 B0 完全同 checkpoint | `f−A−B+C+b0` | 纯 post-hoc reference correction |
| B2 | Centered `e+b0` | Centered | 将中心化纳入优化的候选方法 |
| B3（可选） | Raw | Query-side 或 video-side 单项 correction | 归因于哪一侧，而非直接作为方法 |

B1 的共享 intercept 不影响 AUC；若要报 calibrated BCE / threshold，需只在 Seen-val 拟合 intercept，不能动 U。B0/B2 采用同 bilinear interaction rank、投影、epochs 和自然 BCE；有效参数不同必须记录，可另加容量匹配 raw/centered variants。首轮不加入 01 的 rank loss，避免同时改变监督。

最小 feature projector 建议 128 维、linear bilinear head、50 epochs、seed 3407。学习率沿用共同 head recipe；初始化和正则统一。Linear 投影下先中心化冻结输入均值，验证与四项公式等价；不需要缓存会过期的 trainable output means。非线性 projector 是明确的第二阶段实验，不在第一轮偷偷增大一侧容量。

一至两张 GPU 可按 fold 排队。B1 重算同 checkpoint 的参考项属于未来 inference；当前仅写方案。对 general nonlinear f，记录每 step reference forward 次数，并提供相同预算 raw comparator；不能只说 backbone 冻结就等价于算力相同。

## 4. 机制完整性检查

在实施时用人为的 `f=a(Q)+b(V)+c` 输入验证四项相消，误差随 float precision 给出；这项是公式实现核对，不是研究性能结果。Pure Cq/Cv 经完全相同 reference 路径后应得到常数。Global b0 允许不平衡训练标签，不能让 query-dependent intercept 重新进入 score。

从同一 B0 checkpoint 验证所有真实 quartet 的 D 在 B0/B1 中一致。若不同，先排查 bank、score precision、不同输入或实现不共享。Centering 可以改变严格 group 指标，但不能创造原 scorer 不存在的 D。

## 5. 六类评价与判别

使用原始 pooled Seen/Novel AUC、same-video、same-query、quartet四项/strict-group/D、独立单模态、input-level shuffle。尽量用已经核验的现有支持；若 B2 在原始 pooled 上好但 same-query 不升，则可能主要改善跨 query 排序。

### Reference sensitivity

对固定 B0/B2 参数，在三个 train-only bank seeds 上评价；每个 bank 都保持规模与权重。可将最优一至两个方案扩展到规模32/128/512，报告均值、最差变化与跨 query排序稳定性。不选最好 bank 当主结果，不把不同 banks 当独立实验样本计算显著性。

### 对照现有 normalization

开发有信号后，以同 raw score、同 Seen-only bank 比较 QB-style inverted softmax、CSLS-style neighborhood scaling 与本方向 fixed-bank double centering。DBNorm 要求适配 gallery/query banks 的定义，RCSLS 风格训练也需明确 feature space 和监督映射；不能套名称却改变原机制。论文原方法的经典任务指标不等于本项目存在性 AUC。

### 条件监督 factor

仅当基础 B2 有信号后，做 `raw/centered × BCE/crossed supervision` 的 2×2 factorial：同 verified matrices、aux exposure 和 ranking weights。这样才能判断监督和 score decomposition 是否互补；不能把组合结果全部归给中心化。

## 6. 成功、失败与下一步

| 观察 | 解释边界 | 下一步 |
|---|---|---|
| B1 pooled提高，B2不进一步提高，D未变化 | 支持ranking correction，未支持新交互学习 | 对标QB/DB等现有方法，缩小方法贡献 |
| B2 pooled和条件指标均提高，D改善 | 支持训练出的interaction更可迁移 | 多seed、同scene negatives、三backbone |
| B2条件提高、pooled下降 | 兼容性有所增强，主指标尚未解决 | 分析合法先验/scale，不删失败结果 |
| B2只对一个bank有效 | reference sensitivity大 | 不宣称稳定semantic invariance |
| B2 train失败、raw成功 | 去掉先验后interaction容量/信息不足 | 05序列adapter或03witness |
| Bilinear B2无效 | 当前简单factorization不足 | 不排除一般nonlinear或局部证据 |

推进门槛使用共用协议建议，包含 Seen保持和≥3/4 fold趋势；seed confirmation 与 paired video bootstrap 分开报告。Formal U 只在配置冻结后做，已经看过的历史 test 不能重新声称完全 untouched。

## 7. 对 canonical baseline 的后续归因实验

B0–B2 使用新 raw-feature scorer，属于方法可行性 pilot，不直接归因原三个 baseline 的退化。Pilot 有效后，选择各 canonical backbone 的固定 score，在保持其权重和实际输入路径时重算相同 Seen-only reference terms；报告 prior removal 对其 natural pooled 与条件排序的影响、参考规模和 forward 成本。需要严格 inner Novel 验证时，各 backbone 必须拥有未见对应 holdout 的 checkpoint；不能复用见过该语义的 canonical model 冒充 inner Novel。

如果新 scorer 改善而 canonical correction 不改善，应将结论限定为新的存在性路径/训练机制更可迁移，不能声称已证实原模型的加性偏置是共同主因。

## 8. 最强混杂与资源停止点

Reference normalization 可通过 source density 改变排名，但不能证明事件成立；大量 bank forward 可形成预算优势；raw model 的偏置分支使有效参数不同。最小验证失败时先确定是哪项没有成立，不开展长序列 nonlinear 四项重算或全 backbone 联训。若只有 query prior removal 的逻辑保证而没有自然 AUROC 收益，该结果仍有诊断价值，但不作为缓解 AUROC degradation 的成功方法。
