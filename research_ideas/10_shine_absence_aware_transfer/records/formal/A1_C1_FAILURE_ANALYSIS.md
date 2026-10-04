# A1/C1 正式 Unseen 未改善：原因分析

本次分析基于固定 seed 3407 的 A1/C1 正式结果、50-epoch 训练历史、全精度预测、源码，以及新增的训练数据诊断。不更新模型、不根据 U 更换 checkpoint 或调参。下面区分观察与待验证解释；不能由两 split 单 seed 确定唯一因果根因。

## 1. 最直接的问题：新增目标在训练开始就损伤原有判别

| Split | Arm | Epoch 0 Seen-val | Epoch 1 Seen-val | 最佳训练 epoch / Seen-val | Epoch 50 Seen-val |
|---|---|---:|---:|---:|---:|
| A1 | baseline | .8367 | .8408 | 10 / .8460 | .8403 |
| A1 | SHINE | .8367 | .7743 | 22 / .7824 | .7797 |
| C1 | baseline | .7883 | .7839 | 1 / .7839 | .7676 |
| C1 | SHINE | .7883 | .7599 | 1 / .7599 | .6986 |

损伤在正式 U 评价之前就出现。A1 第一个 epoch 的 Seen-val 下降 6.24 pp，C1 下降 2.83 pp。C1 到第 10 epoch 已降至 .6764。原 GMR train loss 也更高：第一个 epoch A1 baseline .947、SHINE 1.551；C1 1.536、2.091。这些是原任务被新增目标干扰的证据，不能只归因于正式 U 太难，也不支持简单增加 epoch 解决。

选模规则固定为最好的训练 epoch，不允许 epoch 0 作 fallback；两个 SHINE 的最佳 Seen-val 均未超过初始化。即使允许回退，只能保留 canonical，不能证明缓解 Unseen。

## 2. 最优先怀疑的项：额外轮换负例 BCE 强度与分布

实际目标为原 GMR + coarse + fine + rotated-negative BCE + .2 × existence pair。最后两项是本次 GMR existence 适配增加的监督；失败不能直接归因于 SHINE 原始 coarse/fine 思路。

自然训练负例占比 A1 17.43%、C1 14.26%。每个 batch 再增加同权重、全为负例的轮换 BCE 后，仅按两组 existence BCE 的样本权重计算，有效负例占比变成 58.71% 和 57.13%。这不表示标签错误；按用户确认继续视轮换 negative 为有效。标签有效性并不保证监督权重与难度组成适合原任务。

进行了原 canonical checkpoint 的前三个固定训练 batch 的 autograd 诊断，模型 eval 模式关闭 dropout，不运行 optimizer。仅统计共享 transformer 与输入投影的梯度：

| Split | rotated BCE 梯度范数 / 原 GMR | 与原 GMR 梯度 cosine |
|---|---:|---:|
| A1 | 2.01–4.28 | −.198、−.168、+.054 |
| C1 | 2.19–3.06 | −.134、−.117、−.131 |

局部诊断显示新增 BCE 主导初始共享网络更新，并在 6 个 batch 中有 5 个与主任务反向；coarse/fine 的梯度范数比它小。训练使用全局 gradient clipping .1；该操作限制总梯度长度，不消除目标之间的方向冲突。AdamW 的实际更新也不能仅由原始梯度范数精确预测。

这支持“新负例监督过强，造成已有判别能力损伤”的解释，但仅为初始三个 batch 的局部证据，不能替代整个训练期的因果消融。详见 [GRADIENT_DIAGNOSTICS.json](GRADIENT_DIAGNOSTICS.json)，可复现入口 `code/diagnose_gradient_conflict.py`。

## 3. 分数被压低，但 Unseen 的正负排序没有改善

A1 baseline 的 U+ / U− 平均 logit 为 9.296 / 9.434，SHINE 为 −1.004 / −.815：两者都降了很多，但 U− 仍高于 U+。C1 则从 8.768 / 7.867 变成 −.154 / −.275。

均值不能代替 AUC；完整分布与 AUROC 表明排序并未改善。共享单调校准或调整阈值不能修复 AUROC。该现象与新增全负 BCE 的强烈 score 下压相符，但仅靠整体偏移无法解释 AUC 下降，必须同时发生样本间相对排序变化。

同视频 U source-pair accuracy：A1 .6580→.5519，C1 .7126→.6693。Same-query U ranking：A1 .4825→.4381，C1 .4612→.4787。它们没有一致改善，说明新增监督没有稳定增强当前 benchmark 所需的条件判别。

fresh pre-fusion shuffled-video 的 U AUC 仍接近甚至高于原输入：A1 SHINE .4799→.4895；C1 .5672→.5805。错配输入保留原标签，因此这只说明当前评价中对视觉匹配的收益有限，不能把这些分数解释成错配事件的准确率，也不能直接证明完全不用视频。

## 4. 训练负例与正式 U 的困难组成不同

A1 正式 U− 中 action counterfactual 为 964/1119（86.15%）。对全部 U+ 与这类 U− 的 AUROC，baseline .4525、SHINE .4571，仍低于 .5。Object counterfactual 子集从 .7474 降至 .6416。随机轮换即使完全 absent，也不保证提供接近原 query 的未见动作辨别监督；这是难度/语义分布的候选失配，需要训练负例匹配度或受控消融验证。

C1 U− 267/270 为 object counterfactual，只有 3 个 composition 类型。其 object 子集 .5744→.5669；不能仅凭 C1 名称把失败全部解释成 composition-binding 不足，更不能对 3 个 negative 的小子集做强机制结论。

各类型分数都使用该 partition 的全部原正例与相应 negative，属于描述性分层分析，并非配平的因果对照。完整数字见 [FAILURE_SCORE_DIAGNOSTICS.json](FAILURE_SCORE_DIAGNOSTICS.json)。

## 5. Saliency 优化与 GMR existence 之间缺少直接保证

迁移保留了 GMR 原架构：coarse/fine 约束 encoder 视频 token 的 saliency linear head，existence 使用 decoder slots pooling 后的 MLP。两者通过共享表示间接相连；saliency 链成立不保证 decoder existence 的跨 query 排序改善。

此外，fine 使用 upstream 的 `−stopgrad(sigmoid(target)) × log(sigmoid(prediction))`，是单侧交叉熵形式，不是标准 KL 或对称距离。即使编辑层级满足关系完全正确，降低该排序损失也不等价于学习完整事件的 present/absent 判别。本轮 coarse 和 fine 都下降，Seen/U AUC 却没有一起改善，正好说明优化成功不等于目标迁移成功。尚无证据确定 fine 是主要干扰源；局部梯度更优先指向额外 rotated BCE。

## 6. 为什么 pilot 是正向的

Inner pilot 与正式实验不是同一初始化、训练集、辅助 bank 和评测域。Pilot 从 strict-inner V5 checkpoint 初始化，训练 10 epochs；正式从各 split 的 canonical checkpoint 初始化，训练最多 50 epochs。Pilot 第一个 epoch 原 GMR loss：A1 2.744，C1 1.536；正式 A1 .947，C1 1.536。不同阶段的已有能力和损失尺度使同样系数未必有相同作用；不能只看两个方向都叫 A1/C1 就预期复现。

C1 正式最佳模型来自第 1 epoch，所以本次失败不能仅归因于最终训练了 50 epochs。A1 选第 22 epoch，但第 1 epoch 已明显下降。当前结果支持 inner 探索增益未迁移到正式 U；究竟由初始化、语义集合还是负例组成导致，仍需分离控制。

## 7. 已检查但当前缺少支持的解释

特征有效长度对应的 duration/length 中位数约 1 秒，与当前 clip_length=1 接近；两个 split 均没有 GT start 超出特征长度而被强制压到末尾的正例。A1 518/7108、C1 625/9016 个正例存在 GT end 截断。duration/length 不是独立验证的提取 timestamp，不能完全排除时间监督问题，但当前没有明显的倍数时间尺度错误证据。详见 [TEMPORAL_GRID_DIAGNOSTICS.json](TEMPORAL_GRID_DIAGNOSTICS.json)。

冻结代码、checkpoint、预测文件已校验；阈值只由 Seen-val 决定。现有证据不能把失败解释成 U 阈值选择错误，AUROC 本身也不依赖选定阈值。没有将用户确认的轮换 absence 或编辑链有效性当成失败前提。

## 8. 建议的验证顺序

第一优先是拆开“SHINE saliency 机制”与“本次新增 existence BCE”：同初始化、同 seed、同训练数据，比较 baseline、仅 coarse/fine、仅 rotated-existence，以及完整方案。额外 forward 和 exposure 要尽量对齐；先只用 Seen-val 判断是否损伤已有能力，不能根据已看的 A1/C1 U 搜索超参后再称其 untouched test。

然后在 Seen-only 开发数据检验较弱 rotated BCE、warmup、先训练 saliency head 再开放共享网络，确认能保住主任务后再评价泛化。若仍失败，再测试直接作用于 existence 的已核验 hard negatives 与条件排序，而非继续默认 saliency 收益会自动传递。

本轮仅完成分析与无更新诊断，未启动新消融、未改已有训练配方。现阶段最有证据的解释是：新增负例 BCE 在成熟 checkpoint 上产生过强且部分冲突的初始监督，同时 coarse/fine 代理目标没有补上正式 U 的条件排序能力。完整因果判断仍需要上述消融。
