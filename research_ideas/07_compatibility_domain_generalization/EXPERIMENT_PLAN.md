# Idea 07：Compatibility DG 最小实验计划

> 状态：`draft_not_executed`。未划分环境、启动训练或评测。  
> 机制与边界：[IDEA.md](IDEA.md)。统一协议：[COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 启动条件

先得到高优先级 audit 与 ERM+pair 的结果：如果同 query / crossed 可信证据极少，或简单 pair supervision 尚无可学习信号，DG 不应立即投入。最小启动条件是训练侧至少两个具有足够独立视频、标签齐全 pairs、不同构造机制/相关性且身份可追溯的环境。本文不虚构这些数据已经具备。

只用 inner_train_seen 生成环境。每个环境是 complete pair/unit 集合，包含正负 labels；不允许 positive-only 与 negative-only 环境。对 construction environment 明确 altered atom 是任务内容，变的是负例生成机制；对 semantic-family environment 保留动作/对象身份，不做 unconditional feature matching。

## 2. 未来需交付的环境审计

| 项目 | 记录 | 排除的混杂 |
|---|---|---|
| Unit manifest | pair/quartet_id、video/qid、标签、source、构造类型、review、环境 | 错配、随机视频假负例 |
| 环境规模 | units、query 数、独立视频数、复用/连通组件 | 少数共享视频主导 meta-risk |
| 环境相关性 | q-only/cv 能力、长度与语言质量、source/background 支持 | 把环境 label 当 target label |
| 难度支持 | 同监督类型各环境覆盖、action/object/composition overlap | 难度差当 nuisance shift |
| Exposure audit | canonical teacher / raw feature / head 各阶段接触哪些语义 | 错称真正 pseudo-unseen |
| Partition spec | 固定分组规则、seed、处理少样本组规则 | 按 Novel 成绩挑环境 |

没有可信 background/camera metadata 时，先声明只研究 construction/task-family shifts，不用视觉聚类补造“背景标签”。少样本 environments 合并的依据应在训练侧预定；不以当前 loss 或 Novel-dev 分数动态合并。

## 3. 必做最小组

| 组别 | 同结构 head | Natural main | 同一批可信 pairs | 环境处理 |
|---|---|---|---|---|
| D0 | 是 | 完整 BCE | 无 | 无 |
| D1 | 是 | 与 D0 相同 | pair ERM | 忽略环境，pooled pair loss |
| D2 | 是 | 与 D0 相同 | 与 D1 相同 | pooled pair + compute-matched 优化曝光 |
| D3 | 是 | 与 D0 相同 | 与 D1 相同 | compatibility-risk MLDG |

D1/D3 使用相同 unit 顺序、负例、pair temperature 与每例权重；区别是训练侧 A/B 的虚拟更新与 meta-gradient。D2 排除额外 forward/backward 或优化曝光解释，须报告 update 数、example visits、forward/backward 数、实测设备时间，不能只说 epoch 相同就 compute 公平。

第一轮不联合 query-marginal matching/reference centering/witness，否则无法识别 DG 的附加作用。若前一方向的 pairing 已有效，可直接复用其固定 pairs 与 score，但所有对照一起复用。

## 4. Budget 与模型选择

先在一个 action 与一个 composition inner fold、一个固定 seed 做小 head pilot。MLDG 的 `α` 与主 optimizer 学习率尺度先固定一致，`β=1`，DG 辅助总权重 `λ=0.25`；这些是预定起点，不声称 optimal。若 Seen validation 安全性不满足，最多追加 `λ=0.10`，所有 baseline 使用相同候选预算。

本轮 initial variant 用小 head accurate one-step differentiation。若必须一阶近似，建立 D3-FO 单独行，记录实际梯度；不在结果里悄悄替代。每个 step 的 meta A/B 都来自 inner_train_seen；不得让 inner_val_seen 或 Novel-dev 承担带梯度的 meta-test。

Checkpoint 只按相同 inner_val_seen 规则选择；run 冻结后评价 inner Novel-dev。Novel-dev 可以有限跨 run 决定方向，但不是 per-epoch selector，也不是最终独立 test。Pilot 通过才扩展四 fold 三 seed。正式三 backbone/五 split 是确认升级计划，未在本次实施。

## 5. 后续强对照与 ablations

只在 D3 超过 D1/D2 后展开：

1. 同一 pair risk、同环境上的 IRMv1；在 dummy scalar w=1 处施加环境 risk gradient penalty，明确这是已知基线。
2. 同一 pair risk 上的 regularized Group DRO；支持门槛、权重与正则预定，避免小噪声组无界主导。
3. 相同 MLDG 计算，把环境 label 随机打乱但保持环境大小，区分有效环境与 generic meta-regularization。
4. Construction-only、semantic-family-only、两者预定组合；不从所有分组中事后挑一个最高分。
5. Pair-unit DG 与 absolute-BCE DG 同数据对照；检验是否确实需要关系差值这个泛化单位。

每一步独立加一个变量，保留 D1，不能把不断调好的 DG 与只运行一次的 ERM 比较。

## 6. 固定评价

主目标仍是原始 benchmark pooled AUROC(U+ vs U−)，不是最差组 accuracy 或环境风险方差。开发期报告 inner Seen/Novel pooled AUROC、same-video PairAcc、same-query 跨视频 PairAcc/AUROC、四格 row/column/group accuracy、Cq/Cv、可信换视频对照、edit-type/construction strata、hard-positive consistency，以及 frozen localization identity。

训练环境风险只用于解释：更一致的 risks 不构成跨语义改善。必须展示 D3−D1，而不只 D3−canonical。Shuffled 输入原标签评价仅检验 dependence，不能当正确性；same-video 配对仍允许 language prior，same-query 配对仍允许 video prior，四格也不能排除场景—文本交互。

共享视频使多个 pair/环境非独立。按视频或匹配图连通组件聚类计算差值区间；多 seed 离散度另报。环境子集数很少时保留 exploratory 标记，不把环境 mean/CI 当新的独立测试总体。

## 7. 决策表

| 结果 | 下一步 |
|---|---|
| D1 提高、D3 无额外作用 | 以 supervision 为主线；DG 留作消融，不升级模型复杂度 |
| D3 与 D2 相同 | 改善尚可由计算/优化解释；不能写 DG novelty |
| D3 超 D1/D2，环境随机化失效 | 环境信息有机制支持；继续四 fold/三 seed |
| D3 仅减小 source risk 方差 | 不以此推进；检查是否削弱 hard event identity |
| Novel pooled 好而 conditional 无效/更差 | 只能讨论 score comparability，暂停 compatibility 主张 |
| Seen 或 U+ 明显损失 | 降低预定 λ 或终止；不能靠 Seen 下降缩小 gap |
| Action 改善而 composition 退化 | 保留范围，优先 witness/绑定竞争解释 |

共同推进门槛建议为 Novel macro +2 pp、至少 3/4 folds 正向、Seen 下降 ≤1 pp，并有两向/crossed 支持；DG 还须超过同 pairs/compute 的 ERM。阈值是未来预注册决策，不是已实现收益。

## 8. 成本与未来交付

设备规模拟为 1–2 GPU，小 head accurate meta-gradient 的时间/显存先未来测量。最终需留下环境 manifest/spec、exposure audit、选模规则、固定对照表、每样本 full-precision score、共享视频区间、三 seed 结果与失败模式。当前目录只有设计 Markdown，不代表这些产物已生成。
