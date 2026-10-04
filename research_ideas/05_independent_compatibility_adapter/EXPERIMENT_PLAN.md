# Idea 05 最小验证实验计划

状态：`draft_not_executed`。此文档不含运行命令、启动脚本或新实验结果。所有参数和预算是未来实施建议。共同数据、选模与评价规则见 [COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 启动条件与优先级

本方向优先于继续扩大 decoder readout，通常在 crossed-video audit 与初步监督实验之后进行。若已证据表明原视频与错配视频响应不足，或者更好监督没有恢复 conditional compatibility，才值得检查 representation path。

首轮不需要秒级 ROI provenance，但需要原序列 feature 身份、mask、模态拼接顺序和 inner-trained checkpoint 正确。不得从全 Seen baseline 提取带有 inner Novel 训练暴露的 representation。

## 2. 三臂核心比较

| 臂 | existence 输入 | 共用部分 | 用途 |
|---|---|---|---|
| P | pooled decoder 向量 + query tokens | 同投影/交互/scorer 家族与 loss | 与 V5 R2 类型读取路径连接 |
| S | 全 decoder slots + query tokens | 相同 head 家族与 loss | 排除简单 slots 增加解释 |
| R | 原始 CLIP/SlowFast 序列 + query tokens | 相同 head 家族与 loss | 检验 source-adapted 表示之前的证据 |

另设 Cq、Cv，以及 R-mean（视频原序列先 mean，再用相同交互头），区分语言/视频边际信号与时序证据。head 的总参数应记录到组件，分别列出 input projections 与 attention/FFN/scorer；可调整投影 rank 接近参数预算，但不要假装不同输入支持“完全相同有效容量”。

首轮仅自然分布 BCE。若共同研究流程已有可信 quartet loss，另做固定版本的 P/S/R 三臂，三者使用完全相同 supervision。禁止为 R 增加 hard negatives、oracle windows 或更长训练而不提供 matched comparator。

## 3. 数据与选模

- 四 folds：`A1_action_01`、`A1_action_02`、`C1_composition_01`、`C1_composition_02`。
- 梯度只能来自该 fold 的 inner train；main loader 全量样本覆盖不变。
- 用 inner Seen-validation 按共同规则选 checkpoint；不能按每 epoch Novel-dev AUROC early stop。
- Inner Novel-dev 用于不同预先定义 run 的架构开发，按共同协议限定查看次数；开发后冻结方案，再做独立确认。
- 原始五 split 已参与 idea 形成，后续结果需标明 exploratory，不宣称从未查看的 test。

## 4. 小预算执行顺序

未来可先做一 action、一 composition fold 的固定预算 pilot，排除输入和优化失败；pilot 不用于宣布跨语义收益。通过后按同参数方案补齐另外两 fold。建议 `d=128`、一层交互；optimizer、步数、temperature 从现有稳定 readout 配置继承，经 inner Seen-val 选择，首轮不做大网格。

GPU 数量建议 1–2，无需完整 backbone 梯度或重新视频 feature extraction。训练时长、缓存大小和显存不能从文档估算成事实，实施时才记录。

## 5. 必须报告的指标与控制

1. 自然分布 Seen/Novel pooled AUROC，以各 fold 结果和 macro 汇总报告。
2. Source-linked same-video PairAcc 与其样本覆盖；不能称它已控制语言分布。
3. 相同 query 在已标注 present/absent 视频上的 PairAcc/AUROC。
4. 完整 2×2 quartet 的行/列/整体成功率与 interaction contrast。
5. Cq、Cv、R-mean；原始 canonical score 作为冻结参照。
6. 真视频与 shuffled-video input-dependence control。换视频后重新运行 head 及必要的 query-conditioned representation，不交换已融合 slots。
7. action/object edits、query 长度、视频长度分层；GT label 来源及人工核验覆盖。
8. raw localization 输出一致性；只验证 frozen 定位链，不宣称新 adapter 的 attention 是定位 GT。

Shuffled-video 的原标签结果是输入依赖诊断，不是新视频正确标签下的准确率；随机 shuffle 后不存在“应当恰好 0.5”的要求。

## 6. 成功、停止与解释边界

共同推进门槛建议为 Novel macro 至少 +2 pp、3/4 folds 正；Seen 下降不超过 1 pp；条件与 crossed 指标有一致改善。门槛尚未执行冻结，最终按共同协议统一；它们不是预期效果。

即使达 pooled 门槛，若同 query 或 crossed 指标不改善，不能升级成“学到 compatibility”。若仅个别语义 fold 提升，先分析支持不足或负例组成，而不是继续扩大 raw head。

若 R 在训练/Seen 都无法拟合，先诊断投影、mask、梯度与优化。只有 basic competence 正常的 null result 才可弱化便宜旁路假设。若容量/预算敏感，贡献应写成工程适配发现，而不是 representation 的单因素因果定位。

## 7. 结果记录规范

未来每 run 记录 feature/checkpoint hash、训练样本与 fold 身份、projection 参数数、token 长度、forward 次数、runtime、选模规则和逐样本 full-precision logits。paired/crossed tests 对共享视频聚类计算 uncertainty，报告 ties；不得把多个共享 quartet 当作独立样本。

本目录当前没有 run、checkpoint、result table 或 GPU launch 文件。下一步由用户另行授权实施；本轮仅完成研究说明。
