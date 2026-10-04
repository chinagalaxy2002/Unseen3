# Idea10：已完成实验综合评估

当前A1/C1三模型与v2/v3/v4实验均完成；三模型12个50epoch适配run和自动评测已完成。旧五split队列的未完成部分已按用户要求退出当前范围。本次补齐v4六个低权重10epoch方案的正式探索性评测。冻结输入、选模、checkpoint与预测校验记录见AUDIT.json及LOW_WEIGHT_EVALUATION_AUDIT.json。

| 方法 | 预算 | Macro ΔSeen | Macro ΔUnseen | Gap缩小量 | 判断 |
|---|---:|---:|---:|---:|---|
| Moment + coarse/fine | 50epochs | +0.46pp | +0.27pp | -0.19pp | A1退化、C1改善；平均Gap扩大 |
| QD-DETR + coarse/fine | 50epochs | +0.30pp | +0.65pp | +0.36pp | 两折正向点估计，但U区间均跨0 |
| FlashVTG + coarse/fine | 50epochs | +0.59pp | -2.45pp | -3.04pp | 负向，C1 U区间完全低于0 |
| Moment Low_BCE_only | 10epochs | -0.04pp | +0.75pp | +0.78pp | 单seed探索候选，U区间均跨0 |
| Moment Low_Pair_only | 10epochs | -0.15pp | +0.58pp | +0.72pp | 单seed探索候选，U区间均跨0 |
| Moment Low_BCE_pair | 10epochs | -0.29pp | +0.77pp | +1.06pp | 单seed探索候选，U区间均跨0 |

主表增量均相对对应模型、相同预算和forward的B0；10与50epoch不是同预算对照。正Gap缩小量表示退化减少，仍须结合Seen保持与ΔU。

## 三模型50epoch分项证据

| 模型 | Split | ΔSeen(pp) | ΔU(pp) | ΔU 95%CI(pp) | Gap缩小(pp) |
|---|---|---:|---:|---|---:|
| Moment | A1 | +0.41 | -0.61 | [-1.40,+0.19] | -1.02 |
| Moment | C1 | +0.52 | +1.16 | [+0.00,+2.31] | +0.64 |
| QD-DETR | A1 | +0.15 | +0.25 | [-0.82,+1.33] | +0.10 |
| QD-DETR | C1 | +0.44 | +1.05 | [-0.22,+2.29] | +0.61 |
| FlashVTG | A1 | -0.06 | -0.28 | [-1.12,+0.59] | -0.22 |
| FlashVTG | C1 | +1.24 | -4.62 | [-7.22,-2.06] | -5.86 |

## 结论与边界

1. coarse/fine-only不是当前GMR的通用退化缓解方案。QD-DETR保持Seen且两折U点估计均上升，macro Gap缩小.36pp（相对原B0 Gap约1.40%）；但两折U配对区间均跨0。Moment仅C1正向；Flash C1下降4.62pp，区间[−7.22,−2.06]pp，明显不支持普适收益。

2. QD的改善是相对同预算B0：S1 A1 U .5008仍低于canonical .5072；S1 macro Gap .2507仍略高于canonical .2467。不能写成所有原模型退化已消除。QD S1两折均选epoch1，说明本轮最佳Seen选择下50epoch未带来晚期最佳模型。

3. v4低权重BCE=.01、pair=0出现两折U正向点估计：A1 +.55pp、C1 +.94pp；平均U +.75pp、Seen −.04pp、Gap缩小.78pp（相对约3.12%）。但A1 Seen −.42pp，其配对区间完全低于0；C1 Seen +.35pp抵消了平均值。两折U区间均跨0，因此只能视为候选，不能称稳定且无损修复。联合低权重U +.77pp，但平均Seen −.29pp，Gap缩小部分来自Seen损失。

4. 原完整高权重existence方案在Moment上的失败和v2/v3/v4消融共同支持辅助强度会干扰主任务；并不等于轮换absent标签无效或编辑链关系错误。coarse/fine损失下降也不保证decoder existence收益。

5. 条件指标并不一致。三个模型S1的U source-pair accuracy在A1/C1都下降；same-query和raw定位有不同方向。QD shuffled-video U仍高于自然U；这是依赖诊断，不能直接证明完全不使用视频，也不足以宣称事件证据增强。

CI为固定训练seed的1000次视频聚类配对重采样，不含重训不确定性；A1/C1已多次探索，不能视为未触碰确认集。三个模型/两fold的描述性均值不作为独立样本统计证明。

## 后续建议（尚未启动）

优先对Moment低BCE=.01无pair候选做匹配50epoch对照，检查A1 Seen损伤是否扩大；继续标为探索性验证。若关注纯SHINE机制，在QD做coarse-only/fine-only拆分并补训练/开发条件控制。暂不扩大Flash完整方案；先诊断C1为何Seen增益与U损失同时发生。仅seed3407，不自动启动多seed或五split。

## 局部证据侧线补充

另一个已完成的Moment 10epoch侧线将saliency池化证据经零初始化残差分支接入existence。Local_CF相对B0平均U+.36pp、Seen+.28pp、Gap缩小.07pp；Uniform_CF近似，Local_noCF U−.02pp。没有观察到saliency加权分支相对原CF或uniform控制的额外收益。详见[侧线评测](../../local_evidence_transfer/records/evaluation_v1/RESULTS.md)。结构化RESULTS.json已纳入该实验原始汇总。
