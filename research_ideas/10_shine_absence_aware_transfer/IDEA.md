# Idea 10：把 SHINE 的分层 saliency 监督迁入 GMR existence

状态：`implemented_pilot_completed`。本文已按 2026-10-04 最新授权更新：直接实施、训练、单 seed、效果优先。

## 0. 可独立恢复的背景

当前工程 `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`；原 GMR 工程 `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval`。任务 `Y(V,Q)=1[视频中至少一个 moment 满足完整 Q]`，U- 可包含其他事件，目标为 U+/U- existence AUROC。历史三个 backbone Seen→Unseen AUROC 为 Moment .7518→.5287、Flash .7550→.5479、QD .7476→.5144；这些不是本轮结果。

首轮只迁 Moment，复制当前 `models/moment_detr_gmr` 和 `training/moment_detr_gmr` 完整包到本目录 `code/`。QD 来源于原工程；Flash/DQ-CGP 与 QD 分开。四折定义在当前工程 `experiments/moment_detr_gmr_evidence_v5/inner_fold_index.json`，本轮优先 A1_action_01、C1_composition_01。初始化只读对应 `results/moment_detr_gmr_evidence_v5/inner_baselines/<fold>/best.ckpt`，不加载见过 inner 留出语义的 canonical 模型。

小标签复制到本目录 `data/<fold>/`。视频只读 `/home/guoxiangyu/paper/新建文件夹/charades/{vid_clip,vid_slowfast}`；原文本只读原工程 `features/semantic_existence_v2/{A1,C1}/clip_text`。SHINE 生成文本按 inner-train source 的 video+query 匹配，全部新 CLIP 特征保存本目录。所有输出和 loader sidecar 独立，imports 检查本地来源，formal U 不参与。

## 1. 核心假设与用户确定的条件

假设：对共享 video-query fusion 加入 SHINE 的时序语义排序，并对最终存在性分数施加轮换负例监督，能够改善新语义的全视频判别；这需要训练结果支持。

按用户明确确认，本轮将 batch 内循环轮换 query 当作 absent negative，并采用强制三层编辑距离链。记录为 user-attested experiment assumptions，不额外等待人工标签/层级核验。保留训练 split 边界、padding/null 安全和结果真实性要求。

## 2. 已实现机制

A：复用作者发布的三层 `recomposed_queries`。只匹配 inner-train 正例 source，不把上游全部训练样本加入 GMR。过滤原正式 split 与 inner 留出 action/composition，避免编辑引入目标语义；语义过滤基于本项目 spaCy parser，属于解析器层检查。允许未被指定留出的新词，作为 SHINE 数据扩增的一部分披露。

B：保留当前 saliency head，在原 positive GT 内取 topK 均值，约束其高于 GT 外 top1 和轮换 query 在同 GT 内 topK。Fine 复用上游 `div_loss` 与相邻 `red` 距离链：GT→原 query→level1→level2→level3→轮换 query。固定 .25 margin，teacher detach。每个自然 batch 最多抽四条完整链重新融合三层文本，避免显存失控；自然 batch 全部保留，auxiliary 缺链不删除原行。

C：原 GMR loss + coarse + fine + BCE(g(V,Qrot),0) + .2*[.2−g(V,Q+)+g(V,Qrot)]_+。coarse margins=[1,2]，q=8。原定位/foreground/存在性 loss 保持，fusion/head 联训，预计算编码器冻结。Hierarchical edits 只参与距离监督，不把所有 partial-positive 一律标 0。

推理只有原 V,Q；不使用 GT、编辑 query、source id、batch 同伴。训练 GT 沿用当前 GMR 的名义 clip_length=1 映射，不宣称已独立核验视频特征提取时间；这属于本轮与当前训练协议一致的适配限制，不阻止已授权的效果筛查。

## 3. 论文逻辑检查

候选定位 Technique，暂定贡献 B，继承 SHINE 核心思想。

| 环节 | 内容 |
|---|---|
| 背景 | GMR 同时需要定位 present 和拒绝 query-specific absent |
| 限制1 | 当前训练缺少分层 query 的时序响应约束 |
| 限制2 | saliency/localization 的改善未必落到 existence 排序 |
| Key idea | 在共享融合表示上联合训练 SHINE 层级 saliency 和最终 existence logit |
| 挑战1→模块B | 搬入层级距离且保留 null/padding 安全→独立重融合与 masked coarse/fine |
| 挑战2→模块C | 连接局部与最终全视频判别→轮换 existence BCE/ranking |
| 候选贡献1（方法节） | 完整代码级 GMR 迁移，属于可复用实现贡献 |
| 候选贡献2（实验节） | 单 seed 同起点 AUROC/定位效果及之后可做的 temporal/existence 消融；结果出来前不主张提升 |

限制→idea、idea→挑战、挑战→模块、模块→候选贡献四项设计对应检查通过；不等于科学假设成立。后续写论文前需要效果与机制消融，当前不做多 seed。

## 4. 与已有方向区分

03 的完整事件 binding 没有在这里实现；06 冻结候选/聚合与本轮共享融合时序 loss 不同；07 pair-only 是后续强消融；09 为固定特征 evidence increment 审计，本轮不修改其代码。若收益只由 rotated BCE/ranking 解释，不能称 SHINE temporal 是关键。若只定位升或 Seen 受损导致 gap 缩小，不能称解决 GMR 退化。

## 5. 实施结果（2026-10-04）

已完成 seed 3407 的两折 baseline/SHINE 十轮微调；Novel-dev macro +3.88 pp，Seen +1.32 pp。原始数据流未删行，formal U 未参与。具体逐折指标、条件排序和声明边界见 [PILOT_RESULTS](records/PILOT_RESULTS.md)。这支持 pooled 效果候选，不证明层级 saliency 是独立贡献。
