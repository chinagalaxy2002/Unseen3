# 01 · 最小验证与后续实验设计

状态：`draft_not_executed`。不包含已生成的数据、实现或结果。研究定义见 [IDEA.md](IDEA.md)，共同边界见 [COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## Phase 0：未来实施前的数据可识别性检查

只读已标注 inner train，建立 `(video, exact encoded query, Y)` 图。先核验同一 query 的正负支持，再找满足对角正、非对角负的完整矩阵。每格必须有已有标签及 provenance；禁止把缺少标注当 absence。输出未来 audit 表：总格数、独立矩阵/视频/query 数、连通分量、semantic/negative-type 覆盖、每 query support、标签冲突率、来源缺失与去重规则。

已有用户总结中的元数据计数可以用作工作量预估，不算新 audit 结果。稀有四格可能足够进行描述性 pilot，但不够估计稳定泛化。支持不足时用已核验 same-query pairs 做低成本分支，或交由 04 扩充；不自动补齐缺格。

Dev 同样只使用已有可靠标签闭合支持，不能根据新模型 score 挑选最有利矩阵。Inner Novel-dev 的小子集需报告实际视频/query 数；不能将它当作 formal U 的完整代表。

## Phase 1：冻结表示的小 head 对照

使用 V5 四个严格 inner baseline / feature banks。首轮选择相同 family 的简单 pooled head；可从 R1 容量级别起步，但必须记录实际参数，不能假定代码中某个既有名称自动满足匹配。所有 arms 同 seed、epoch rule、自然训练行、四格 batch 与 forward 次数；C0 是旧任务 reference，C1–C3 才是严格曝光匹配比较。

| Arm | 主流及额外曝光 | 辅助 ranking | 主要用途 |
|---|---|---|---|
| C0 | 全部 natural BCE | 无 | 确认新 head 本身表现及历史 replay |
| C1 | natural BCE + 同一四格的 aux BCE | 无 | 控制额外标签重复和 forward 预算 |
| C2 | 与 C1 完全相同 | 两个 row margins | 同视频条件排序，仍可能依赖 query prior |
| C3 | 与 C1 完全相同 | 四个 row/column margins | 最小双条件监督方法 |
| C4（仅 C3 有信号后） | 与 C1 完全相同 | smooth-max joint margins | 检查最差约束耦合是否超越均值双向排序 |

C2 也计算四格所有 scores，以匹配 C3 的 forward。四项/两项均按有效项平均，固定 scalar loss 权重；同时报告梯度范数，避免只是损失更大。C3 的四项平均与同格上的双向 pairwise objective 完全相同，不能把它们当两个不同 arms。

### 建议初始设置（尚未冻结）

- 50 个 head epochs、seed 3407，与 V5 开发粒度一致；optimizer 沿用统一 head recipe，不能每个 arm 单独搜学习率。
- `λe=0.1, λr=0.1, m=0.2` 仅作为 logit-space 起点；若训练梯度显示尺度不合适，允许 Seen-only 调整一次，并将所有相关 arms 同步重跑。C4 的 `τ=0.2` 仅为候选。
- Natural batches 完整遍历；quartet 辅助批按唯一矩阵采样，控制每 query 和每 video 重复上限。明确 aux 次数，不伪称自然主样本未删就完全没有权重变化。
- 一至两张 GPU 分 fold/arm 排队；不预估不存在的精确 GPU-hours。先测一个 fold 的时间和显存，再写预算。

Model checkpoint 由 inner Seen-val pooled AUC 决定，平局取最早。评估训练拟合仅用于检查机制可学性；不使用 Novel-dev 挑 epoch。保留 raw C1–C4 结果和 baseline fallback policy，不能把回退掩盖成方法成功。

## Phase 2：机制评价

每个 arm 报告 original pooled Seen/Novel AUROC、source-pair PairAcc、same-query PairAcc/query-macro AUC、quartet row/column/strict-group/D。训练 Cq/Cv 配套对照；在完全对称、相同编码的 quartet set 上检查其 AUC=0.5 和纯 query column / 纯 video row 必然 ties 的预期。

Shuffled-video 在原始输入前替换并重算。如果缓存没有覆盖这些组合，记录未来新增 forward 预算。随机错配的原标签 AUC 仅作依赖诊断，不能当新输入的真实准确率。

对 source matched pairs 的语言变化、actor/scene、object/action/composition negative-type 分层，检查是否只有一种类型贡献。把同场景 hard natural pairs 与容易跨场景 pairs 分开；模型只有 scene×query 交互也可能通过四格，因此需人工小 subset 或其他可靠事件核验进一步控制。

Pooled CI 按 video 同步 bootstrap，quartet 按共用协议处理共享节点。训练和评估 quartet 不按矩阵 ID 随机划分后假装独立；保留既有 fold 边界，另报告 video 重复带来的限制。

## 最小推进与停止规则

首轮最多 C0–C3 × 4 folds × 1 seed。若 C3 的训练条件目标没有可学性，先检查标签/表示，不继续大范围调参。若只一个 fold 有明显收益，报告不稳定，优先辨别其支持结构。若 Novel macro 与条件指标同向改善、Seen 保持，再追加两 seed，随后才增加 C4 或移植其他 backbone。

| 结果 | 下一步 | 可排除的解释边界 |
|---|---|---|
| C1≈C3，均优于 C0 | 检查额外标签曝光；发展 04 | 当前 ranking 没有超过同数据 BCE |
| C2 pooled 好但 column 不好，C3 两者都好 | 复核矩阵和跨 backbone | 双条件监督比单向更有解释力 |
| C3 条件好、pooled不升 | 02 reference-centered score | 单靠条件排序尚不足以解决主指标 |
| C3 train拟合、Novel不迁移 | 07 DG 或 03 witness | 当前条件关系仍可能 source-specific |
| C3 train不拟合、05能够拟合 | 新表征路线 | 原 pooled state 可能限制监督作用 |

不把 small subset CI 不显著等同于理论被否定。主推进参考 macro≥2 pp、≥3/4 positive、Seen下降≤1 pp 等共用建议；同时保留每折实际差值和置信区间。

## 后续贡献验证

若 pilot 支持：固定配置后在三个 backbone 验证 source coverage、存在性收益和 gated localization；比较 D-TSG/CroCs 式单向/双向 ranking 的适配，使用同一 verified support 与预算。将自然跨视频 negatives 和编辑 negatives 分开评价。最终报告方法贡献与数据/评价贡献各自的消融，不能只凭八个指标改善就宣称全新 loss。
