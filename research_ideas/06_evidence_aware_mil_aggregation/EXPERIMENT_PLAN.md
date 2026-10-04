# Idea 06 最小验证实验计划

状态：`draft_not_executed`。仅列未来实验设计，不启动 audit/推理/训练。共同规则见 [COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。

**独立实施入口：**先读本目录 [IDEA.md 第 0 节](IDEA.md)，其中完整记录研究目标、三个原始 GMR baseline 的真实代码路径、数据/特征/checkpoint/记录位置及本方向的复制清单。未来只改本 idea 内的代码工作副本，输出全部留在本目录；保持 A/B 原始模型、共享标注、旧结果及其他 ideas 不变。以下是未来实验设计，本轮没有复制代码或启动实验。

## 1. 前置 gate

必须先确定 candidate-level full-query scores 从哪里来。decoder foreground confidence、saliency、full-query compatibility logits不是同一变量，应分别记名，不能任意互换。若现有 cache 只有 pooled states/slots，首轮需要未来训练或推理以获得候选分数；当前文档不表示这些结果已存在。

同时核实 slot 数、valid mask、predicted spans、样本 ID。精确秒级 ROI 未核实不阻止 proposal/scope grouping 和 bag MIL，但阻止把 GT seconds 强制映射成 positive mask。

## 2. Stage 0：不改变 candidate score 的敏感性检查

未来在相同 logits 上测试 max、LSE、normalized LSE、精确去重后的 group-normalized LSE。max 等原有聚合的变化必须使用一套固定温度；不要在 Novel-dev 为每方法选最佳温度。

| 干预 | 是否可沿用原视频标签 | 能说明什么 |
|---|---|---|
| 完整 candidate set 精确复制 | 是，候选表征复制且原事件未变 | 检查计数性质；normalized LSE 本来就应不变 |
| 单个高分 candidate 精确复制 | 是，已知同一 candidate 重复 | max 与 group control 应不变；普通 normalized LSE未必 |
| 插入 padding 并标 invalid | 是 | mask implementation健全性，不是新方法贡献 |
| 候选子采样/删除 | 沿用标签只作敏感性，删掉支持后可能改变可见证据 | 不能直接计算真实新输入准确率 |
| 从别的视频加入“背景”candidate | 未核验时不可以 | 其他视频可能含 query；不能默认负标签 |

精确 duplicate trace 使用 identity/support 元数据，不能用 GT label。真实 overlap grouping 的成功不能从人工复制不变性直接推出。

## 3. Stage 1：监督×聚合因子分离

建议最小五臂，顺序运行即可，无需一次完整大网格：

| 臂 | candidate loss | aggregation | 解释目的 |
|---|---|---|---|
| A0 | bag BCE | normalized LSE | 当前 head 家族的基础 |
| A1 | bag BCE | group-normalized LSE | aggregation-only |
| A2 | bag BCE + verified-negative local loss | normalized LSE | supervision-only |
| A3 | 与 A2 相同 | group-normalized LSE | interaction/additive value |
| A4 | 与 A3 相同 | group score − nuisance offset | 最后单测背景/长度校正 |

每臂同 candidate feature、scorer 参数、样本 exposure、训练步数及 checkpoint 规则。另有 fixed-logits aggregation 比较隔离 aggregation数值性质；训练时改变 aggregation 会改变梯度，不能把它和纯后处理混为一谈。

如果预算只允许一次小训练，优先 A0 vs A2，判断完整-query 局部负证据是否有效，而不是直接上 A4。只有 Stage 0 显示候选冗余敏感性且 candidates 本身有可信 conditional competence 时，才优先 A1/A3。

## 4. 标签构造与 GT gate

Negative-local loss 只用于已有 full-video absent 真值的候选，且读取完整 query。未知 cross-video mismatches 不自动加入。S+ 正例首轮仅 bag BCE；不得为 positive video 的全序列置正，也不得把 GT 外全部置负。

未来恢复 feature extraction timestamps 且确认 annotation coverage 后，可增加单独 support-loss arm。该 arm 明确拥有额外 supervision；不得与没有同等支持的 baseline 比较后宣称仅 aggregation 更好。没有新增 mask GT 就不做 hypothetical oracle false-positive filtering。

## 5. 数据、选模与预算

四个 inner folds 与共同协议一致。训练 labels 仅 inner train；checkpoint 仅 inner Seen-validation。Novel-dev 用于固定 run 之间的开发比较，不能每 epoch 选最好结果。主 loader 覆盖全部自然样本，不重复 V3 的仅 paired-group 主采样。

先一 action、一 composition fold 的 fixed pilot；通过 basic competence 后补齐四 fold。建议 candidate head 尺寸与 V5 lightweight readouts接近；温度与 grouping 阈值仅在 Seen 开发，首轮各只一个预设值。成本低至中、1–2 GPU 可顺序跑小 head；准确 GPU-hour 需实施实测。

## 6. 必须报告

- 自然 Seen/Novel pooled AUROC；不能只报长度重采样后的数字。
- Same-video source PairAcc、same-query cross-video PairAcc/AUROC、crossed tests及覆盖。
- Cq、Cv、nuisance-only head，shuffled-video control。
- 真实事件数量、effective K、redundancy M、length、negative edit type 分层；支持不足的 strata不宣称稳定结论。
- copy sensitivity 与 fixed-K constant-shift sanity check。
- 哪些 grouping 合并了真实独立事件的人工审计；分组规则不看真值。
- 冻结定位输出一致性，candidate scores 的 full precision 与候选 ID。

换视频需重新产生 query-conditioned candidates。随机 shuffle 原标签 AUROC仅是输入依赖诊断，既不是新视频正确率，也不能规定它等于 0.5。

## 7. 成功与停止规则

共同建议门槛为 Novel macro +2 pp、至少3/4 folds正、Seen下降不超过1 pp，尚待实施前统一冻结。还应满足条件与 crossed指标改善、收益在真实视频下出现、候选冗余压力更稳。A3 必须比 A2 有独立价值，才支持 aggregation 机制。

若 candidates 在可信 same-query pairs 上不可分，停止 aggregation 扩展并转 representation/supervision。若只有 A2有效，保留 evidence supervision、放弃当前 grouping 新贡献。若 A4有效但条件指标无改善，限定为 nuisance/comparability 校正；不能称更强 event compatibility。若 Known-duplicate checks正确但 U AUROC不变，该性质仅是健全性保证。

## 8. 确认与审计记录

通过四 fold开发后才做多 seed、视频聚类区间、其他 backbone和新的确认数据。Shared candidates与视频上的多个 quartet不视为独立样本。记录 grouping定版、temperature、feature/ckpt identity、annotation coverage、valid masks与逐样本logits。原始五 split 已影响研究形成，标 exploratory。

当前没有命令、run files、results或假定已完成的oracle。实施另行授权。
