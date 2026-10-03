# V5 实施清单

状态：四个内层 baseline 完成，P3 decoder 读取对照运行中。完整定义见 [工作方案](WORK_PLAN.md)，首批实际产物见 [运行记录](RUN_STATUS.md)，后续见 [P3 执行记录](P3_RUN_STATUS.md)。

## P0：评测可信

- [x] 固定 foreground/background index、softmax 等价 margin 排序和首 slot ties 规则；输入反例与随机 logits 检查通过，见 `input_verification.json`。
- [ ] 固定 raw/postprocessed/official soft gate/hard diagnostic 的输出 schema 与单位。
- [ ] 完成 feature clip grid、duration、截断和 GT ROI 时间映射审计。
- [ ] 写入必要 evaluator 回归检查并通过。
- [ ] 统一全精度 baseline；保存 qid/feature/label/hash 审计。
- [x] 从已有预测生成 V4 localization revision，保留历史文件；见 `localization_revision_01.json` 和每 split 的 `metrics_revision_01.json`。
- [ ] 更新 V3 5/5 结果和根 README 的版本索引。
- [ ] G0 记录通过，附 `evaluation_audit.json`。

## P1：数据与缓存

- [x] 导出 Seen-only train/val source pairs，检查 source label、同视频和语义资格；见 `source_pair_audit.json`。
- [x] 正式主流数据未删缺源负例，辅助 pair 记录排除原因；内层训练额外按语义暴露规则排除无法验证源语义的负例。
- [ ] 建立 vid 去重视频缓存与 query-specific slots/tokens 索引。
- [ ] 加入 cache identity，不凭文件存在直接复用。
- [ ] 测量存储/内存开销；确认 masks、time mapping 与缓存精度。
- [ ] 审计主流全量覆盖、pair 覆盖、可用 ROI、特征缺失与 NaN/Inf。

## P2：内层开发协议

- [x] 冻结 Seen-only 数据角色和 fold 生成规则；见 `INNER_BASELINE_FREEZE.json`。
- [x] 生成 action/composition 各两个有效 fold，保存支持量与视频组；见 `inner_fold_index.json`。
- [x] 检查 source query 的语义暴露，阻止衍生样本跨角色；输入验证通过。
- [x] 确认 composition primitives 在内层训练有独立正例支持；生成器检查通过。
- [x] 从任务训练前随机初始化训练内层 baseline；每 fold 完成100 epochs，按 inner Seen val MR-mAP 选模。
- [x] 保存 fold/teacher hash、训练 counts、epoch 和选模记录；见 `INNER_BASELINE_COMPLETION.json`。
- [x] G1 内层 baseline 资格通过，冻结代码/语义/视频角色审计通过。

## P3：读取对照

- [ ] 完成 V4 pooled reference 及单因素 bound/anchor 诊断。
- [ ] 同一 BCE/训练预算下完成 R1–R4；R2→R3 隔离 slots 与 pooling。
- [ ] 完成 query-only/video-only 与候选容量匹配必要对照。
- [ ] 完成同视频 source-pair 和共享 GT ROI 诊断，注明非正式性能。
- [x] 独立比较真实样本 readout optimizer step 前后 baseline raw localization；误差为0，见 `bank_replay_verification.json`。局部 verifier 后续单独验证。
- [ ] 根据预定义分支选结构；保存所有尝试和失败，不只保存获胜者。

## P4：监督与门槛

- [ ] 完成 S0 BCE、S1 source-pair、S2 ROI 辅助对照。
- [ ] 仅满足预定义条件时开发 S3 global ranking。
- [ ] 完成有限 lr/容量/pair 权重开发；保存共同配置。
- [ ] 对候选方案完成三个 verifier seeds，分别报告 teacher seed。
- [ ] 记录 G2/G3；未达标时停止正式实验并写机制结论。
- [ ] 测量运行时间和显存，生成有实际数值的资源预算。

## P5：正式运行

- [ ] freeze code/config/data/features/baseline hashes 和开发决策。
- [ ] 固定五 split 共用设置与 Seen-only checkpoint/fallback/threshold 规则。
- [ ] 分离 train/select/calibrate 与 test U 入口，增加状态验证。
- [ ] 五 split 各完成 50 epochs，完整保存失败/fallback。
- [ ] 对正式 checkpoint 保存全精度预测、候选 evidence 和覆盖报告。
- [ ] 计算逐 split/macro AUROC、PairAcc/n、Seen、raw/official 定位与 edit 分层。
- [ ] 完成视频 cluster CI 和三 seed 报告，区分 seed/抽样不确定性。
- [ ] G4 与 final aggregate 完成，不丢弃退化 split。

## P6：研究收尾

- [ ] 对 H1–H5 分别写支持、反驳或未能区分的结论。
- [ ] 明确结果仍为 exploratory，避免 representation failure 的过强归因。
- [ ] 做完整近邻方法检索与新颖性核验。
- [ ] 如需确认性结论，另冻结新语义/数据协议。
