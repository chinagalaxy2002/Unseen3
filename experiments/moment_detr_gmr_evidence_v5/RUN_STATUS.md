# V5 首批实验运行记录

启动时间：2026-10-03 20:14，Asia/Shanghai。阶段：内层 baseline。本文记录启动与依赖；动态进度以队列 JSON 和 `scripts/status_evidence_v5.py` 输出为准。

## 已完成

- 修正 V4 raw localization 的候选排序为 binary foreground margin（softmax 等价排序），通过反例和随机 logits 对照检查。
- 从已有 V4 predictions 生成独立 localization revision，未改历史预测和 existence AUROC。
- 五正式 split 导出 Seen-only source-pair manifests，各 1499 条合格 train pair、1 条源样本不在允许集合的负例。
- 四内层 fold 按 Seen metadata 生成；全部角色无视频重叠；训练 query 及负例 source 不暴露各自留出语义。
- 四 fold train/Seen-val/novel-dev 全部特征覆盖完整。novel-dev 不参与 baseline checkpoint selection。
- 保存 `INNER_BASELINE_FREEZE.json`、输入审计、resolved configs 和代码/数据哈希。

## 实际任务

| GPU | Fold | 留出语义 | Train S+/S− | Seen val rows | Novel dev +/− | 队列 |
|---|---|---|---:|---:|---:|---|
| 0 | A1_action_01 | open | 5920/691 | 625 | 150/185 | 首个任务 |
| 0 | A1_action_02 | close | 6408/999 | 714 | 76/117 | 自动接续 |
| 1 | C1_composition_01 | close::door、open::door | 8046/1119 | 943 | 106/96 | 首个任务 |
| 1 | C1_composition_02 | open::cabinet、open::refrigerator、sit::sofa、open::box | 8601/1437 | 946 | 50/94 | 自动接续 |

每任务随机初始化，沿用 canonical checkpoint 保存的配置（只读取配置，不加载 trained model weights），seed 3407、100 epochs、关闭早停，按 inner Seen-val MR-full-mAP 选 checkpoint。主 loader 为自然分布 shuffle，包含 fold 内全部训练 rows。

已确认首批两个任务完成完整 epoch 的训练与验证并保存 `best.ckpt`。这是运行健康证据，尚不能据早期 Seen mAP 判断语义泛化收益。

动态检查命令：

```bash
python scripts/status_evidence_v5.py
```

所有任务日志/checkpoint 位于 `results/moment_detr_gmr_evidence_v5/inner_baselines/<fold>/`。队列状态为本目录 `baseline_queue_gpu0.json` 和 `baseline_queue_gpu1.json`；每任务最终成功/失败另记录在其 `status.json`。

## 时间映射审计与阶段边界

检查 A1/C1 Seen 覆盖的 4918 个视频：CLIP/SlowFast 长度差最多 1 个 clip，无 max_v_l 截断，文件均无嵌入时间戳。按 checkpoint clip_length=1 的名义特征末端与视频时长差在约 −0.499 至 +0.500 秒之间。

有 532 个视频的 GT end 超过名义末端；这通常与上述取整差异相关，不能直接解释成原始视频片段缺失。原 GT 归一化使用 T×clip_length，预测缩放使用 duration；本轮 baseline 保持原协议。

因此 P0 的 local ROI 时间映射 gate 尚未通过。可独立启动遵循原协议的内层 baseline；局部 verifier/GT ROI 辅助实验需要先核实抽取时间轴或明确记录可验证的近似映射，再通过对应 gate。完整时间审计见 `time_grid_audit.json`。

## 后续依赖

当前队列仅自动执行四个 inner baselines；不自动启动尚未实现/验证的 verifier 或正式 U 评测。内层 baseline 完成后，使用其 best checkpoint 提取 query-specific pooled/slots bank，再开始统一 BCE 的读取对照。局部视频分支另受时间映射 gate 约束。

正式五 split 训练仍需通过工作方案的内层收益、Seen 保持及定位一致性门槛。
