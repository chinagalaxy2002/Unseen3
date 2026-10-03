# P3 读取实验执行记录

日期：2026-10-03；完成状态于 2026-10-04 核验。状态：四个内层 baseline 完成，修订后的 P3 decoder 读取队列已完成全部 36 runs。实时状态用 `python scripts/status_evidence_v5.py` 查看；GitHub 文件仅为提交时快照。

## baseline 完成核验

核查冻结代码和 fold 文件哈希、checkpoint 哈希、实际日志、Seen mAP 最早最佳 epoch；全部通过。每个 baseline 均从随机初始化训练 100 epochs、验证 100 次。

| Fold | Best epoch（1-based） | Seen-val MR-full-mAP | 训练耗时 |
|---|---:|---:|---:|
| A1_action_01 | 4 | 26.69 | 2549 秒 |
| A1_action_02 | 7 | 25.62 | 2778 秒 |
| C1_composition_01 | 58 | 23.52 | 3616 秒 |
| C1_composition_02 | 18 | 24.90 | 3913 秒 |

这些定位指标不等于 existence AUROC。完成审计见 `INNER_BASELINE_COMPLETION.json`。

## P3 当前范围

| Variant | 输入/变化 | 可训练参数 |
|---|---|---:|
| R1 | pooled decoder state，独立 MLP | 50049 |
| R1_matched | 容量匹配 pooled MLP | 200433 |
| R2 | pooled state + query tokens | 199809 |
| R3 | 完整 slots + query tokens | 199809 |
| Cq | query-only | 83329 |
| Cv | video-only mean，按 vid 去重 | 382849 |
| D_v4 | 完整 V4 residual 架构和目标参考 | 17091 |
| D_no_bound | 仅去掉 residual bound | 17091 |
| D_no_anchor | 仅去掉 anchor | 17091 |

共 4 folds ×9 variants=36 runs；当前 seed=3407，每项 50 epochs，无 early stopping。R1/R2/R3/control 的主目标相同，为自然分布 BCE，每 epoch 完整遍历训练集。D_* 继承 V4 的 BCE/global/same-semantic/anchor 设置，batch256、无 gradient clipping；用于约束诊断，不与 BCE readout 直接做单因素归因。

R2 与 R3 模型参数完全一致，区别在于池化后的单向量或完整 slot set。R1_matched 用于排除读出容量解释。Cq/Cv 作为 shortcut 控制，容量不同需在解释中说明。

新模块通过 padding、slot permutation、zero-init、finite gradient、checkpoint replay 检查。真实正负样本上的 readout optimizer step 前后，独立 baseline 的 spans/class logits 完全相同；baseline 参数无梯度。测试见 `readout_verification.json`、`bank_replay_verification.json`。

## 缓存和选模

使用 fp32 mmap 保存 pooled、完整 slots、query tokens/masks、full-precision baseline logits、spans/class logits 及 video mean。视频源按唯一文件记录 checksum，原始视频特征仍通过 vid 对应的外部路径去重读取，当前尚未生成 local ROI 缓存。

缓存 identity 包括 checkpoint、annotation、manifest、extractor、dataset、源特征内容哈希、顺序、归一化与维度；训练读取时再次核查数组和 metadata 哈希。每 role 所有 qid 覆盖，不能因配对资格丢主流样本。

训练模块只打开 inner_train 和 inner_seen_val。每项按 Seen-val pooled AUROC 选最早最佳 checkpoint。standalone head 的 epoch 0 是随机初始化，不等于 baseline。若最佳 head 未超过 baseline，则部署策略回退 baseline；learned 模型自身结果仍完整报告。

独立 eval 入口先验证所选 checkpoint 和协议哈希，Seen-only 冻结阈值，然后打开 inner_novel_dev。正式 U 数据未进入本轮缓存、训练、选模或评测。

## video-only 修订

试启动期间发现，按 query batch 对相同视频做 mean reduction 会出现约 1e-7 的差异。虽然源视频相同，微小差异仍可能打破 PairAcc ties。

已停止并归档试启动，四个 baseline 保留；重新冻结并重启全部读取变体，避免混用试运行数据。修正为：每 role 同 vid 统一取首次缓存的 mean representation；推理每 vid 只计算一次，然后广播到该视频所有 query。真实训练 source pairs 的 video-only PairAcc 严格等于 0.5，已通过测试。

修订依据见 `READOUT_RESTART_RECORD.json`；旧协议、状态和试启动产物保存在 archive 中。修订不依赖正式 U 或局部 ROI 性能，试启动结果不用于选择当前参数。

## 时间映射与 R4 边界

继续检查原数据目录、可用脚本和特征来源说明后，确认它们是已发布的 CLIP/SlowFast 预计算特征，但没有找到本批特征逐 clip 时间戳或可独立核实的原提取记录。checkpoint clip_length=1 与长度审计不能证明每个特征的精确时间支持范围。

因此当前 freeze 明确排除 R4 和 GT ROI 辅助项。pooling/slots 读取对照可独立进行；这些实验不能用来否定原始局部视频证据。后续需恢复时间来源，或另行冻结有明确误差边界的近似映射与敏感性对照，再运行 R4。

## 运行与资源

- GPU0：两个 action folds；GPU1：两个 composition folds。
- 每设备一个持久队列，依次提取其 folds，再按冻结 variant 顺序 train/eval。
- 每设备当前阶段最大 wall budget 为 4 小时，总上限 8 个设备小时；超限停止并报告，不悄悄缩短 epochs。
- 每个任务保存 `history.json`、`status.json`、`best.ckpt`、阈值、全精度预测和 result。部分表自动汇总到 `P3_READOUT_RESULTS.md/json`，不控制正在运行的配置。
- 三 seed、source-pair loss 与正式五 split 尚未启动，G2 尚未通过。

入口：

```bash
python scripts/status_evidence_v5.py
python scripts/aggregate_evidence_v5_readouts.py
```

训练环境使用 `/home/guoxiangyu/miniconda3/envs/univtg/bin/python`。计划完成后按四 fold 的完整对照决定下一步，保留失败和 fallback；不能仅按早期单 fold 点估计修改剩余运行。

## 完成结果与阶段判断（2026-10-04）

两张 GPU 队列均 completed，全部 train/eval 任务 exit code=0；队列记录的 freeze 哈希与当前 READOUT_FREEZE.json 一致。36 个 runs 均完成 50 epochs，完整逐 fold 结果见 [P3_READOUT_RESULTS.md](P3_READOUT_RESULTS.md)。本次仅更新结果与记录，未修改模型、loss、冻结配置或选模规则。

R3 的 ΔNovel 分别为 A1_action_01 +6.48 pp、A1_action_02 +1.34 pp、C1_composition_01 −0.45 pp、C1_composition_02 −2.66 pp。Macro 为 +1.18 pp，达到 +1 pp 的效果目标，但仅 2/4 folds 为正，未达到至少 3/4 folds 为正的本轮推进条件，不能据此将 slot readout 定为后续主结构。

R3 相对等参数 R2 仅在 A1_action_01 提升，其余三个 folds 均下降；当前证据更符合以 A1_action_01 为主的 fold-specific 收益，尚不支持稳定的 pooling bottleneck 结论。Cq query-only 的 macro ΔNovel 为 +3.16 pp，4/4 folds 为正，需要在后续实验中解释 query 侧信号与潜在 shortcut；该结果本身不能证明 shortcut。

上述结果均为 seed-3407 的 Seen-only 内层开发结果。三 seed G2 尚未通过，source-pair supervision、正式五 split 和 R4 尚未启动。
