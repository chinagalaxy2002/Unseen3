# Idea 10：SHINE → GMR 单 seed 效果优先迁移

> 最新状态：当前A1/C1三模型50epoch实验和v2-v4均完成，已生成[综合评估](records/final_assessment/RESULTS.md)。QD有小幅正向趋势；Moment纯coarse/fine平均Gap未缩小；Flash C1明显负向。低BCE=.01无pair的10epoch方案出现候选收益，但A1 Seen有损伤、U区间跨0。没有新训练任务运行；下文保留各阶段历史。

状态：`implemented_pilot_completed`。2026-10-04。已完成独立代码、文本特征、验证与单 seed 两折训练。

目标是提高 semantic-novelty 下的 GMR existence AUROC，保留当前 Moment-DETR-GMR 的 CLIP+SlowFast、模型、existence head 与输出接口，迁入 SHINE 的 coarse/fine saliency ranking，并连接最终 existence logit。

按用户最新指令采用两个条件：batch 轮换 query 作为 absent negative；强制编辑层级距离链有效。这是用户确认的实验前提，不是本轮独立审核结论。仅 seed 3407，不进行多 seed。首轮 A1_action_01 / C1_composition_01 两折，每折比较同起点 baseline 微调与 SHINE 联合训练，均 10 epochs、lr=1e-5、Seen-val AUROC 选模。正式 U 不参与。

- [IDEA](IDEA.md)：目标、机制和论文逻辑。
- [MIGRATION_PLAN](MIGRATION_PLAN.md)：源码差异与已经落地的文件。
- [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)：首轮配置与评价。
- [RELATED_WORK](RELATED_WORK.md)：上游来源与贡献边界。
- [训练程序](code/train_pilot.py)、[损失实现](code/shine_losses.py)、[query bank 与特征准备](code/prepare_bank.py)。

上游 [zxccade/SHINE](https://github.com/zxccade/SHINE) 固定 commit `dfbaab1cf8f6f88ca3c3a39bee2bbe8dd999f466`。完整本地 vendor 在本目录，所有可修改代码、数据、特征、运行与记录独立存放。结果以 [PILOT_RESULTS](records/PILOT_RESULTS.md)、[RESULTS.json](records/RESULTS.json) 和每个 run 的 `metrics.json/status.json` 为准。

## 重现入口

在 Unseen3 根目录运行；下面展示 A1，一次只使用 seed 3407。C1 改 `--fold C1_composition_01 --device cuda:1`。已有 run 拒绝覆盖，重跑指定新 `--run-id`。

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python -m pip install --target research_ideas/10_shine_absence_aware_transfer/code/_deps -r research_ideas/10_shine_absence_aware_transfer/code/requirements-text.txt
/home/guoxiangyu/miniconda3/envs/owvtg/bin/python research_ideas/10_shine_absence_aware_transfer/code/prepare_bank.py prepare
/home/guoxiangyu/miniconda3/envs/gmr/bin/python research_ideas/10_shine_absence_aware_transfer/code/prepare_bank.py encode
/home/guoxiangyu/miniconda3/envs/gmr/bin/python research_ideas/10_shine_absence_aware_transfer/code/check_transfer.py
/home/guoxiangyu/miniconda3/envs/gmr/bin/python research_ideas/10_shine_absence_aware_transfer/code/train_pilot.py --fold A1_action_01 --device cuda:0 --run-id pilot_v2
```

两折完成后用 `code/summarize_results.py --run-id <run_id>` 生成共同结果。此入口依赖本机只读数据/权重路径与现有 gmr/owvtg 环境，环境记录见 [ENVIRONMENT](records/ENVIRONMENT.md)。原数据角色和 source hashes 见 records 内 manifests；单目录代码不从共享模型 import。

## 本轮效果

| Fold | 微调 baseline Novel AUROC | SHINE Novel AUROC | 增量 |
|---|---:|---:|---:|
| A1_action_01 | 0.6122 | 0.6541 | +4.18 pp |
| C1_composition_01 | 0.5455 | 0.5813 | +3.58 pp |

Novel-dev macro +3.88 pp，Seen macro +1.32 pp。定位未提高，same-query 排序变化不一致；这是单 seed 两折的 pooled 效果筛查，非正式 U 或跨 backbone 结论。训练均完成，checkpoint、全精度预测和 loss history 保存在本目录 runs。

## 正式 Unseen 比较（历史 50 epochs）

用户明确目标为 COMMON_PROTOCOL 的五个正式 split 的 Seen→Unseen 退化缓解。当前已准备 canonical→baseline50 / SHINE50 对照，seed 3407，无多 seed；方案见 [FORMAL_EXPERIMENT_PLAN](FORMAL_EXPERIMENT_PLAN.md)，固定参数与来源见 [FORMAL_FREEZE](configs/FORMAL_FREEZE.json)。只按 Seen-val 选 epoch，每个待评测 split 的两 arms 均训练完成、按 Seen-val 选模并冻结后才评测正式 U。上文 Novel-dev +3.88 pp 是 pilot，不是正式 Unseen 提升。用户已取消五 split 收尾优先级，当前聚焦 A1/C1 诊断与修复。A1/C1 正式评测已完成，当前未观察到退化缓解：相对 baseline50，两 split macro Unseen −0.67 pp、Seen −4.06 pp；Gap 缩小主要来自 Seen 损失。详见 [A1/C1 正式结果](records/formal/A1_C1_FORMAL_RESULTS.md)。

A1/C1 的训练曲线、局部梯度、分层负例与条件排序分析见 [未改善原因分析](records/formal/A1_C1_FAILURE_ANALYSIS.md)。初始 train-only 梯度诊断优先指向额外 rotated-negative BCE 的强度与主任务冲突，后续 v2/v3 消融已支持 existence 辅助监督造成损伤；原历史冻结配方保持不变。

## 清空上下文后的恢复入口

首先阅读 [完整交接文档](records/formal/WORK_HANDOFF.md)，再按 [下一轮任务与可复制提示](records/formal/NEXT_ACTIONS.md) 执行。诊断证据见 [A1/C1 未改善原因](records/formal/A1_C1_FAILURE_ANALYSIS.md)，运行快照见 [HANDOFF_STATE](records/formal/HANDOFF_STATE.json)；恢复时重新读取实时 status，不能按快照重启任务。v2/v3 消融与正式探索性评测已完成，v4 修复开发正在运行；恢复时查看最新交接顶部和 v4 实时状态。

## A1/C1 分项实验与当前修复阶段

v2/v3 各方案均为 canonical 初始化、seed3407、10 epochs、Seen-val-only 选模。独立[正式探索性结果](records/ablation_test_v2_v3/RESULTS.md)：saliency-only 相对同预算 B0 的 A1/C1 Unseen 分别 −.40/+1.16 pp，Seen 约0/+.52 pp；两折 macro ΔU +.38 pp、ΔSeen +.26 pp，未形成两折一致改善。Pair-only/weak BCE 联合虽有小幅 macro U 增量，但 Seen 分别损伤1.92/2.49 pp，不能判定修复成功。条件排序与定位结果也不一致，详见[评估与下一步](records/ablation_test_v2_v3/ASSESSMENT_NEXT_STAGE.md)。

已启动 v4：保留 coarse/fine，分别使用 BCE/pair 有效权重 .01/0、0/.02、.01/.02；两卡各三任务、10 epochs、seed3407、Seen-only。权重在新 Unseen 评测前依据 v3 Seen 结果预先固定。见[方案](configs/ABLATION_V4_PLAN.md)与[运行入口](records/ablation_v4/README.md)。

## 当前：coarse/fine-only 的 50-epoch 验证

用户要求进一步检验训练预算。10epoch匹配比较的Gap：A1扩大.40pp，C1缩小.64pp，两折macro缩小.12pp（相对.47%），缓解有限。已启动[独立saliency_v5](records/saliency_v5/README.md)：同canonical初始化、seed3407、总50epochs，B0/S1匹配forward与exposure，各卡两任务；完成后自动冻结选定checkpoint并评测正式探索性Seen/U及Gap。旧v4六组已完成。新结果不会以Seen下降带来的Gap缩小冒充成功。
