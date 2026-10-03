# Moment-DETR-TRM-GMR-Joint-v1 实验数据与脚本文档

本目录记录 **Moment-DETR-TRM-GMR-Joint-v1** 端到端联合模型在 Charades-STA 语义新颖性与事件存在基准（`semantic_existence_v2`）上全部 5 个划分（`A1`, `A2_alt`, `A3`, `C1`, `C2_alt`）的实验脚本、客观结果与数据汇总。

---

## 1. 对应实验脚本清单与执行方式

| 脚本文件 | 脚本定位 | 主要功能与调用方式 |
| :--- | :--- | :--- |
| `scripts/train_trm_gmr_joint.sh` | 单划分训练脚本 | 启动指定划分的端到端联合训练（默认 100 轮，学习率 1e-4，无测试集泄露）。<br>`bash scripts/train_trm_gmr_joint.sh <SPLIT>`（例如 `A1`） |
| `scripts/infer_trm_gmr_joint.sh` | 单划分推理评测脚本 | 加载 Seen 验证集最佳权重，在 Seen 验证集标定存在性阈值，并在测试集四象限（$S^+, S^-, U^+, U^-$）上完成全量评测并输出指标。<br>`bash scripts/infer_trm_gmr_joint.sh <SPLIT>` |
| `scripts/run_joint_experiment_a1.sh` | A1 独立全流程脚本 | 自动化运行 A1 划分的自检、训练、最优权重重命名、阈值标定与推理评测全流程。<br>`bash scripts/run_joint_experiment_a1.sh` |
| `scripts/start_multisplit_parallel_joint.sh` | 双卡并发队列启动脚本 | 初始化实验任务队列（`A2_alt`, `A3`, `C1`, `C2_alt`），调度双卡后台并发运行。<br>`bash scripts/start_multisplit_parallel_joint.sh` |
| `scripts/queue_worker_joint.sh` | 队列工作进程脚本 | 由调度器调用，分别绑定 GPU 0 / GPU 1 自动认领划分任务并执行完整的训练与推理。<br>`bash scripts/queue_worker_joint.sh <GPU_ID>` |
| `scripts/tmux_runner_gpu0.sh` | tmux 会话守护脚本 | 用于在 tmux 守护会话中稳定执行后台流水线任务。<br>`bash scripts/tmux_runner_gpu0.sh` |
| `scripts/aggregate_multisplit_joint.py` | 多划分结果聚合脚本 | 读取各划分输出的 `joint_summary.json`，计算宏平均指标，并输出聚合报告与 JSON。<br>`python3 scripts/aggregate_multisplit_joint.py` |
| `scripts/smoke_test_trm_gmr_joint.py` | 冒烟与端到端梯度自检脚本 | 7 步单元自检（前向推理、梯度回传验证、L_exist 反向传播至 Transformer 验证、单批次拟合测试）。<br>`python3 scripts/smoke_test_trm_gmr_joint.py` |

---

## 2. 对应实验结果文件与目录结构

所有实验结果均按划分独立归档，并通过聚合脚本汇总：

```text
experiments/trm_gmr_joint_v1/
├── README.md                     # 本文档（实验脚本与结果索引）
├── EXPERIMENT_FREEZE.json        # 实验协议、数据范围与超参数冻结记录
├── METHOD_SPEC.md                # 联合模型方法与架构定义文档
├── MULTI_SPLIT_RESULT.md         # 5 划分完整指标对比 Markdown 表格
└── multi_split_summary.json      # 5 划分机器可读格式综合汇总指标

results/moment_detr_trm_gmr_joint/
├── A1/                           # Action holdout: put, take
│   ├── joint_summary.json        # 本划分核心汇总（AUROC, R1@0.5, 拒识率, 配对准确率）
│   ├── official_test_metrics.json# 官方测试集全量指标
│   ├── threshold_frozen.json     # 在 Seen 验证集上无泄露标定的固定阈值
│   ├── resolved_config.json      # 训练与推理全量超参数配置
│   ├── diagnostics.json          # 详细诊断与四象限统计
│   └── training_meta.json        # 最佳 epoch 与验证集指标记录
├── A2_alt/                       # Action holdout: open, close, pour, cook
├── A3/                           # Action holdout: sit, stand, walk, run
├── C1/                           # Composition holdout: sit + bed/chair/couch
└── C2_alt/                       # Composition holdout: open/close + box/cabinet
```

*(注：`.ckpt` 权重与大体积 `.jsonl` 预测文件保留在本地机器以避免仓库膨胀，所有指标 JSON 与配置均已完整上传至代码仓库)*

---

## 3. 客观实验数据汇总

### 3.1 存在性 AUROC 指标对比 (vs Moment-DETR-GMR Baseline)

| 划分 (Split) | Baseline Seen AUROC | Baseline Unseen AUROC | Baseline Gap | Joint Seen AUROC | Joint Unseen AUROC | Joint Gap | Gap 变化 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 0.8044 | 0.4973 | 0.3071 | 0.8131 | 0.4625 | 0.3506 | +0.0435 |
| **A2_alt** | 0.7690 | 0.5511 | 0.2180 | 0.7578 | 0.5061 | 0.2517 | +0.0337 |
| **A3** | 0.7488 | 0.5643 | 0.1845 | 0.7410 | 0.6713 | 0.0696 | -0.1149 |
| **C1** | 0.7610 | 0.5621 | 0.1989 | 0.7429 | 0.6025 | 0.1404 | -0.0585 |
| **C2_alt** | 0.6759 | 0.4687 | 0.2073 | 0.6945 | 0.4507 | 0.2438 | +0.0365 |
| **Macro Avg (宏平均)** | **0.7518** | **0.5287** | **0.2232** | **0.7499** | **0.5386** | **0.2112** | **-0.0119** |

---

### 3.2 定位能力对比 (vs Moment-DETR-TRM Phase 1 Baseline)

| 划分 (Split) | TRM Seen R1@0.5 | TRM Unseen R1@0.5 | TRM Gap | Joint Seen R1@0.5 | Joint Unseen R1@0.5 | Joint Gap | Unseen R1 变化 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 43.87% | 23.01% | 20.86% | 40.62% | 26.02% | 14.60% | +3.01% |
| **A2_alt** | 36.38% | 25.60% | 10.79% | 38.46% | 27.98% | 10.48% | +2.38% |
| **A3** | 36.85% | 51.56% | -14.71% | 35.31% | 51.04% | -15.73% | -0.52% |
| **C1** | 37.01% | 47.53% | -10.52% | 36.55% | 53.70% | -17.15% | +6.17% |
| **C2_alt** | 37.95% | 36.21% | 1.74% | 38.23% | 34.78% | 3.45% | -1.43% |
| **Macro Avg (宏平均)** | **38.41%** | **36.78%** | **1.63%** | **37.83%** | **38.70%** | **-0.87%** | **+1.92%** |

---

### 3.3 门控、误拒识与同视频配对判别准确率 (Robustness & Discrimination)

| 划分 (Split) | U+ Raw R1@0.5 | U+ Soft-Gated R1@0.5 | U+ Hard-Gated R1@0.5 | U+ 误拒率 (FRR) | U- 拒识率 (RR) | 最佳 Epoch | 标定阈值 $\tau$ | 同视频成对判别率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 25.59% | 26.02% | 21.72% | 11.83% | 10.90% | 7 | 0.7642 | 59.78% |
| **A2_alt** | 27.38% | 27.98% | 27.38% | 0.00% | 0.00% | 50 | 0.9809 | 51.90% |
| **A3** | 51.56% | 51.04% | 48.44% | 6.77% | 15.82% | 35 | 0.9299 | 47.29% |
| **C1** | 53.70% | 53.70% | 49.38% | 6.79% | 12.22% | 14 | 0.9329 | 68.75% |
| **C2_alt** | 36.52% | 34.78% | 0.87% | 95.65% | 97.64% | 10 | 0.5484 | 81.82% |
