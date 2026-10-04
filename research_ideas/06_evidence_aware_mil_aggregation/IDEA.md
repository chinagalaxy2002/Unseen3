# Idea 06：完整查询证据监督与相关性受控的候选聚合

状态：`draft_not_executed`。对应原研究文档 Idea 6。本轮只撰写 Markdown，未运行 candidate audit、训练、模型推理或结果选择。

## 0. 清空上下文后先读：研究目标、基线地图与独立实验边界

本节在每个 idea 中完整保留，单独打开本目录即可恢复背景。这里的代码、checkpoint 和数据都是已有资产；本轮只补充 Markdown，尚未复制实验代码或启动实验。下文的独立目录结构是未来实施约定。

### 0.1 我们到底在研究什么

我们研究 **GMR（Generalized Moment Retrieval）框架在 semantic novelty 下的存在性退化机制，并设计能够缓解这种退化的方法**。输入是完整视频 V 与自然语言 query Q；输出既要判断视频中是否存在满足 Q 的事件，也可定位其时间区间。存在性标签是 `Y(V,Q)=1[至少一个 moment 满足完整 Q]`，不是“视频里是否发生任意事件”。

Benchmark 为 Charades-STA 派生的 `semantic_existence_v2`。S+ / S- 分别是 Seen semantic 下的 query-event present / absent；U+ / U- 是 task-level held-out semantic 下的 present / absent。**U- 可以包含其他事件，只是 Q 所描述的事件不存在。** “Unseen”指本次下游训练未见，不保证 CLIP/SlowFast 预训练从未见过该概念。

| 已有 GMR baseline | 五 split 等权 Mean Seen AUROC | Mean Unseen AUROC | Mean Gap |
|---|---:|---:|---:|
| Moment-DETR-GMR | 0.7518 | 0.5287 | 0.2231 |
| FlashVTG-GMR | 0.7550 | 0.5479 | 0.2071 |
| QD-DETR-GMR | 0.7476 | 0.5144 | 0.2332 |

三个模型的五个 split 都 Seen > Unseen；这是单 seed、共享 benchmark 的共同观察，不是具体根因已成立的证明。主目标是提高 **`AUROC(U+ vs U-)`**，同时保住 Seen 和 localization；不能靠降低 Seen 缩小 gap。还要证明收益来自视频条件证据，而不是 text-only prior。必须保留 pooled AUROC、same-video PairAcc、same-query cross-video ranking、query-only / video-only、input-level shuffled-video 和可信 counterfactual / 四格 controls。

### 0.2 两个代码根目录，以及什么才是原始 baseline

- **原始 GMR 项目 A**：`/home/guoxiangyu/paper/Openword/generalized-moment-retrieval`。三个 GMR baseline 的模型、训练、配置，以及五个 split 的原始 checkpoint / 结果都在这里。本文的“原始 baseline”指这些已加入存在性适配的 GMR 版本，不是未加入 GMR 的上游 vanilla VMR 论文代码。
- **当前研究仓库 B**：`/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`。研究证据版本 SHA 为 `6cd96d723806e4b2b2474c68f289921eb543ffcc`；里面包含 Moment 和 Flash 的代码副本，以及后续研究变体。**B 没有 `models/qd_detr_gmr/`、`training/qd_detr_gmr/`、`configs/qd_detr_gmr/`，QD 必须从 A 获取。**
- 本轮读取到 A 的当前 HEAD 为 `bc88348e0e6b7b629a77d884fe66efe8c87b2774`。这是当前源树定位，不等于 15 次历史训练的源码版本证明；未来复制时仍须记录实际文件 hash、dirty 状态、checkpoint hash 和保存配置。
- 代码副本不能自动视为字节完全相同：本轮核对 Moment 的主模型 / head / train / dataset 相同，Flash 的主模型相同但 `train.py` 有差异。跨根目录混搭时需写明来源，不能将整个 B 默认为 A 的完全镜像。

**三个原始 baseline 的明确代码入口（以下均为 A 下的真实文件/目录）：**

| 模型 | 模型与存在性实现 | 数据 / 训练 / 推理目录 | 配置目录 |
|---|---|---|---|
| Moment-DETR-GMR | [moment_detr.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/models/moment_detr_gmr/moment_detr.py)、[gmr_adapter.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/models/moment_detr_gmr/gmr_adapter.py) | [training/moment_detr_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/training/moment_detr_gmr) | [configs/moment_detr_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/configs/moment_detr_gmr) |
| FlashVTG-GMR | [model.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/models/flash_vtg_gmr/model.py)（含 `exist_head`、`loss_exist`） | [training/flash_vtg_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/training/flash_vtg_gmr) | [configs/flash_vtg_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/configs/flash_vtg_gmr) |
| QD-DETR-GMR | [model.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/models/qd_detr_gmr/model.py)、[gmr_adapter.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/models/qd_detr_gmr/gmr_adapter.py) | [training/qd_detr_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/training/qd_detr_gmr) | [configs/qd_detr_gmr](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/configs/qd_detr_gmr) |

B 中可用副本的具体位置为 [models/moment_detr_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/models/moment_detr_gmr)、[training/moment_detr_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/training/moment_detr_gmr)、[configs/moment_detr_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/configs/moment_detr_gmr)，以及 [models/flash_vtg_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/models/flash_vtg_gmr)、[training/flash_vtg_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/training/flash_vtg_gmr)、[configs/flash_vtg_gmr](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/configs/flash_vtg_gmr)。

### 0.3 框架的文件分工与一次 forward 的含义

| 文件/组件 | 怎么读、负责什么 |
|---|---|
| 各 `training/<model>/dataset.py` | 读取 annotation JSONL、加载 qid 对应文本特征和 vid 对应视频特征、构造 masks / spans / existence labels；negative 必须保留空窗口样本，不能被 positive-only loader 过滤 |
| 各 `training/<model>/config.py` 与 `configs/<model>/` | 解析网络、特征、训练及输出目录；Moment/QD 通常是 base + feature + model + dataset YAML，Flash 还使用 `model.py` 的 nncore 配置 |
| 各 `models/<model>/` 完整包 | Backbone、位置编码、跨模态/时序模块、criterion 和内部 utils；Moment 的 transformer 为 `moment_transformer.py`，QD/Flash 为各自 `transformer.py` |
| Moment/QD 的 `gmr_adapter.py` | 对 decoder states 做 checkpoint 指定的 max/mean pooling，再经过 MLP 得到 `pred_exist_logits`；计算 existence BCE，并定义定位 score 的 existence gate |
| Flash 的 `model.py` 与 `blocks/` | 时序金字塔和候选预测；existence 使用 pooled query/video 表示拼接后 MLP；`blocks/` 的注册、generator 和 losses 也属于复制依赖 |
| 各 `train.py` | DataLoader、优化、训练日志、Seen-val checkpoint selection；Moment/QD 的 inference 入口是 `evaluate.py`，Flash 是 `inference.py`，并各有 `postprocessing.py` |
| Moment/QD 的 `training/<model>/standalone_eval/`；Flash 的 `models/flash_vtg_gmr/standalone_eval/` | Backbone 自带的 grounding/validation metric；也要随完整包复制，不能只复制主模型一个文件 |
| A 的 [scripts/analyze_semantic_existence.py](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/scripts/analyze_semantic_existence.py)；B 有同名诊断脚本 | 用 Seen-val 校准阈值，输出四象限 AUROC、matched PairAcc、raw/gated localization 和拒绝率；新 exact-query / 四格 / shuffle metrics 需要在本 idea 的副本扩展 |
| A 的 [eval/](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/eval)；B 的 [eval/](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/eval) | `eval_main.py` 是官方 GMR 评分入口，依赖 `metrics.py`、`normalization.py`、`utils.py`；官方全 test AUROC 不等于 Novel/U-only AUROC |

最小流程：`JSONL + 冻结 CLIP/SlowFast/text 特征 → dataset/collate → video-query backbone → spans / candidate scores + existence logit → 正例定位监督及 existence BCE → Seen-val 选模 → 全精度 score 的 pooled/conditional 评价`。标注中的 `exist_label` 应与可靠的 `relevant_windows` 空/非空语义一致；当前 Moment loader 实际由空/非空窗口生成 label，新增数据必须检查一致性。“没有标注”不能自动推出事件 absent。

### 0.4 数据、特征、checkpoint 与历史记录在哪

正式 split 为 `A1`（put/take）、`A2_alt`（drink/pour）、`A3`（run/walk）、`C1`（sit 与 bed/chair/couch 组合）、`C2_alt`（open/close 与 box/cabinet 组合）。这些是分别训练的语义保留划分，共享视频域，不是五个统计独立的数据域。

| 资产 | 真实位置 / 文件含义 |
|---|---|
| 正式标注根目录 | `/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2/<split>/`；A 的 `data/release/semantic_existence_v2/` 有仓库发布副本 |
| 每 split 的标签文件 | `train.jsonl`（只允许 S+/S- 训练）、`val.jsonl`（使用 Seen 子集选模）、`test.jsonl`（S+/S-/U+/U-）、`matched_u_pairs.jsonl`（同视频正负 query 配对；不是同 query 跨视频） |
| 数据来源与冻结 | 同目录 `manifest.json`、`statistics.json`、`review_report.json`、`semantic_inventory.json`、`split_spec_provenance.json`；记录构造、语义保留与审查来源 |
| Seen validation 视图与文本特征 | `A/features/semantic_existence_v2/<split>/val_seen.jsonl`、`A/features/semantic_existence_v2/<split>/clip_text/qid<qid>.npz` |
| 视频特征 | `/home/guoxiangyu/paper/新建文件夹/charades/vid_clip/` 与 `/home/guoxiangyu/paper/新建文件夹/charades/vid_slowfast/`；按 vid 加载 `.npz`。CLIP 512 + SlowFast 2304，通常拼接顺序 CLIP→SlowFast；TEF 另加 2 维，仍以保存配置为准 |
| 原始三模型运行与结果 | `A/results/semantic_existence/multi_split_v2/<split>/<model>/`（model 为 moment、flash 或 qd）；`run_metadata.txt`、训练/验证日志、逐 query predictions、`diagnostics.json`、`official_test_metrics.json` |
| 原始 checkpoint | Moment：`.../<split>/moment/best.ckpt`；QD：`.../<split>/qd/best.ckpt`；Flash：`.../<split>/flash/charadesSTA-*/model_best.ckpt`，在带运行时间的子目录内，不能假定为 `flash/best.ckpt` |
| 发布结果副本 / 原始交接 | [A/docs/semantic_existence_v2_metrics](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/docs/semantic_existence_v2_metrics)、[第二阶段交接](/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/docs/20260929_PHASE2_CURRENT_WORK_HANDOFF.md)；本地结果和大特征/权重不保证随 Git 发布 |
| 三 baseline 退化汇总 | [BASELINE_AUROC_DEGRADATION.md](/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen/BASELINE_AUROC_DEGRADATION.md)，指向 15 份原始 diagnostics |

JSONL 中重点读 `qid/query/vid/duration/relevant_windows/exist_label/partition`；`source_qid/construction_type/semantic_graph/verification_status/annotation_provenance` 用于配对与来源审计，不作为模型输入。Match 表必须保留正负 qid 与 video 身份，不能用 action group 替代 exact query。

**原训练协议和新 readout 协议分开：**历史三个 baseline 是 seed 3407、100 epochs、关闭早停，只用 Seen 训练与验证。Moment/QD 主要按 Seen-val MR-full-mAP 选模（QD 有保存代码中的回退规则）；Charades Flash 按 Seen-val `(R1@0.7+R1@0.5)/2`。本组 ideas 的冻结 head 开发默认按 inner Seen-val pooled AUROC 选模，不要把这个规则倒写成原 baseline 的历史协议。**默认 dataset YAML 仍可能指向 v1、30 epochs 或早停；复现 v2 要从原 checkpoint 保存配置/启动覆盖恢复。**阈值只影响 gate，不等于 AUROC 排序。

### 0.5 V1–V5 是历史诊断，哪些可以复用

V1–V5 不属于“原始三 baseline”。B 的 `models/moment_detr_trm_gmr_joint/`、`..._v2/`、`..._v3/` 是 V1/V2/V3 研究变体；`models/moment_detr_gmr_auc_v4/` 是 V4 residual；`models/moment_detr_gmr_evidence_v5/` 是 V5 readouts，分别有配套 training/configs/experiments。`models/flashvtg_dq_cgp_v3_gmr/`、`flashvtg_hs_dq_cgp_gmr/`、`flashvtg_dq-cgp-gmr-v2/` 是 Flash 的 DQ-CGP 系列变体；**DQ-CGP 不等于 QD-DETR**。它们都不能作为无改动 baseline 起点冒充对照。

V1 定位增强、V2 generic eventness、V3 semantic-group ranking、V4 pooled residual、V5 更多 readout 均未提供稳定解决主退化的证据。V5 Cq 的 inner Novel pooled AUC 四折均提高约 +3.16 pp，而 source-pair PairAcc 四折均下降，提示要认真区分先验与兼容性，不能直接认定 benchmark 有 shortcut。

| 可复用的 V5 资产（均在 B） | 用途 / 边界 |
|---|---|
| `experiments/moment_detr_gmr_evidence_v5/inner_fold_index.json`、`inner_folds/<fold>/manifest.json` | 四个严格 inner folds：A1_action_01、A1_action_02、C1_composition_01、C1_composition_02 |
| `inner_folds/<fold>/{inner_train,inner_seen_val,inner_novel_dev}.jsonl` 与 source pairs | 仅来自 formal Seen 的开发划分；Novel-dev 不是正式 U。Auxiliary 只用 inner train，Seen-val 选 epoch，Novel-dev 可有限开发但不进梯度/选 epoch |
| `results/moment_detr_gmr_evidence_v5/inner_baselines/<fold>/best.ckpt` | 重新从排除 holdout 的数据随机训练的 Moment baseline，适合严格 inner Novel；不能换为已经见过相应语义的 canonical checkpoint。当前没有可直接替换的 Flash/QD 同协议 inner baseline |
| `results/moment_detr_gmr_evidence_v5/banks/<fold>/<role>/` | `pooled.npy`、`slots.npy`、`base_logit.npy`、`query_tokens.npy`、`query_mask.npy`、`video_mean.npy`、`spans_seconds.npy`、`class_logits.npy` 等；按 metadata/hash/row identity 读取，不含可任意换 Q 后复用的表示，也不等于完整原始视频序列 |
| `training/moment_detr_gmr_evidence_v5/{protocol,feature_bank,metrics,train_readout,infer_readout,train_inner_baseline}.py` | 现有 split、特征缓存、选模与评价模板；只在 idea 内复制/改造。`protocol.py` 的 ROOT/RESULTS/EXPERIMENT 和部分绝对路径需本地化，不能直接运行旧队列写入 V5 目录 |
| `models/moment_detr_gmr_evidence_v5/readouts.py`、`experiments/.../READOUT_FREEZE.json`、`P3_READOUT_RESULTS.md`、`time_grid_audit.json` | Head/容量及旧结果对照；freeze 中的哈希绑定旧代码，新副本应建立自己的 provenance，保留 parent hash，不能覆盖旧 freeze。秒级 ROI extraction 时间来源尚未核实 |

### 0.6 未来如何在本 idea 内复制并独立运行

每个 idea 的未来实验根目录就是当前 `IDEA.md` 所在目录；可以整目录搬到其他位置。推荐保留相同布局，但**当前仅存在文档，下列 code/data/runs 等目录尚未因本任务创建**：

```text
<IDEA_ROOT>/
  IDEA.md / EXPERIMENT_PLAN.md / RELATED_WORK.md
  vendor/                 # 从 A/B 复制的不可修改源码快照及依赖清单
  code/                   # 本 idea 的可修改工作副本；只从这里 import
    models/<backbone>/    # 完整模型包，含 utils/blocks/注册代码
    training/<backbone>/  # 完整训练包，含 dataset/eval/config/postprocessing
    configs/<backbone>/   # 配置副本；保存输出路径覆写
    eval/ scripts/        # 实际使用的评分/诊断工具副本
  data/                   # 小标注副本、local manifests、pairs/quartets
  configs/                # 本 idea、每 split/fold 的 resolved config
  artifacts/              # 新缓存和 reference banks，绝不覆盖旧 V5 banks
  runs/<backbone>/<split_or_fold>/<arm>/seed<seed>/<run_id>/
  records/                # 来源/环境/数据/导入/变更范围与汇总记录
```

复制单位为完整 `models/<backbone>/`、`training/<backbone>/`、`configs/<backbone>/`，以及实际调用的 `eval/`、诊断脚本和相应评估依赖；保留父 package 层级。选择一种清晰来源：复现 canonical 三模型默认以 A 为源；需要沿用 V5 inner assets / helpers 时再显式从 B 复制。不要只复制单个 `model.py` 后通过 sys.path 导回原工程；本 idea 的 module 实际 `__file__` 必须指向 `<IDEA_ROOT>/code/`。模型/训练代码用普通物理复制，不用软链或硬链；如果需要本地 package initializer，只在本 idea 创建，避免 namespace package 混入原仓库。

先记录 vendor hashes，再建立 code 工作副本：baseline arm 使用无方法改动副本，method arms 只在当前 idea 修改。新配置可借原 checkpoint 的 `opt` / Flash run options 恢复，但必须明确重定位 annotation、feature、results、checkpoint、log、prediction、TensorBoard、tmp/cache 的全部路径。环境依赖以 A 的 `requirements.txt` 和实际运行环境导出为依据，在 records 内登记；这轮不创建新环境。

标注/小 metadata 用物理副本落入 `data/`，保留内容 hash 和父 manifest；**loader 可能写 `<data_path>.missing_features.jsonl`，所以即使数据只读语义，也不能把可写 sidecar 指向共享 release**。大视频特征、已有 checkpoint 可以作为只读输入引用；如选本地复制 checkpoint，则记录源/hash。输入目录只读约定不能代替输出路径隔离：任何重算、人工标签、pair/quartet/reference cache 和排查报告都写本 idea 内，不能改 release、A、B 的 models/training/configs、V1–V5 或其他 idea。

运行前在当前 idea 的副本检查 resolved paths / module origins / checkpoint identity；不从原仓库执行 train、prepare、extract、queue 脚本。有源码中的硬编码路径或清理逻辑时，只改副本并登记，不能修改其他项目来“兼容”本实验。同样的接口/导入错误可以在多个 idea 的副本分别修正，但不得自动把一个 idea 的方法改动同步给其他 idea。

### 0.7 数据与实验记录框架

以下文件名是将来产物的约定，没有在本轮伪造完成记录：

| 本 idea 内的记录 | 最少内容 |
|---|---|
| `records/SOURCE_MANIFEST.json` | A/B 的源路径、commit/dirty、每个复制文件 SHA256、checkpoint hash、parent freeze/manifest、copy time、baseline/method 来源区分 |
| `records/ENVIRONMENT.md` | Python/PyTorch/nncore/CUDA、conda/包版本、GPU 信息、随机种子；准确资源与时长只能运行后填 |
| `records/DATA_MANIFEST.json` | 每 split/fold/role 的标注源/本地路径/hash、query/video/label counts、特征顺序与归一化、source pairs、absence 审查、holdout 及 feature-cache identity |
| `records/ISOLATION_AUDIT.md` | 实际 import 文件路径；所有输出路径均在当前 idea；源代码/其他 ideas/旧结果未被修改；label sidecar 不落回共享 release |
| 每 run `resolved_config.json`、`status.json`、`train.log`、`history.json` | arm/loss/预算、训练/验证/选择规则、fallback 与 raw learned 区分、种子、运行状态及失败原因；当前只有 draft，不能写成 completed |
| 每 run `predictions.jsonl` / score arrays、`metrics.json` | row/qid/video/partition 对齐、完整精度存在性分数、raw/gated spans、pooled AUC、各条件指标和 Cq/Cv/shuffle；不能只保存四位小数 |
| `records/SUMMARY.md` 与失败案例 | 每 fold/seed/backbone/arm 差值、coverage、paired video CI、quartet 节点共享、Seen/localization 代价、confounds 与实际结论边界 |

自然主训练流保留全部可用 S+/S-；配对等是额外 auxiliary stream，不能为了 loss 丢弃不能配对的正例。Formal U 不进训练、bank mining、epoch/threshold/超参选择；历史 U 已看过，应标记 exploratory。数据支持不足、时间来源缺失或模型未具备条件判别时，依据本 idea 的证伪规则收窄结论，不能把目录隔离误认为实验本身已有效。

### 0.8 本 idea 的代码起点、改动位置与独立验证

**本方向先做什么：** 先检查候选完整-query evidence 的条件信号以及重复/长度敏感性，再做监督×聚合消融。V5 normalized LSE 已存在，固定 K 减 logK 不改 AUROC，不作为新机制。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment baseline 完整包、V5 `readouts.py` 与其 bank/metric helpers作为候选对照。读取正确 (V,Q) 的 slots/class/spans metadata；如果改用原始 index windows，复制 raw dataset 完整依赖并独立记录，不能默默换 candidate 定义。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/candidate_scorer.py`、`code/idea_method/redundancy_groups.py`、`code/idea_method/mil_aggregation.py` 及本地评估。Aggregation-only 固定 candidate logits；supervision-only 固定 aggregator；组合 arm 之后再单独研究 length/background correction。 |
| 本方法特别要记录的数据 | 保存 candidate IDs/support/masks、重复分组规则、每 bag 有效 K/M、精确复制压力测试、unique events 审计、candidate-negative provenance。一个完整 query 已认证 whole-video negative 才支持全部完整-query candidates 为负。 |
| Baseline 与机制对照 | 复制源码不等于复制 GT oracle：不能把正例全部 candidates 标正、把 GT 外皆标负，或把 phrase negative 当完整事件 negative。Grouping 不读标签，秒级 positive-support loss 等待 time gate。Normalized LSE 只对整集复制不变，对单 candidate 复制通常不变性不成立。 |
| 扩展到另外两个 backbone | 从 A 复制 QD/Flash 整包后分别定义 slots 或 temporal-pyramid candidates，披露差异；三模型原 existence representation 不相同，不强行声称共同使用相同 max/MIL formula。只在可验证候选证据存在时扩展。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 研究目标与优先级

目标是改善 semantic-novelty GMR 的 `AUROC(U+ vs U−)`，同时判断收益是否来自真实的视频条件证据。U− 中仍可出现大量其他事件；只有 query 描述的完整事件不存在。

本方向不能以“V5 已证明 pooling 是瓶颈”启动。V5 全 slots R3 相对 R2 只有一个 fold 为正，稳定 pooling bottleneck 证据不足。本方向优先级较低，应由以下诊断触发：candidate evidence 本身具有可靠的条件排序，但整体 score 对无效候选数量、重复相关性或视频长度异常敏感。

## 2. 核心假设与三个 baseline 的共同问题

- H06.1：presence 是“至少一个真实完整事件存在”，但某些聚合容易把多个重复、部分匹配或背景响应变成假高分。
- H06.2：新语义可能改变 background response 分布与 candidate correlation，导致 candidate-to-video score 的迁移不稳定。
- H06.3：训练时约束完整 query 的局部证据、推理时对冗余相关性控制，可能比无监督地扩大 readout 更可靠。

Moment-DETR/QD-DETR 当前 existence scalar 不必等于 max candidate confidence，FlashVTG 使用 pooled video/query。三个 baseline 共享的是“局部信息最终压缩成 scalar 的学习问题”，不是相同 MIL 公式。首轮从一个 backbone 开始验证；只有找到可迁移的机制，才讨论另外两模型的 candidate 定义与相应适配。

## 3. V1–V5 提供的约束

| 证据 | 支持提出什么 | 不能推出什么 |
|---|---|---|
| V1 correspondence/localization 不稳 | candidate 需要直接针对完整 query absence 训练 | 继续堆 phrase attention 就会解决 |
| V2 generic eventness 不稳 | candidate score 必须 query-conditioned | 有事件即存在当前 query |
| V3 采样和联合训练破坏定位 | 冻结主链，保留全部自然样本的 BCE | 一切 conditional ranking 无效 |
| V4 pooled residual 近零收益 | 回到局部 evidence→aggregate 链而非只改 scalar | 表示没有信息 |
| V5 R3/R2 不稳 | 先测 candidate competence 与聚合敏感性 | 已证明更多 slots 更好 |
| V5 Cq pooled 提升、PairAcc 下降 | 必须有条件排序、Cq 与 video controls | benchmark shortcut 已证实 |

[V5 readout 代码](../../models/moment_detr_gmr_evidence_v5/readouts.py) 已使用 `logsumexp(candidate)−log(K)`。因此 normalized LSE 本身不是新贡献，更不能把重跑这一公式称为改进 V5 的新机制。[研究证据总览](../../semantic_novelty_GMR_next_stage_research.md)

## 4. 首先区分三类聚合问题

1. **有意义的独立事件证据增加：** 视频里多次真实事件可提高存在性的可信度，不应为追求不变性把它们全部删掉。
2. **同一事件候选重复：** 重叠 proposals、复制 slots 不应被误当独立证据。
3. **无关候选或视频长度增加：** 真正 absent 视频的 false-match 极值可能随着候选规模变化；但长度与类别构成也可能相关，不能只拟合 length prior。

若所有样本固定 K，

\[
\log\sum_{i=1}^K e^{r_i}-\log K
\]

相对 unnormalized LSE 只是全体共同常数，**不改变 AUROC 排序**。若有效 K 随视频变化，归一化才可能改变排序，但同时改变真实证据稀疏性权重。复制整个 candidate set 时 normalized LSE 不变；只复制一个高分 candidate 时通常仍改变。因此不能把它称为普遍 duplication invariant。

Naive noisy-OR `1−∏(1−p_i)` 只有在合适独立性/概率假设下有特定解释。相关候选复制会人为提高输出；本方向不默认使用这种概率合并。

## 5. 输入、完整查询证据与标签边界

候选可来自冻结 decoder slots（带预测 spans）、或原始序列上固定 relative-index windows。每个 candidate `h_i(V,Q)` 与 full-query features 交互后产生 compatibility logit `r_i`；首轮不拆 phrase、不把某个动作或对象存在当整个 query 存在。

候选路径必须固定，以便对同一组分数比较 aggregation。预测 span 可用于 proposal overlap grouping，它是模型预测，不是 oracle。全序列 relative-index windows 不需要精确秒级 mapping，但不能当作 GT temporal localization。

训练标签有严格限制：

- 如果 full video 已核验 `y=0`，则其内所有真实候选都不满足完整 query，可用 candidate-negative supervision；这个结论不适用于 query 的所有独立 phrases。
- 如果 `y=1`，只有“至少一个候选为正”的 bag-level 标签；不能把全部 candidates 设正，也不能默认 GT 外皆负，除非事件时间标注确实穷尽所有发生位置。
- 有 GT support、且 feature-time provenance 核实后，可增加 positive-candidate support loss。首轮 provenance 未就绪时仅 MIL，不伪造 ROI/mask/完整事件 oracle。
- 无效 padding 不参与聚合或监督；特征复制是已知冗余的机制控制，不是新的负标签。

## 6. 具体机制：证据训练与冗余控制分别消融

设 `C={r_i,h_i,a_i}_{i=1}^K`，`a_i` 是 proposal 或 relative-index 支持。首轮的三类 baseline 都基于同一 candidate logits：max、LSE、normalized LSE。

发展版本用不读取真标签的 candidate grouping `G_1,…,G_M`。对精确复制候选，group identity 由原 candidate 身份/精确 support 决定，保证它们并入同组。对自然重叠 proposals，只能用预定 overlap 与 feature relation 规则做近似冗余分组；阈值按 Seen 规则确定，不能借 GT 改分组。高 overlap 也可能是不同事件，必须做 error audit。

\[
c_j=\max_{i\in G_j}r_i,
\qquad e_{group}=\tau\log\left({1\over M}\sum_{j=1}^M\exp(c_j/\tau)\right).
\]

精确复制若留在相同组且不改变 `c_j` 与 M，上式不变。这是公式性质；只能解释结构冗余，不意味着模型已学会视觉事件真值。group 内 max 是透明 pilot，未来可比较 learned representative，但不能一开始加入许多自由度。

可选 background-response 校正：

\[
e=e_{group}-b_\eta(\log M,\log T,\rho_C),
\]

其中 `ρC` 是预定 candidate redundancy summary。`bη` 不读取 query identity、action group 或 Novel 标签，在 inner train 的自然 BCE 下学习；首轮先测试 `b=0`，再单独增加。它是 nuisance-conditioned offset，不是已经校准的 null likelihood ratio，也仍可能学习 video/length prior。必须有仅 nuisance 的 Cv 对照，并报告长度分层。

基本 loss：

\[
\mathcal L=\operatorname{BCEWithLogits}(e,y)+
\lambda_-\mathbf 1[y=0]\,{1\over K}\sum_i\operatorname{softplus}(r_i).
\]

正例靠 bag BCE 选择至少一个候选；negative-local loss 压低所有完整-query candidates。与 baseline 对照需要相同正负采样与参数预算。若增加两向可信 pair loss，应在所有 aggregation arms 上一致增加，不把额外 supervision 的收益误归于聚合。

## 7. 为什么可能改善 U+ vs U− AUROC

若 U− 的高分来自重复部分匹配或背景候选极值，而 U+ 含可迁移的完整事件候选，本机制可能降低假高分并保留真实存在证据。candidate-negative supervision 提供了 scalar BCE 之外的 evidence constraint，group control 避免冗余产生不当证据累积。

这要求 candidate representations 已有分辨能力。若 feature 无法区分 novel action，任何聚合都可能无效；压低全部新语义分数则会伤害 U+。不存在仅调整 threshold 改善 AUROC 的途径。

## 8. 最小实验与成功判据

见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。先不训练做候选敏感性诊断；之后分别比较 aggregation-only、candidate-supervision-only、二者结合，最后才加 background offset。

推进需要自然 Novel pooled AUROC 与同 query/crossed 指标改善，同时候选冗余压力测试更稳定、Seen 与 frozen raw localization 保持。只在人工 candidate-copy 测试不变，或者只在长度再配平子集提高，不够支持主目标。候选复制控制实现了公式保证，也不能当成强独立 empirical contribution。

## 9. 失败后的解释

| 失败方式 | 可排除/弱化 | 不可排除 |
|---|---|---|
| candidate score 本身无条件区分能力 | aggregation-only 足以解决的解释 | 更好局部 evidence representation 或 supervision |
| 聚合敏感但更稳不提高 U AUROC | 已测冗余敏感性是主要退化原因 | 其他 candidate bias、其他语义机制 |
| supervision-only 有效，group 无新增 | 当前 aggregation 新机制必要性 | candidate evidence 监督方向价值 |
| group 有效但 Seen/U+ 下降 | 不加区分的冗余抑制可安全泛化 | 区分重复 proposals 与多个真事件的机制 |
| pooled 改善、同 query不变/变差 | 真兼容性增强的主张 | length/query mixture 校正收益 |

## 10. 最大 confound 与保留/放弃

最大 confound 是候选身份、数量、时间支持和局部监督同时改变。要保持 candidate logits 不变做 aggregation-only，再固定聚合做 supervision-only。其他混杂包括：feature 缺信息、有效 K 与标签相关、proposal grouping 合并真实重复事件、query prior 被 candidate scorer 复制、GT 不是 exhaustive annotation。

保留 V5 的读取对照、Cq/Cv、inner holdout、冻结主链与原始 logit；保留局部 evidence 尚未被否定的判断。放弃“full slots 一定更好”、固定 K 减 logK 可修 AUROC、把 correlated candidates 当独立 probabilities、从整句 negative 推导全部 phrase negative、无时间 provenance 的 oracle ROI。

## 11. 成本与论文故事

成本为低至中：aggregation audit 可用已存在可追溯候选输出；若候选 compatibility logits尚未缓存，未来需要推理或小 head 训练。无需大型 backbone 重训，但不能宣称零 GPU。完整 positive support 监督还需要 provenance 与标注审计，成本更高。

可发展的故事是“semantic shift 下存在性证据与候选多重比较的关系”：区别真正独立事件、重复 proposals 和背景响应，用 evidence constraints 与明确冗余结构提高新语义判别。normalized LSE 或普通 MIL 本身已有，本项目必须证明新增收益来自 candidate evidence/冗余机制并跨语义使用。[近邻与 novelty](RELATED_WORK.md)

## 12. 最强 reviewer objection

“这只是 MIL pooling/NMS/length correction，V5 已有 LSE；成功可能只是负例监督更强或 GT oracle。”应对需要监督×聚合分离、固定候选分数、无 oracle grouping、GT provenance 明确、条件评价、真实重复事件 preservation。若只获得 known-duplicate invariance，应将其视为健全性检查，不把它包装成足够的方法贡献。
