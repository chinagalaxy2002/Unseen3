# Idea 04：查询边际配平的辅助存在性训练

> 状态：`draft_not_executed`。仅为研究设计，未实施数据处理、训练或推理。  
> 对应原总结 Idea 3；目录编号表示下一阶段优先级，不是版本号。  
> 证据基线：`6cd96d723806e4b2b2474c68f289921eb543ffcc`，2026-10-04。  
> 来源：[下一阶段研究总结](../../semantic_novelty_GMR_next_stage_research.md)。关联：[实验计划](EXPERIMENT_PLAN.md)、[文献比较](RELATED_WORK.md)、[共同协议](../COMMON_PROTOCOL.md)。

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

**本方向先做什么：** 首轮保持小 existence head 和所有自然主训练行，只改变 exact-query 支持集里的辅助 label 权重。比较相同 query 分布/支持/曝光的原比例 M1 与 q 内正负各半 M2；这是数据监督干预。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment 的完整 baseline 包，或以 A 的同名完整包建立 canonical source；来源选定后不能混用。需要 V5 inner replay 时复制 B 的 bank/metric/readout helpers 及其本地依赖，禁用其中与本方法无关的 pair sampler/损失；inner annotation/manifest 复制进 data。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/query_support.py`、`code/idea_method/query_balanced_aux.py` 和本地 trainer。自然 BCE 与辅助流分开，无法配对的行仍参与 main；第一轮不加入 ranking、DG、language-score 动态筛选或新 negative 生成。 |
| 本方法特别要记录的数据 | `data/query_support_manifest.jsonl` 记录相同实际编码输入、各 q 的 present/absent 视频、审查来源、重复预算和 exact sampling probabilities。M1/M2 使用同 q 分布，不能把 q 频率变化混入 y 内配平作用。 |
| Baseline 与机制对照 | 保留 baseline 代码副本作为 M0 的原 recipe；只在本 idea 的辅助 data/weight/loader 改动。模型继续读 query 以识别事件，不能用去掉 text 来实现配平。辅助目标中 Cq 的最优 logit 为0，不意味着自然主流也无 query signal。 |
| 扩展到另外两个 backbone | 配置冻结后从 A 复制 Flash/QD 完整包，维持同样支持与权重干预，检验是否跨 backbone 作用一致；不能把三模型各自重新筛出的不同支持集称为同数据公平验证。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 研究问题与核心假设

研究对象是 `Y(V,Q)=1[视频 V 内至少存在一个满足完整 query Q 的事件]`。S−、U− 允许视频包含其他事件。目标是提高语义未见条件下的 `AUROC(U+ vs U−)`，并说明提高来自视频条件证据。

本方向提出一个可单独验证的数据监督假设：训练分布中 `P(Y=1 | Q)` 的差异，使 existence learner 可以通过 query 的措辞、语义常见性和编辑来源降低训练损失；这种关联在 semantic novelty 下不稳定。在同一个 query 既有可信 present 视频又有可信 absent 视频的子集上，建立 `P_aux(Y=1 | Q)=1/2` 的辅助训练分布，可能削弱语言边际奖励，使学习更依赖 video-query compatibility。

这是待检验的机制解释。当前证据尚未证明三个 backbone 主要利用 query prior，也没有证明 benchmark 整体存在 shortcut。配平训练更不等于在数学上删除整个模型的语言信息。

## 2. 三个 backbone 的共同问题

Moment-DETR-GMR、FlashVTG-GMR、QD-DETR-GMR 的 Seen/Unseen 均值分别为 0.7518/0.5287、0.7550/0.5479、0.7476/0.5144，且各自五个 split 全部 Seen > Unseen。三者存在性输入并不相同，却共享 source 正负构造与 scalar existence supervision。因此，优先测试监督分布中的语言边际，而不假设 Moment-DETR 的 max pooling 是共同根因。

Query-marginal matching 指**相同文本 query 的标签边际配平**。它不等同于同一个 action label 下正负数量相同，也不等同于整批正负各半：两个正负总量相等的 action groups 内仍可能每条 query 都只有一种标签。

## 3. 已有证据、反证与缺口

| 证据 | 支持提出本方向 | 必须保留的边界 |
|---|---|---|
| V5 Cq 四个 inner folds 的 pooled Novel AUROC 均提高，平均约 +3.16 pp | query 边际具有预测力，值得干预监督分布 | 不证明多模态 baseline 实际主要靠它 |
| Cq source-pair PairAcc 四 folds 均下降，等权约 −12.89 pp | pooled 收益不能自动解释为视频条件判别改善 | same-video PairAcc 本身也没有消掉 query prior |
| V3 sampler 失去约 14.71%–45.09% S+ 覆盖，Seen/localization 受损 | 干预应放在辅助 stream，保留自然主流全部覆盖 | 不能由 V3 失败推出所有配平或 conditional supervision 无效 |
| V4 简单 pooled correction 无稳定提升；V5 增容量/read slots 不稳 | 先改变有效监督，再增加结构 | 不证明所有存在性表示无信息 |
| V1、V2 的局部定位/eventness 改进不稳定 | query-conditioned 标签关联值得单独研究 | 仍可能存在视觉 feature 或完整事件证据不足 |

主要缺口是：同文本跨视频标签的可信支持有多少；该子集是否过于模板化；控制 query 后是否又留下明显的 video prior；配平能否改善原始全体样本而非仅支持子集。

## 4. 具体的数据定义

只使用当前 fold 的 `inner_train_seen` 标注。对 query 做保守的大小写/空白规范化用于候选关联；保留原始文本，并审查规范化是否改变专名或其他语义。第一轮不把 embedding 近邻、同 action tag 或未经核验的 paraphrases 当作同 query。

记 `D_q^+={V:(V,q,1) 已标注}`，`D_q^-={V:(V,q,0) 已标注}`，并按视频去重。支持集为 `Q_*={q:|D_q^+|>0 且 |D_q^-|>0}`。没有双标签支持的 query 不进入配平辅助 stream，但仍完整保留在自然主 stream。不存在标注不能由“来自另一视频”自动推断。

关联 manifest 应保留 qid、video_id、原始标签记录、source-pair linkage、文本规范化规则、构造类型、复核状态及 fold 身份。尤其检查同视频重复 annotation、一个视频多个 query 与跨 fold video 泄漏。

## 5. 模型、监督与损失

初始冻结 canonical grounding backbone 与定位输出，只训练与对照完全一致的小 existence head `s_θ(V,q)`。仍保留 query 输入：目标是消除语言边际的判别奖励，不是让模型无法理解事件身份。

自然主 stream 不改变 loader、label distribution 或 coverage：

\[
L_{nat}=\mathbb E_{(V,q,y)\sim D_{train}}\operatorname{BCEWithLogits}(s_\theta(V,q),y).
\]

辅助 stream 均匀选 q，再以 1/2 概率选标签，再均匀选该标签下的独立视频：

\[
L_{qm}=\frac{1}{|Q_*|}\sum_{q\in Q_*}\frac12\left[
\frac{\sum_{V\in D_q^+}\ell(s_\theta(V,q),1)}{|D_q^+|}
+\frac{\sum_{V\in D_q^-}\ell(s_\theta(V,q),0)}{|D_q^-|}\right],
\quad L=L_{nat}+\lambda L_{qm}.
\]

固定采样概率或等价权重由训练 manifest 确定，停止梯度，不由当前 Cq 分数、Novel-dev 标签、baseline 错误或当前训练 loss 动态产生。初始通过 query/y 分层采样实现配平；不能独立截断正负权重后继续声称精确配平。为避免极少数视频反复出现，记录 unique coverage 与重复次数，使用固定辅助预算，并把支持规模不足视为启动条件不满足。

若模型只输出 `g(q)`，对每个 q 的辅助期望 BCE 为

\[
\tfrac12\ell(g(q),1)+\tfrac12\ell(g(q),0),
\]

其最优 logit 为 0。由此可以说：纯 query-only 分数在该辅助目标上无法利用标签边际获利。**自然主流仍存在，且 video-only 及 scene-text interactions 未被配平，因此整个训练目标没有“完全无 shortcut”的保证。**

## 6. 为什么可能改善 Unseen AUROC

AUROC 衡量不同 query、不同视频的正负排序。辅助配平可能把 source-only language correlation 的梯度奖励转向视觉条件证据；如果这些 compatibility 规则能跨语义迁移，则 U+ 分数更容易超过 U−。这不是总体正负比例改变导致 AUROC 自动上升，也不是阈值校准。

收益存在条件：支持子集具有足够事件和视频多样性；视觉表示能够判别该事件；主目标与辅助目标权衡适当；conditional improvement 能传到跨 query pooled ranking。最后一项并无保证，所以必须保留自然分布 BCE 与 pooled 主评价。

## 7. 与 V1–V5 的继承和放弃

继承 V5 的 inner semantic holdout、Cq/Cv 控制和缓存身份核验，V4 的 frozen localization 对照，V3 的全量覆盖教训。保留 source-pair linkage 作为标签关联依据。

放弃 V3 的全 backbone balanced sampler、用 action-group 回退替代同 query、默认增大 R3 readout、generic eventness residual，以及仅通过 within-group AUROC 判定成功。本方向先只改变辅助分布；same-query pair ranking 是后续独立分支，不能混入第一轮后归因于配平。

## 8. 最小验证实验与成功判据

第一轮固定 head、训练 steps、辅助 forward 数量和同一个支持 manifest，比较：`自然 BCE`、`自然 BCE + 支持集原比例辅助 stream`、`自然 BCE + query-matched stream`。后两者只改变 query 内标签权重，避免把额外训练或更好支持样本当作配平效果。

建议在四个已定义 inner folds 上做开发，最初先运行一个 action 与一个 composition pilot；达到门槛再完成四 fold 和三 seed。训练只读 inner_train_seen；checkpoint 只按 inner_val_seen；Novel-dev 只作为预先有限的跨 run 开发评价。完整安排见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

推进标准为共同协议的预定门槛：原始 Novel pooled macro AUROC 建议至少 +2 pp、至少 3/4 folds 为正、Seen AUROC 下降不超过 1 pp；same-query、same-video 与 crossed 条件排序同时有支持；冻结定位保持输出不变。小支持子集的四格指标只能作机制证据，不能替代主指标；确认阶段还需多 seed 与三 backbone。

## 9. 失败后能排除什么

| 结果 | 可支持或弱化的解释 | 不能声称 |
|---|---|---|
| 支持子集变好，全体 Novel 无收益 | 配平有效范围有限、覆盖或迁移不足 | query prior 对整个 benchmark 无影响 |
| 原比例辅助与配平同样提高 | 额外数据曝光/优化是竞争解释 | 改善来自边际移除 |
| pooled 提高但条件排序不变 | score comparability/其他边际变化可能有用 | 真实视频条件证据改善 |
| Cq 在辅助分布仍明显高于 chance | 标签或采样未真正配平、泄漏、实现错误需要排查 | 理论性质失效 |
| 可信支持上多 seed 无收益 | 当前表示/支持/预算下，单独配平不足 | 所有语言偏差去除均无效 |

## 10. 最大 confounds 与控制

支持 query 可能更常见、更短或模板化；因此对照必须使用同支持集合，并分别报告支持/非支持与 edit-type strata。匹配 q 没匹配视频：场景、时长、视频来源仍可能预测标签，需要 Cv、背景/长度分层和已标注 crossed pairs。重复视频不能增加有效样本量，置信区间按共享视频/配对连通组件处理。

不能让 query-only scorer 同时定义采样权重和作为唯一去偏成功证据。语言 plausibility 仅作预定诊断或独立后续干预，不成为当前训练者自适应的 label selector。U− 必须核实整段视频没有该 query 事件，包括重复发生的同类事件。

## 11. 成本与执行优先级

成本低。已有表示缓存与关联标签足够时，可在 1–2 张 GPU 的小 head 实验启动；具体时长取决于缓存是否可信、支持规模及 frozen feature pipeline，本文不承诺未测量的 GPU 小时。额外标注成本主要是核验双标签 query 支持。

它在四格机制审计与优先方向 A/B 后实施。若 audit 已显示相同 query 跨视频判别良好，优先级下降；若 source query prior 有效而同 query 条件判别弱，优先级上升。

## 12. 论文故事、novelty 与 reviewer objection

可发展的贡献是：以严格 query-level 双标签支持构造一个保持自然主分布的辅助监督机制，实证区分“降低语言边际可预测性”和“提高跨语义兼容性”，并在三 backbone 上验证迁移。它把数据去偏思想推进到完整视频 event existence，而非仅重新筛掉 evaluation 中容易的语言负例。文献先例应帮助选择强对照，不能因为存在就放弃该方向；详见 [RELATED_WORK.md](RELATED_WORK.md)。

最强 objection 是“普通重采样/重新加权，论文方法贡献太弱”。响应需要超出一条加权公式：可信支持与覆盖协议、相同曝光对照、先验收益与视频条件收益的分解、以及跨语义/构造机制迁移证据。若最终只有局部 challenge improvement，应把它定位为诊断与监督研究，而不是宣称解决 semantic-novelty GMR。
