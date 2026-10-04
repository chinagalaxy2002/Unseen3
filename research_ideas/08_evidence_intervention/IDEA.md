# 事件证据干预：Evidence Sufficiency and Necessity

状态：`draft_not_executed`。

> 优先级 08；原总结 Idea 7。本文只提出可验证设计，不实施训练、推理或数据干预。该方向需要时间来源与干预标签 gate，不能作为无需准备的立即实验。共用要求见 [COMMON_PROTOCOL](../COMMON_PROTOCOL.md)，步骤见 [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)，原始文献比较见 [RELATED_WORK](RELATED_WORK.md)。

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

**本方向先做什么：** 首轮研究改变真正 query-event evidence 后的存在性响应：verified target drop 与同算子 matched background damage 对比，保留完整事件时保持 positive。需要先通过时间来源和 all-occurrence absence gate。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment baseline 与 raw dataset/feature 依赖、V5 时间审计记录和所需 metrics；可复制 05 的已选轻量 scorer工作版本到本 idea 并记录来源/hash，随后与 05 完全隔离。不能从另一个 idea 可变 code 目录直接 import。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/evidence_transform.py`、`code/idea_method/intervention_labels.py`、`code/idea_method/response_loss.py` 及本地 trainer。干预必须作用于未融合的原始视频特征，之后重算 video-query interaction；baseline定位主链冻结。 |
| 本方法特别要记录的数据 | 记录 feature extraction 时间来源或有误差界的近似映射、原/变换后 input hash、operator/intensity/context、所有 witnesses、重复事件、verified变换后label。必要性/保留/双差只对标签支持充分的样本定义，unknown不强制转负。 |
| Baseline 与机制对照 | 只删一处 GT 不能证明整视频 absent；背景也可能是 Q 的必要条件。相同破坏预算、边界数、能量/运动和第二算子泛化控制分别记录。新的裁剪/重拼不沿用旧时间窗；不得修改共享 feature npz 或把损伤识别当事件理解。 |
| 扩展到另外两个 backbone | 方法有效后从 A 复制三模型整包，保持一致可信干预支持和真实自然 U评价；不能对正式 U在线适配/生成新训练负例。时间gate未满足时保持待准备，把不用局部time的真实cross-video监督作为01方向，不冒充本干预已经实施。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 核心问题和假设

Existence 是 \(Y(V,Q)=\mathbf1[\exists e\in\mathcal E(V):e\models Q]\)，不是 generic eventness。三个 backbone 的 absolute source BCE 都可能接受 query prior、场景或周边状态作为证据，却没有区分“事件发生”与“场景使事件听起来合理”。H-I 假设：对真正的 query-event 证据进行破坏，应比等量背景破坏更影响 score；保留完整事件、改变无关背景，不应使存在性反转。训练中的正确干预约束有机会把这个差别迁移到新语义。

这个假设不是普遍的硬规则。删除某一 GT 区间后，事件可能在其他位置再次发生；背景也可能是 query 的必要条件，例如“在厨房开柜子”。干预破坏了必要参与者、前置状态或 temporal relation 时也不是无关背景。所有标签都必须是相对于完整 query 和变换后完整视频重新判断的。

## 2. 已有实验为什么值得提出、又为什么不能直接运行

| 证据 | 对本方向的启发 | 边界/反证 |
|---|---|---|
| 三 backbone Seen→Unseen 一致下降 | 可检验共用监督是否未建立事件敏感性 | 共享 feature 与 benchmark 也可能导致同样现象 |
| V1 增强 localization 不稳定 | “预测到位置”不等于依赖真实事件证据 | 不能否定所有 temporal supervision |
| V2 generic eventness 不稳定 | 干预须 query-specific | 不能把删任何活动都降低分数当成功 |
| V3 main sampler 改变分布并损伤 Seen | 干预作为辅助 stream，保留完整 main loader | 不把一切 ranking 失败归因于反事实思路 |
| V4 pooled 修正几乎无收益 | 需要改变证据学习而非只调一个 scalar | 不能证明表示无信息 |
| V5 Cq 提高 pooled 但 paired 退步 | 必须展示视频证据对 label 的正确响应 | text-only 强不等于 baseline 完全不用视频 |
| V5 R4 未通过 time gate | 时间干预必须先恢复来源或冻结可验证近似映射 | `clip_length=1`和长度吻合不是精确 timestamps |

继承冻结 canonical localization、inner splits、Cq/Cv、可信配对、full-precision score；放弃 generic eventness、主 sampler 重排、仅看局部删除后的拒绝率、teacher 自确认 absence、正式 U 测试时适配。与方向 03 的区别：03 检查联合事件满足性，08 改变输入证据并监督响应；两者先独立验证，不在 pilot 中一起增加模型和干预。

## 3. 时间来源和全视频 absence 是先决条件

现有 V5 记录没有找到本批 CLIP/SlowFast 逐 clip 时间戳或可独立核验的抽取日志，R4 仍 pending。启动前必须选择并冻结一种路径：

1. 恢复真实 extraction 时间轴，包括 sampling、stride、padding、末端取整、模态对齐、truncate 与可见时间支持。
2. 若不可恢复，另建有明确假设和误差界的 feature-index 近似映射，独立记录敏感性范围；用宽松 support 而非假精确秒数，并明确只研究近似特征干预。
3. 对固定小子集从 raw video 重新抽取有记录时间轴的特征；所有 arms 用同样新 feature，另列与 canonical 不同的 confound。

缺少任一路径时，本方向保持待准备，不能用未核验 `round(t*T/duration)`产生 GT 干预标签。近似 mapping 的证明强度有限：若移位一个 index 使结论翻转，就不能宣称定位到必要证据。

完整 absence 的核验独立于 mapping。列出 \(\mathcal W(V,Q)\) 中所有已核验 witness；只删一个标注不意味着集合为空。没有穷尽性保证时，变换后标签为 unknown 或保留 positive，不能训练 global negative。无需局部时间的同 query 真实 present/absent 视频对可以先支撑方向 01，不能冒充本方向的 evidence deletion 实验。

## 4. 模型与干预输入输出

使用固定的 video-query compatibility scorer \(s_\theta(X,Q)\)。canonical backbone、原始 localization 输出冻结；scorer 读取未融合的有序视频序列与 query tokens，修改视频后重新计算交互。不能遮掉已经包含原事件信息的 decoder states 后声称移除了证据。head 可采用 [方向 05](../05_independent_compatibility_adapter/IDEA.md) 的简单实现，但在本 pilot 所有 arms 固定相同结构，禁用其他方向的新增监督。

未来每个合格 \((V,Q)\) 构造以下变换并记录 label provenance：

| 变换 | 语义含义 | 允许的标签 |
|---|---|---|
| \(V^{\rm keep}\) | 保留一个完整 witness 与必要前后语境，改变已核验无关部分 | 若完整条件仍成立则 positive |
| \(V^{\rm drop}\) | 移除全部已核验 support，不保留别的 witness | 只有独立确认变换后 whole-video absent 才 negative |
| \(V^{\rm bg}\) | 同算子、同长度/clip 数、近似同视觉能量的无关部分破坏 | 独立核验目标仍成立才 positive |
| \(V^{\rm partial}\) | 只删一处，有其他 witness 或 label 不确定 | 不施加 global negative 或强制 necessity loss |

mask replacement、interpolation、clip excision 是不同实验。pilot 只固定一种算子，target 与 background 共享强度、边界处理和可见 mask 模式。decoder/TEF 等输入不得携带“这是 target 干预”的特殊标志。若重拼 video，重新建立其有效 index/时间轴，不能沿用旧 span 作为新视频 GT。替换 clip 来源只限 inner_train，且须核验不会新增 query 事件；feature cosine filter 本身不是 absence 认证。

推理只读取原视频和 query，输出 \(s_\theta(V,Q)\)。不调用 test-time GRPO、不读取 formal U labels、不根据当前测试 query 生成新训练负例。

## 5. 训练机制与可识别的证据响应

以全量 main BCE 为基础：

\[
\mathcal L=\mathcal L_{\rm BCE}^{\rm all}
+\lambda_{cf}\mathcal L_{\rm BCE}^{\rm verified\ interventions}
+\lambda_n\mathcal L_{\rm necessity}
+\lambda_s\mathcal L_{\rm preservation}.
\]

对“目标破坏后 whole-video absent、匹配背景破坏后 present”的已核验子集，定义

\[
\delta_{t}(Q)=s(V,Q)-s(V^{\rm drop},Q),\quad
\delta_b(Q)=s(V,Q)-s(V^{\rm bg},Q),
\]

\[
\mathcal L_{\rm necessity}=\operatorname{softplus}(m-\delta_t(Q)+\delta_b(Q))
=\operatorname{softplus}(m-s(V^{\rm bg},Q)+s(V^{\rm drop},Q)).
\]

它要求 matched 背景破坏比 target 破坏更兼容；不是“任何破坏都降分”。原 video 正与 verified-drop 负还提供 same-query ranking/BCE。

保留正例的 loss 可为

\[
\mathcal L_{\rm preservation}
=\operatorname{Huber}(s(V^{\rm keep},Q)-s(V,Q))
+\operatorname{Huber}(s(V^{\rm bg},Q)-s(V,Q)),
\]

只用于无关部分与事件完整性核验通过的样本。绝对 score consistency 可能过强，未来单独比较仅 positive BCE；不得强迫每个裁剪视频与原 video 的 score 一模一样。低置信 label 不采用更高 loss 权重“修正”。

进一步的 query specificity 诊断使用同一 video、同一 target/background 变换及一个已核验不受这些变换影响的 \(Q_c\)：

\[
\eta=[\delta_t(Q)-\delta_b(Q)]-[\delta_t(Q_c)-\delta_b(Q_c)].
\]

对 \(s(V,Q)=a(Q)+b(V)\)，该双差严格为 0。只有存在 \(Q_c\) 真值支持时才定义；不能拿随机无关 query 的原标签作证。\(\eta>0\) 排除了该加性形式，却不能排除其他场景—query 交互 shortcut。第一轮不默认优化 \(\eta\)，先测它，避免同时改变监督太多。

## 6. 为什么可能改善 AUROC、什么时候不会

正确干预把相同 query 的 present/absent video 都暴露给模型，语言边际不能解释标签改变；matched damage 避免模型只识别 video 损伤；preservation 支持背景变化的不变性。若这些约束能学到 query 相关的证据响应，semantic novelty 下高相似但缺事件的 U−有望降分，同时 U+在不熟悉背景中保持高分。

该推理依赖干预标签真实性、feature 信息和迁移。只在合成 masked video 上成功不能证明 natural U improved；只调 threshold 不改变 AUROC；只因 U+变得更保守、Seen 下降而 gap 缩小不是成功。若 query 需要长时序或全场景 context，局部 support 错误会制造 false negatives。

## 7. Success、failure、confounds 与成本

建议同其他方向采用四 innerfold Novel macro +2 pp、3/4 为正、Seen ≥−1 pp 的推进门槛；同时可信同 query、samevideo 与四格指标改善。target-vs-background 效应和\(\eta\)必须改善，并报告 Cq/Cv、shuffled-video、hard positives 和 U+false rejection。门槛是未来建议，非效果预测；统计按原 source video 聚类，多个干预不能增加独立样本数。

主要 confounds 依次是 absence 标签错误、剪接/zero mask 伪影、重复事件、时间映射误差、算子/clip 长度捷径和原 feature 不足。target 与 background 只等 clip 数仍可能不等损伤，应匹配支持长度、motion/feature 能量、边界数与输入可见性；另用第二算子做泛化检查。若只有合成拒绝改善，结论限于该算子；若\(\eta\)不变但 pooled 变好，不能称 query-specific evidence；若背景破坏也被拒，说明伤害检测而非事件证据。

成本中至高：通过 time gate 后的 cached-feature 小 head 可 1–2GPU 试验，但恢复来源、复核重复事件与必要 context 可能是主要成本。在 gate 前应优先低成本的方向 01/02/07。

## 8. 论文故事与贡献空间

CCL、CVA 和 HRVTG 已明确研究 robust / destructive 变换、背景不变性或证据删除。拟新增的是“带 absence 有效性约束和 matched damage 的 semantic-novelty existence 干预监督”：区分事件敏感性与任意输入损伤，区分局部证据减少与完整视频 absence，并把它连接到 natural U AUROC 和 prior-free 条件控制。模型本身可以简单，贡献来自可识别干预机制与可信监督；若仅复现已知 mask ranking 而缺上述证据，不应夸大 novelty。

## 9. 项目依据

[用户研究总结](../../semantic_novelty_GMR_next_stage_research.md)、[V5 时间审计](../../experiments/moment_detr_gmr_evidence_v5/RUN_STATUS.md)、[V5 R4 freeze 边界](../../experiments/moment_detr_gmr_evidence_v5/P3_RUN_STATUS.md)、[V5 readout 结果](../../experiments/moment_detr_gmr_evidence_v5/P3_READOUT_RESULTS.md)。证据快照 SHA：`6cd96d723806e4b2b2474c68f289921eb543ffcc`。
