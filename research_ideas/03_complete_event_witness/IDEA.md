# 完整事件 Witness Verification

状态：`draft_not_executed`。

> 优先级 03；原总结 Idea 5、最终方向 C。本文是研究设计，不是已实现方法或实验结果。当前阶段只撰写 Markdown，不运行实验。共同数据和评价约束见 [COMMON_PROTOCOL](../COMMON_PROTOCOL.md)，具体对照见 [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)，文献边界见 [RELATED_WORK](RELATED_WORK.md)。

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

**本方向先做什么：** 首轮冻结原始视频/文本特征及 baseline 定位，比较普通整句 head、跨候选独立 atom 聚合、显式 action–argument 联合 witness。目标是 query 的完整事件满足性，不能把同窗口共现直接当同实例绑定。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 `training/moment_detr_gmr/dataset.py` 所在完整依赖包和相应模型/配置；V5 metrics/inner manifests 可复制为模板，V5 pooled/slots bank 不是完整 raw sequence，需读取原始预计算视频序列。V1 phrase modules 若未来尝试，只复制到本 idea 并作为单独消融，不默认继承。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/witness_verifier.py`、`code/idea_method/index_candidates.py`、`code/idea_method/query_structure.py` 及本地 auxiliary loader。普通 W0 不显式路由 atom/argument，W1 允许独立 maxima，W2 验证同候选有序联合条件；数据与容量预算匹配。 |
| 本方法特别要记录的数据 | 在 `data/` 保存 verified binding negatives、hard-positive paraphrases、uncertain 标记、parser 版本与错误；记录 candidate index support 和 masks。`artifacts/` 保存新 sequence cache，不能写旧 V5 bank；所有窗口不依赖正负身份或 GT oracle。 |
| Baseline 与机制对照 | 首轮不用未核验秒级 ROI，也不把完整 query negative 的所有 phrases 都标负。只有 bag positive，不把全部候选标正。若 W2 只优于 W1 而不优于 W0，结论只能是独立聚合有害，不是实例绑定机制已成功。 |
| 扩展到另外两个 backbone | 最终验证可从 A 三个 baseline 复制定位模型并用各自 canonical 输出作原参照，保持统一 witness 分支输入/预算。旁路在三 backbone 上使用同一存在性 head 的收益，不自动证明三者内部都缺同一种 representation。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 研究问题与可以被证伪的假设

目标是缓解三个 GMR backbone 在 semantic novelty 下的 existence AUROC 退化，而非延续版本。标签为

\[
Y(V,Q)=\mathbf 1[\exists e\in\mathcal E(V):e\models Q].
\]

U− 可以包含大量事件，只是没有满足当前 query 的事件。假设 H-W 是：模型容易接受分别支持 query 的若干词的证据，却没有检验这些条件是否成立于同一个事件实例。例子：视频中有人拿书、又有人放杯子，并不支持“有人拿杯子”。同一宽区间中同时出现动作和对象，也不等于动作作用于该对象。多个参与者、先后事件和角色交换会让这种差别更明显。

H-W 包含两个层次，应分别验证：W1 是跨时间的独立 primitive maxima 导致错误组合；W2 是同一候选内仍缺乏 action–argument / participant binding。简单 shared-window MIL 最多检验 W1，不能自动宣称解决 W2。初始方法是能读取完整查询与有序候选序列的联合 verifier；真正的角色实例绑定可能需要对象轨迹或更细视觉特征，应作为后续升级而非 pilot 的既成能力。

## 2. 三个 baseline 的共同问题

三个模型并不共享 max-pooled decoder architecture。它们更可能共享的是 source 正负标签、absolute binary discrimination 和一个可跨 query 排序的 scalar。训练可以通过语义相似、词汇可行性或局部事件共现得到较高分，而没有要求分数由一个完整事件 witness 支撑。本文检验这种监督/表示问题是否跨 backbone 存在；不能从三个共同退化直接推出 H-W 已成立。

主指标仍为 AUROC(U+ vs U−)。应提高缺失某个必要条件的 U− 的拒绝能力，同时保持新语义 U+ 的接受能力，而不是把未见词当作拒绝信号。

## 3. V1–V5 对本方向的约束

| 证据 | 提出本方向的理由 | 必须保留的限制 |
|---|---|---|
| V1 phrase/localization 增强不稳定 | phrase correspondence 与完整事件成立是不同目标 | 不得宣称加更多 phrase loss 会自然成功 |
| V2 generic eventness/residual 不稳定 | 应验证当前 query 的 action–argument 关系 | V2 不能否定 query-conditioned local evidence |
| V3 ranking+sampler 损害 Seen/localization | 使用全量 main stream，辅助配对不改变定位训练分布 | V3 是混合干预，不能当纯 conditional objective 反证 |
| V4 pooled residual 约 +0.0004、CI 含 0 | 从证据与监督重新定义 scorer，非 scalar 修补 | 冻结表示是否足够仍未知 |
| V5 R3 无稳定 pooling 证据、Cq pooled 提升 | 不能仅增大 slots readout；必须做视频条件验证 | Cq 不证明 backbone 主要依赖文本；R4 尚未运行 |

继承：冻结 canonical localization、inner semantic holdout、同视频 source pairs、容量匹配、Cq/Cv 和换视频控制。放弃：V1 的独立 phrase 聚合、把整句负例全部 phrases 标负、V2 generic eventness、V3 主 sampler 重排、V4 residual bounds、R3 必选架构。

## 4. 可实现的最小模型

### 4.1 输入与候选

输入是原始预计算视频特征序列 \(X\in\mathbb R^{T\times d_v}\)、query tokens \(Z\in\mathbb R^{L\times d_q}\)、有效 masks。canonical backbone 与定位输出冻结。训练标签、source qid、edit type、semantic group、是否属于 Novel 等不进入模型输入。

候选集合 \(\mathcal C(X)=\{X_{I_k}\}_{k=1}^K\) 从 feature indices 上的确定性多尺度连续窗口得到，保留窗口内顺序和 valid mask；正负 query 使用同一候选集合。不得以 source-positive query 的 hidden states 供负 query 使用。pilot 不用 GT 秒级 ROI，也不把 checkpoint 的 `clip_length=1` 当精确 extraction provenance。所有模型使用相同 index windows、候选数量和采样预算；候选是否覆盖真实完整事件需单独诊断，不能借定位 GT 只帮正例。

### 4.2 查询条件与联合验证

对可以可靠解析的 query，将必要条件写为 \(G(Q)=(A,O,S,R)\)：动作、对象、主体以及 relation/order。解析器在训练前固定；复杂或低置信度 query 保留 full-query verifier，不丢出 main stream。解析器只读文本；edit metadata 只路由已核验的辅助监督。pilot 优先 action–object，不声称具备角色轨迹。

同一候选经小型 temporal encoder 和 query-conditioned cross-attention 后得到有序 token sequence \(H_k\)。可以设置动作条件表示 \(u_k^A\)、对象条件表示 \(u_k^O\)、full-query 表示 \(u_k^Q\)，以及由动作与论元交互注意力形成的 joint representation \(u_k^{AO}\)。联合 logit 定义为

\[
j_k=f_\theta(u_k^Q,u_k^A,u_k^O,u_k^{AO},\operatorname{TemporalPool}(H_k)).
\]

其中 \(u_k^{AO}\) 从 action-conditioned visual tokens 与 object-conditioned tokens 的有序交互产生，而不是仅拼两个独立全视频 maxima。独立 atom score \(a_{kr}\) 只作辅助诊断。若采用完整条件约束，候选 logit 可为

\[
z_k=-\tau_b\log\left[\frac{e^{-j_k/\tau_b}+\sum_{r\in R_Q}e^{-a_{kr}/\tau_b}}{1+|R_Q|}\right],
\]

再以固定温度的长度归一化 MIL 聚合：

\[
s(V,Q)=\tau_m\log\left(\frac1{K_{\rm valid}}\sum_{k\in\mathcal C_{\rm valid}}e^{z_k/\tau_m}\right).
\]

最终 score 是 logit，AUROC 用全精度值。温度、窗口规格和是否加 soft conjunction 均为未来开发候选，不是已冻结参数；pilot 可先只用 \(z_k=j_k\) 防止 atom 模块过度增加混杂。soft-min 与共同候选不能保证真实实例 binding，它们必须在错绑定 challenge 上被证伪。若没有联合验证收益，只能报告 shared-evidence aggregation。

## 5. 监督与训练

\[
\mathcal L=\mathcal L_{\rm BCE}^{\rm all}+\lambda_p\mathcal L_{\rm pair}+\lambda_h\mathcal L_{\rm hard+}+\lambda_b\mathcal L_{\rm bind}.
\]

- \(\mathcal L_{\rm BCE}^{\rm all}\)：完整 inner_train 样本的 existence BCE。正视频是 MIL bag positive，不把所有候选标正；verified 全视频负例提供完整 query 的 bag negative，不凭“没有 GT span”判负。
- \(\mathcal L_{\rm pair}=\operatorname{softplus}(m-s(V,Q^+)+s(V,Q^-))\)：只用真实核验的 source pair；若负 query 可在视频其他位置出现，则不能当全视频负例。
- \(\mathcal L_{\rm hard+}\)：相同视频上保持语义的 paraphrase consistency，加正例 BCE；修饰语、数量、顺序改变可能改语义，需复核，不靠 LLM 自述保证。
- \(\mathcal L_{\rm bind}\)：对核验“各 primitives 出现但关系不成立”的完整 query 施加联合 evidence 排序。标签不足时不运行该项。未修改的 phrases 不标负；仅在有独立 atom 标签时监督被修改的 atom，否则只监督 full-query compatibility。两种监督边界分开保存。

配对和绑定子集是额外 auxiliary stream；没有配对的样本仍进入主 BCE，不从 main loader 删除。定位链无梯度。训练只读取 inner_train；checkpoint 由 inner_seen_val 选择；Novel-dev 仅用于有限配置开发，不每 epoch 选 checkpoint；formal U 不参与选择。

## 6. 为什么可能提高 Unseen AUROC

若现有 U− 的高分来自独立语义成分的共现，联合 verifier 会降低这类负例分数；若 action–argument 关系规则能从 Seen 迁移到新组合，U+ 可获得合理 witness。AUROC 提升来自 U+、U− 排序分离，不依赖统一 threshold 调整。该论证对新组合强于完全未知动作：若预计算特征无法识别新动作，显式绑定不能补出不存在的信息。严格 conjunction 也可能损害新语义正例，应同时报告 U+ score 分布、召回与 hard-positive 保留。

## 7. 成功、失败与混杂

建议推进门槛：四 inner folds 的 Novel macro AUROC ≥ +2 pp，至少 3/4 为正，Seen 下降不超过预定 1 pp 容差；同时 same-video、same-query、crossed 指标改善。具体数值是计划建议，启动前冻结。组合错配 challenge 必须额外改善；query-only 或 shuffled-video 无法复现同等方向的收益。定位 raw 输出保持一致。

失败解释应分层：ordinary joint head 与 witness 都失败，支持“现有特征/标签不足”的竞争解释；joint 优于 pooled 但绑定不优于 joint，不能主张 binding 贡献；仅 challenge 好、natural pooled 不好，说明覆盖或跨 query 分数可比性仍是瓶颈；U− 变好但 U+ 大幅变差，属于过度拒绝而不是有效泛化。任何单个失败都不能排除所有事件证据方法。

最大 confound 是更好的负例、更多训练曝光、输入更长和参数更多，而非绑定机制。所有机制对照需同输入、候选、正负数据、loss budget，并做容量匹配；解析错误、candidate truncation、participant identity 不可见和重复事件按子组报告。

## 8. 成本与论文故事

最小版本成本中：缓存视频序列上训练小 head，计划用 1–2 GPU 判断可行性，不承诺具体时长。人工复核组合负例可能比 GPU 更贵。升级 object tracks、多角色、顺序绑定成本高，应等待 pilot。

可以发展的故事是：semantic-novelty existence 的关键可能是“完整事件满足性”，而非更好的定位或更宽 readout；用受控错绑定与边际先验控制证明这一 failure，再建立联合 witness scorer。最强 objection 是“SHINE/RA-RFT 加 verifier”；回应依赖同数据的 joint-vs-independent-vs-binding 消融与 U+ 保留，而不是换个方法名。文献思想可以继承；新增贡献须是被实验验证的事件实例绑定与 query-conditioned absence 泛化。

## 9. 本文依据

[用户研究总结](../../semantic_novelty_GMR_next_stage_research.md)、[V1](../../experiments/trm_gmr_joint_v1/MULTI_SPLIT_RESULT.md)、[V2](../../experiments/trm_gmr_joint_v2/CURRENT_AUROC_RESULTS.md)、[V3](../../experiments/trm_gmr_joint_v3/MULTI_SPLIT_RESULT.md)、[V4](../../experiments/moment_detr_gmr_auc_v4/MULTI_SPLIT_RESULT.md)、[V5 readout](../../experiments/moment_detr_gmr_evidence_v5/P3_READOUT_RESULTS.md)、[V5 temporal gate](../../experiments/moment_detr_gmr_evidence_v5/P3_RUN_STATUS.md)。基线证据快照 SHA：`6cd96d723806e4b2b2474c68f289921eb543ffcc`。
