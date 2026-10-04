# 02 · Reference-Centered Interaction：参考分布下的纯交互存在性打分

状态：`draft_not_executed`。原总结 Idea 2；推荐方向 B，优先级 02。成本：低至中，取决于 scorer 是否支持廉价分解。研究目标和 V1–V5 证据见 [COMMON_PROTOCOL.md](../COMMON_PROTOCOL.md)。本方向尚未实现，也没有新增性能结果。

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

**本方向先做什么：** 首轮使用冻结原始 feature 的双线性 scorer，比较 raw-trained/raw-tested、同 checkpoint post-hoc centering 和 centered-trained。它先验证候选评分方法；成功不直接归因原三个 canonical baseline 的退化。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment 的 dataset/feature 相关完整依赖及需要的 metrics；若保留 frozen localization，再复制其完整 models/training/configs。V5 `video_mean/query_tokens/query_mask` 等 bank 可以按 identity 只读使用，不能拿其 query-conditioned slots 任意换 Q。首轮未使用某 backbone forward 时无需为凑三个模型一并训练。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/interaction_scorer.py`、`code/idea_method/reference_bank.py` 和本地 train/evaluate。`f−A−B+C+b0` 的项全部使用当前同一 f 与固定权重；linear projection 可在冻结输入上预中心化。Output scorer 不默认混合 raw baseline prior。 |
| 本方法特别要记录的数据 | 在 `artifacts/reference_banks/` 和 `records/` 保存 Seen-only bank 成员、weights、输入 hash、均值/投影版本、bank seed/size、有效参数和计算量。B0/B1/B2 的特征、训练行、选模与 rank 匹配；B1 保持同 raw checkpoint。 |
| Baseline 与机制对照 | 核对纯可加 score 相消、共享 intercept、post-hoc 处理不改变同一四格 D。Reference 不是已认证 absent 的 null videos。所有 bank 缓存都在本目录；不根据正式 U 选 bank/temperature。 |
| 扩展到另外两个 backbone | 后续对原三模型归因时，从 A 分别复制完整 Moment/Flash/QD 三组包，保持固定 canonical weights，真正 forward `(Vref,Q)` 与 `(V,Qref)`，对比 correction。跨 backbone 和表示变化分别记账；基于新 scorer 的成功不能冒充原 baseline offset 已被证明。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 核心假设

存在性分数可以概念性分为 `f(V,Q)=a(Q)+b(V)+c(V,Q)`。Source BCE 可能同时学习语言频率、视频总体可匹配性和真实交互；semantic shift 后，跨 query 的偏置变化会破坏 pooled ranking。假设是：显式去掉可加单模态分量，并直接在剩余交互分数上训练，有机会改善新语义存在性排序。

此分解不是因果识别。交互也可以编码 scene×word 的伪相关；参考分布改变时分解会改变。方法需要双条件监督/评价与换视频控制支撑，不能把公式里写了 `interaction` 当作实际使用事件证据的证明。

## 2. 三个 baseline 的共同问题

三个 backbone 都输出可跨 query 比较的 scalar score；它们没有统一约束该 score 中语言偏置、视频偏置与关系项的比例。具体 pooling 不同，不影响这个问题。本方向可作为共享存在性评分层，或者独立 factorized head，检验的是 score formulation 和训练的一致性。

若只存在 threshold miscalibration，全局单调校准不会改变 AUROC；本方向改变样本间排序。它不是简单 temperature scaling，也不是把 Seen threshold 调到 U。

## 3. V1–V5 依据与反证

- V5 Cq 的稳定 pooled Novel 收益使 query-dependent nuisance 成为应验证的解释；但也可能是合法标签分布相关性，不能直接叫 shortcut。
- V2 generic eventness 不稳定，提示 video marginal 不能替代 query 条件证据。
- V4 已否定一套 pooled-state residual 的实用效果，因此“训练与评估都只用交互分数”必须与“训练 raw score 后 post-hoc 修正”区分。
- V5 R2/R3 无稳定收益，对“简单加 query 或增容量就能分离偏置”不支持。
- 反对：移除合法频率与总体可匹配性可能降低 natural pooled AUROC；reference bank 不能自动代表 Novel semantics。当前没有证据证明相减后的交互比 raw score 更有信息。

缺失证据：单模态可加项的贡献；只作 post-hoc 是否有效；训练 centered scorer 是否产生更好的交互；中心化效果对 reference 变化的稳定性。

## 4. 与 V1–V5 的区别

保留冻结 baseline、Seen-only inner development、容量对照和 Cq/Cv。放弃 baseline score 加 residual 的默认形式、decoder slots 必选以及语言先验混入最终 score 的捷径。模型可以从最简单的共享双线性 scorer 起步；首轮不联训定位，也不把 centered score 与 raw baseline 任意线性混合，因为那会重新引入需被检验的先验。

## 5. 分数定义与保证边界

### 5.1 两个固定 Seen-only reference banks

`RV={Va}`、`RQ={Qb}` 都仅来自 inner train，使用固定、非负、归一化的权重 `pa`、`wb`；最终部署只需要当前 `(V,Q)` 及这些 reference，不能用 evaluation query pool 做 transductive normalization。Reference 视频不是已知 absence 的 null videos，随机视频可能满足 query，因此只称 reference，不宣称负例认证。

所有项使用完全相同的 scorer 参数及真实输入组合：

\[
\begin{aligned}
A(Q)&=\sum_a p_a f(V_a,Q),\\
B(V)&=\sum_b w_b f(V,Q_b),\\
C&=\sum_a\sum_b p_aw_b f(V_a,Q_b),\\
e(V,Q)&=f(V,Q)-A(Q)-B(V)+C,\\
s(V,Q)&=e(V,Q)+b_0.
\end{aligned}
\]

`b0` 是共享可训练 intercept，用于自然 class prevalence；不能使用 query-dependent intercept。对固定 finite banks，任意可加 `f=a(Q)+b(V)+c0` 的 e 严格为 0，最终 s 为常数，故单模态 AUROC=0.5。随机独立采不同 bank、使用 stale/non-shared scorer 或不一致权重会破坏此性质，必须记录并检查。

有限 bank 的 reference mean 下，e 对每个 reference Q 的 video 平均、每个 reference V 的 query 平均均为 0。该保证只涉及同一参考分布，既不保证 Novel 的校准，也不保证任何交互都是真实事件。

### 5.2 便宜的双线性 pilot

原始预计算视频和文本分别 masked mean 后得到 `x(V), z(Q)`，先学习投影，再使用：

\[
f(V,Q)=u(V)^T Wt(Q)+a(Q)+b(V),\qquad
e=(u(V)-\mu_u)^TW(t(Q)-\mu_t).
\]

不能直接把 raw video / text 向量点乘，维度和空间均需确认。若 u、t 为线性投影，可先中心化冻结输入均值，训练时无需反复 forward 整个 bank；若为非线性投影，必须按当前参数重算 bank mean 或明确近似误差。均值若只算在旧 encoder 参数上，不能声称严格等于四项打分。

Raw comparator 保留相同交互 rank 和投影。由于 a、b 在 centered arm 中被抵消，其参数可能无有效梯度，必须同时报告有效参数量；“总参数相同”不自动说明容量相同。可加同有效预算的 centered-only head，不能用无效分支虚凑容量。

### 5.3 更一般的非线性交互

若 pilot 表明双线性容量不足，单独扩展共享 `f([u;t;u⊙t;|u−t|])`，四项都执行当前 f。不得用独立的小 prior head 估计 A、B，却声称严格去除了所有可加项；这种版本是另一近似机制，需要独立验证。若用 GMR conditioned features，所有 `(Va,Q)` 和 `(V,Qb)` 都要在 fusion 前替换并重算，不能复用原 Q-conditioned slots。

## 6. 训练机制

首轮在 s 上做全量 natural BCE：`L=BCEWithLogits(s,Y)`；global b0 允许不平衡 class prevalence。额外双条件 ranking 来自 01 的已核验支持，但作为单独 factor，不把同时换监督的收益归给中心化。Negative sampler、主流覆盖和选模遵循共同协议。

需要比较：训练 raw / 测 raw；同一 raw checkpoint post-hoc centered；训练 centered / 测 centered。只有这样才能判断新增训练机制是否超越 V4 式后处理。Reference banks、尺寸和 weights 从 Seen train 固定，不能根据 U 的最好结果挑 bank。

一个重要恒等式是：**固定 raw scorer 的后处理中心化不会改变任何闭合四格的 D**。因为减去的是 row/column 可加项。它可以改变单个 margin 和 pooled ranking，但不能凭后处理创造新的交互差分。若 centered-trained scorer 的 D 和 strict group 同时改善，才有证据支持交互学习变化。

### 6.1 Pilot 与根因归因分开

首轮 raw-feature 双线性 scorer 检验的是候选评分方法：即使 centered-trained 有效，也不能直接证明三个 canonical GMR 的原始 score 都因加性偏置退化。要对原 baseline 做机制归因，后续需保持 canonical scorer，实际重算 reference 输入组合，并区分其 post-hoc 校正与重新训练的效果。Feature 路径、容量和训练目标改变造成的增益须单独报告。

## 7. 为什么可能改善 U+/U- AUROC

如果退化主要来自 query-dependent offset 或 video hubness，reference subtraction 可改变跨样本排序，降低单模态偏置支配。更关键的是 centered training 迫使 BCE 梯度落到 interaction 结构，而非在训练末尾删除模型依赖的主要信号。

但 generic interaction 仍可能过拟合 source 语义，纯中心化也可能删掉合法统计。方法是否缓解主 AUROC 只能靠 natural pooled 与双条件 evaluation；per-query 严格单调 CDF 不能改变同 query 内排序，不能作为本方向成功证据。

## 8. 最小实验与成功判据

[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 给出冻结原始特征上的三个小 scorer arms、单模态对照和 bank sensitivity，一至两张 GPU 资源即可进行 pilot。成功需要 centered-trained 超过 raw 与 raw-posthoc、Seen 保持、Novel pooled 和 same-query/四格指标同向；不允许 Cq/Cv 或错配视频复制相同机制收益。一个 fold 有效不够支撑跨语义稳定性。

## 9. 失败可排除什么

- Posthoc 好、centered training无新增：当前贡献更接近现有检索 normalization，训练扩展未获支持。
- 条件指标好、pooled下降：单模态分量包含主数据的合法判别信息，或 centered scale 仍不稳定；不能为了更干净的叙事忽略主指标失败。
- 所有 centered 模型 train 都难拟合：可能容量、bank 或支持限制；不能排除更好的 relational evidence。
- Frozen raw features本身无条件信号：不支持“仅分数分解即可解决”的简化解释。
- 不同 banks反转收益：削弱统一稳定 interaction 的论证，应报告 source-reference dependence。

## 10. 最大 confound 与实施成本

最大 confound 是 reference 分布造成的重排序、同类视频密度与额外计算，而非新事件证据。双线性数学等价于 centered features，不能夸大机制；更一般 f 的 bank 计算增加预算，须 exposure/compute 对照。Mean pooling 的 feature 本身可能弱，所以失败仅限于该表示；不能声称 whole representation 无信息。

双线性缓存版成本低；非线性 bank、重复 fusion 或适配三个 backbone 成本中；完整长序列四项重算可能更高，优先等待 pilot。所有成本是未来测量前的定性估计。

## 11. 最近邻与新增贡献

QB-Norm、DBNorm、CSLS、RCSLS 已有 reference / neighborhood correction，RCSLS 甚至将检索校正纳入训练。因此“首次去双侧偏置”“首次训练与归一化一致”都不可主张。详见 [RELATED_WORK.md](RELATED_WORK.md)。候选定位 **B/C**：明确可加项消失的 fixed-bank interaction score，面向 task-semantic-held-out whole-video existence，借助 Cq/Cv/四格/错配 controls 区分 ranking correction 与新增视觉条件学习。

## 12. 最强 reviewer objection

“这只是 centered bilinear / CSLS 的存在性应用，可能通过删掉 class prior 美化条件指标。”回应需要 exact scoring 对照、posthoc-vs-trained factorization、reference sensitivity、同有效预算、原始 pooled AUROC 与 U+ 保留，以及与 QB/DB/CSLS/RCSLS 思想适配的对照。若没有额外训练机制的稳定贡献，研究仍可报告有价值的失败解释，但不写成全新模型方法论文。
