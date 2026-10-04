# Idea 07：面向兼容性差值的语义与构造机制泛化

> 状态：`draft_not_executed`。仅研究设计，未实施环境划分、训练或推理。  
> 对应原总结 Idea 4；当前优先级 07，是在条件证据得到支持后的升级方向。  
> 证据基线：`6cd96d723806e4b2b2474c68f289921eb543ffcc`，2026-10-04。  
> 来源：[下一阶段总结](../../semantic_novelty_GMR_next_stage_research.md)；关联：[实验计划](EXPERIMENT_PLAN.md)、[文献比较](RELATED_WORK.md)、[共同协议](../COMMON_PROTOCOL.md)。

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

**本方向先做什么：** 在可靠条件 pairs/quartets 上比较相同数据和预算的 ERM+pair 与 environment-aware compatibility risk，泛化的是关系判断而不是删除事件语义。需要高优先级条件监督已有可学信号。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment baseline 和必要 inner feature/metrics helpers；条件支持使用当前 idea 的本地固定 manifest，可从已核验上游 idea 产物复制并记录父 hash，但不在线 import/修改另一个 idea 的 sampler 或模型。新 paired loss 也复制为本地依赖。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/environment_manifest.py`、`code/idea_method/compatibility_risk.py`、`code/idea_method/episodic_optimizer.py` 和本地 trainer。只对 existence learner 进行虚拟更新/环境风险；canonical localization 冻结，full natural BCE保留。 |
| 本方法特别要记录的数据 | 记录 pair-unit 环境归属、每环境独立视频/标签/语义覆盖、元更新一阶/二阶实现、额外 forward/backward 次数。ERM comparator 同 pairs/compute。MLDG/IRM/GroupDRO 是已有机制对照，不将普通风险相加或 detach 后的近似误写为新元学习。 |
| Baseline 与机制对照 | 不能让自然正例独占环境、负例按编辑类型独占另一环境；环境附着于完整 pair/unit。Semantics 决定事件身份，不能无条件对齐掉。Teacher 已见 training episode 的语义，所以 episode holdout 不等于 teacher-unseen，真正 Novel 以严格 inner边界评价。 |
| 扩展到另外两个 backbone | 从 A 复制各 backbone 整包，保持相同环境/监督/预算并单独冻结各模型输出。严格 inner Flash/QD checkpoint 需建立；不能借原 formal Seen 训练权重宣布其 inner-held-out 泛化。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 研究问题与核心假设

本方向研究：同一个 event-existence 判断规则能否跨 Seen 语义家族及负例构造机制迁移，而不是只在 source environment discrimination 上成功。假设训练中的背景共现、语言编辑习惯和 semantic-family 难度共同影响 scalar existence；仅最小化合并后的 BCE，可能奖励 environment-specific correlations。

希望泛化的是**判断 video 是否满足 query 的关系规则**。动作、对象、角色和顺序决定事件身份，必须保留。绝不要求 query semantic embedding 在不同动作间相同，也不对全部视频/语言表示做无条件分布对齐。标签为 `Y(V,Q)=1[存在满足 Q 的事件]`，U− 不是无事件视频。

核心可证伪预测是：在可信配对上，对 compatibility score differences 做跨环境训练，能超过使用相同 pairs、相同计算的 ERM+pair，并提升 unseen semantic pooled AUROC 及两向条件排序。DG 是否有独立作用必须与 pairing 的作用分开。

## 2. 三 baseline 的共同问题

三个 GMR backbone 的存在性输入和定位结构不同，但共享 Seen 训练分布、负例构造与单 scalar supervision；全部五 split Seen > Unseen。它们可能都学到 source 有效而 semantic shift 下失效的兼容性近似。因此本方向只修改 existence learner 的跨环境目标，初始冻结 localization，不依赖 Moment-specific pooling 假设。

跨环境的难点有两类：新语义内容与 source 不同；构造/背景等相关性改变。第二类可以在 Seen 内模拟，第一类只能通过真正 holdout 评价，不能靠任意把 action groups 叫 domain 就宣称解决了新语义。

## 3. V1–V5 提供的支持、反证与缺口

| 项目证据 | 对本方向的启发 | 限制 |
|---|---|---|
| 三 backbone 的退化一致 | 共享 supervision/domain correlations 比单 backbone 结构更值得检验 | 单 seed 共享 benchmark 不能识别具体 causal mechanism |
| V5 Cq pooled Novel 稳定提高而 source PairAcc 降低 | 泛化目标须约束视频条件证据，不能只学 source text-label rules | Cq 本身不证明三个 backbone 都走 language shortcut |
| V3 semantic-group ranking + sampler 损害覆盖、Seen/localization | 不重写主分布；DG 的附加价值需在相同 pair supervision 上比较 | 不能用 V3 混杂失败否定 DG，也不能当成 DG 正证据 |
| V4 residual correction 无稳定收益 | 学习机制可能需要改变，不只后处理分数 | 更复杂目标也可能无收益 |
| V5 pooled/full-slot readout 不稳 | 先检验跨环境监督而不增加 slot 容量 | frozen 表示可能已缺乏所需动作/关系信息 |
| V1/V2 不稳定 | 泛化对象必须是完整 query compatibility，不是 generic saliency/eventness | 所有 localization/visual evidence 机制并未被否定 |

缺失证据包括：可信训练环境是否具有不同的 nuisance-label 关联；同一 semantic 任务在各环境是否有支持；query/video priors 是否解释每个环境的风险；普通 ERM+pair 是否已足够；DG 训练能否在不抹掉事件身份的情况下改善迁移。

## 4. 环境如何定义

环境只由 `inner_train_seen` 可用元数据与预先固定规则构造，不能读取 outer U test。第一轮优先使用已有负例构造类型，例如 action edit、object edit、其他有可信记录的机制。**环境标签附着于完整正负 pair 或 quartet，不只附着于负样本。** 若把所有 object-edit negatives 放一个环境、自然 positives 放另一个，环境就变成标签代理，不能研究机制泛化。

另用 Seen action/object/composition families 做 episodic task-family partition，模拟关系规则向另一语义家族的迁移；它们是语义任务家族，不是应该删除的 nuisance。视频来源、场景、camera 等环境只有在可信元数据存在且支持充分时才能使用；不可把视频 feature 的任意 k-means 簇直接称作“纯背景 domain”。

每个 pair/unit 必须保留标签、query 原文、video 身份、source linkage、edit mechanism、review provenance、环境归属及权重。环境不同不要求标签/语义难度分布完全相同，但必须报告支持量与 overlap；极少独立视频的环境不独立承担 worst-group 或 meta-test 梯度。

## 5. 模型与 compatibility 监督单位

初始使用 frozen canonical backbone 与同结构 existence head `s_θ(V,Q)`，保持主训练完整自然 BCE。优先使用与高优先级方向相同的可信 pairs，减少结构和数据差异。

同 query 跨视频的差值为

\[
d_q=s_\theta(V^+,q)-s_\theta(V^-,q).
\]

同视频跨 query 的差值为 `d_v=s_θ(V,q^+)−s_θ(V,q^−)`；它控制视频，但没有排除 query prior。标签齐全的四格差值为

\[
D=s(V_1,q_1)+s(V_2,q_2)-s(V_1,q_2)-s(V_2,q_1).
\]

四格约束与行/列排序共同使用，不能只要求总 D>0。Pure query-only 或 video-only 的加性分数在 D 中消去，但 scene-text interactions 仍可能存在。

环境风险 `R_e(θ)` 在固定 pair 类型/可信程度上计算，例如 `mean softplus(−d_q/τ)` 与已有四格行/列损失。`τ` 是全局预定常数，不能给每个语义 group 学自由 temperature 来伪装风险一致。对照与 DG 用完全相同的风险、pairs 和 score formulation。

## 6. 最小训练机制：episodic transfer of compatibility risk

在训练侧环境中分出 A 与 B，虚拟更新小 existence head：

\[
\theta'=\theta-\alpha\nabla_\theta R_A(\theta),
\qquad L=L_{nat}(\theta)+\lambda\{R_A(\theta)+\beta R_B(\theta')\}.
\]

这是将已有 MLDG 思想用于 compatibility 差值，**公式本身不是新的 meta-learning 算法**。初始拟议小 head 的准确一步梯度；若改一阶近似，必须单列变体并说明梯度实际保留什么，不能把 detach 后不再通过虚拟更新的普通 risk sum 叫 MLDG。

推理只使用共享 θ 的存在性 score；不访问 target labels、不做 target adaptation、无需环境标签，也不根据 U query 分配一个训练后新增专家。每个主 epoch 保留自然样本覆盖，辅助 episodes 只是 existence 分支的有限附加预算。

低成本替代对照包括：在相同差值上施加 IRMv1 penalty、带正则与支持门槛的 Group DRO。它们是已有方法对照，不与 MLDG 一次堆叠。第一轮不使用 noisy raw group-max，也不强制不同环境的平均 margin 完全相同：真正难度不同会产生不同风险，equal-risk 不是 compatibility 的必要性质。

## 7. 为什么可能改善 U+ vs U− AUROC

跨环境梯度需要在不同 construction/semantic family 上都帮助兼容性判别，可能削弱只在一种 source 相关性下有效的解。若模型学会“在视频中验证该 query 的事件”而不是“该语言编辑常为负”，这种规则可在新动作/新组合下保持部分有效。

只提升 pair risks 不保证跨 q pooled ordering 改善，故 `L_nat` 与原始 pooled 主指标不能删除。新语义视觉特征不足、环境未覆盖相关 shift、环境风险仅编码难度而非 nuisance 时，都可能无效。没有普适 OOD 或因果保证，不能借 IRM/DG 名称承诺这些性质。

## 8. 与 V1–V5 的区别

保留 V5 的 inner holdout 开发、Cq/Cv 控制与容量公平，V4 frozen localization，V3 全覆盖教训，以及可信 source-pair linkage。放弃 V3 主 loader 重采样、含噪 semantic-group 回退、把最差组 loss 无界放大，以及对整套 backbone 联合训练施加新目标。

与 Idea 04 query-marginal matching 的区别：04 改辅助分布、首轮目标仍 BCE；07 在**相同配对数据**上改变跨环境优化规则。与 Idea 01 四格监督的区别：01 验证关系监督；07 必须证明 environment-aware optimization 超过同 pairs 的关系监督，否则不新增 DG 贡献。

## 9. 最小实验与成功标准

固定 frozen 表示、head、pairs、环境 manifest 与 budget，比较 `natural BCE`、`natural BCE + ERM pair`、`natural BCE + compatibility MLDG`。IRM/Group DRO 只在后续作为相同风险上的强对照。先一 action 与一 composition inner pilot，再四 fold 三 seed；Novel-dev 不参与每 epoch checkpoint。

采用共同协议门槛：原始 inner Novel AUROC 建议 macro +2 pp、至少 3/4 fold 为正、Seen 下降 ≤1 pp，same-query/same-video/crossed 有一致收益；DG 要超过 ERM+pair，且跨 construction strata 不依赖单个环境。确认阶段再三 backbone，所有历史正式 split 结果标 exploratory。

## 10. 失败可排除的解释

| 观察 | 结论边界 |
|---|---|
| ERM+pair 有收益，DG 无额外收益 | 当前环境/预算下，关系监督足够；不支持独立 DG 主贡献 |
| 训练环境风险更均匀，但 Novel 无收益 | source risk robustness 未迁移；不能声称 invariant compatibility |
| pooled 提高而视频条件指标不变 | 可能是跨 q score comparability，不能证明视频证据提升 |
| semantic-family episodic 好，construction holdout 差 | 泛化能力有范围，不能统一称对所有 semantic shift 鲁棒 |
| DG 与延长 ERM 同样提高 | compute/优化曝光解释仍成立 |
| 可信支持多 seed 全无收益 | 当前环境划分与 frozen head 下 DG 不足；不排除视觉 feature/新环境设计 |

## 11. 最大 confounds、成本与依赖

最大混杂是把语义身份当 nuisance，或把环境变成正负标签代理。次要混杂是 meta-test 环境样本太少、head 训练不稳、额外计算和 hyperparameter 搜索。均需在设计阶段控制，不能事后选择最有利的 environment partition。

Frozen canonical teacher 可能已经训练过 training episodes 中的语义。episode holdout 只表示**该步 head 更新暂未用该环境**，不能称 teacher-unseen 或真正 pseudo-unseen；真正 Novel claim 只能由已有严格 inner semantic holdout 的完整 feature/teacher 数据边界支持。预训练 encoder 的语义暴露也应与任务级 unseen 区分。

成本中。小 head 的 episodes 可在 1–2 GPU 试验；如果 exact second-order 开销过高，可有限比较一阶变体，并设置 compute-matched ERM。准确时长需未来测量。此方向依赖高优先级 audit 或 ERM+pair 证明视频条件证据可改善，否则优先修 supervision/representation 而不是扩大 DG。

## 12. 论文故事与最强 objection

潜在故事是：semantic-novelty existence 要迁移的是 relationship，而不是把不同事件语义压成一致表示；配对条件风险与真实 negative construction 环境提供可检验的泛化对象，并通过 pooled 与两向条件评价说明何时 source robustness 转化为视频存在性迁移。已有 DG/temporal-debiasing 思想值得发展，新增空间在监督单位、环境有效性、评价目标和机制验证，详见 [RELATED_WORK.md](RELATED_WORK.md)。

最强 objection 是“把 MLDG/IRM 用在现成 head 上，没有新方法；提升只是 pairs 或更多计算”。只有相同 pairs/compute 的 DG 额外收益、事件身份保留、不同 construction 与 semantic holdout 的迁移边界，以及三 backbone 证据，才足以支撑方法贡献。若无这些证据，应合并为监督方法的消融或负结果，避免包装成一个复杂新版本。
