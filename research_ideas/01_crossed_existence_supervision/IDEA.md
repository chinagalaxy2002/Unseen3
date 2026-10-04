# 01 · Crossed Existence Supervision：闭合四格的存在性监督

状态：`draft_not_executed`。原总结 Idea 1；推荐方向 A，优先级 01。实施成本：低至中，主要不确定性是可靠闭合四格的数据支持量。来源：[用户总结](../../semantic_novelty_GMR_next_stage_research.md)、[共用证据与协议](../COMMON_PROTOCOL.md)。本目录提出未来方案，不包含新增实验结果。

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

**本方向先做什么：** 首轮选 Moment-DETR-GMR；冻结对应 inner checkpoint，比较相同 head 的自然 BCE、四格额外曝光 BCE、row ranking 和 row+column ranking。研究的是存在性监督的可识别性，不是新定位 backbone。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 `models/moment_detr_gmr/`、`training/moment_detr_gmr/` 和 `configs/moment_detr_gmr/`；只将 B 的 V5 `feature_bank.py` / `metrics.py` / `train_readout.py` / `protocol.py` 等所需依赖当模板复制到本 idea。V5 helper 若间接引用 V4 sampler/helper，也要本地复制并登记，不能导回旧工程。标签和四个 fold manifests 物理复制；已有 inner banks/checkpoints 只读。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/crossed_loss.py`、`code/idea_method/quartet_sampler.py` 及本 idea 的训练/评价入口。Natural BCE loader 全量不变；quartet stream 独立采样。冻结 baseline 只提供各真实 (V,Q) 的表示和定位输出；每个 cell 独立正确计算。 |
| 本方法特别要记录的数据 | 在 `data/` 保存 exact-query 编码/文本 hash、四格四个 cell 的标签来源、视频/query 重用与 unique matrix 支持；记录 C0–C3 曝光和 gradient/loss scale，joint smooth-max 仅作为后续独立 arm，采用非负 loss。 |
| Baseline 与机制对照 | 不能从未标注视频补 absence，不能把 same-semantic 当 same-query。Baseline arms 使用本地未加方法的 head recipe；method 只改变对应 auxiliary loss。换 query/video 在 fusion 前重算，不交换原 conditioned slots。 |
| 扩展到另外两个 backbone | Flash 复制 A 的 `models/flash_vtg_gmr/`、`training/flash_vtg_gmr/`、`configs/flash_vtg_gmr/`，接 scalar logit；QD 复制 A 的三组 `qd_detr_gmr/` 包。若做严格 inner Novel，需另建未见对应 holdout 的 Flash/QD task checkpoint，不能直接套 canonical 权重。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. Idea 与核心假设

把存在性训练的最小监督单位从孤立 `(V,Q,Y)`，扩展为两段视频、两个自然 query 的已核验 2×2 标签矩阵。对同一 Q，视频改变时标签翻转；对同一 V，query 改变时标签也翻转。模型需要同时满足两种条件排序，无法仅靠 query prior 或 video prior 解开所有约束。

核心假设是：三个 backbone 的共同退化，部分来自训练仅要求 source 分布的绝对二分类，而未充分约束“这个 query 为什么在这个视频里存在”。若条件可识别监督已经充分而仍退化，该方案应在 held-out semantic 条件排序上失败，因此可证伪。

## 2. 针对三个 baseline 的共同问题

三个 backbone 有不同的时序网络和 existence representation，但共享按 source 正负分布训练、输出单个 existence score 的任务设置。同视频的自然描述与编辑 negative 可以具有不同语言统计；BCE 不要求在保留 Q 的情况下改变 V 后判断翻转。四格补充的是标签之间的关系，不依赖 Moment-DETR-specific decoder 模块，理论上可接入三个 backbone 的 scalar logit。

不能说现有 BCE 一定完全没有条件信号。四格用于检验“条件监督不足是否是可缓解的共同原因”，并明确其支持范围。

## 3. 已有实验支持、反对与缺失证据

- 支持提出：V5 Cq 在四折 pooled Novel AUROC 上都获益，但 source-pair PairAcc 四折都下降，说明主 AUROC 与条件比较可能给出不同结论。Same-video ranking 本身仍可由 query prior 部分完成，因此应增加 same-query 翻转。
- 支持设计约束：V3 的 semantic-group pairing 与 exact-query pairing不同，并且 sampler 丢失大量 S+，其失败不能排除保持自然主流的条件监督。
- 反对盲目乐观：V3 的 ranking 组合未稳定提升；V5 多种冻结 readout 无稳定改善。即使监督正确，现有 pooled state 仍可能不足以支撑新语义。
- 缺失：exact query、两类标签、独立视频组成的四格覆盖；四格是否主要由少数 easy scene difference 支撑；保持数据和预算时四格相对双向 ranking 的增益。

## 4. 与 V1–V5 的关系

保留 canonical baseline、严格 inner folds、冻结 feature replay、Cq/Cv controls、source-pair provenance 和自然训练覆盖。放弃版本连续性、semantic-group balanced 主 sampler、generic eventness、默认 full-slot readout，以及“定位提升自然带来存在性提升”的预设。

本方向改变监督信息和条件评估。首轮使用简单 pooled readout 就足够；如果无效，再通过 05 的原始特征 adapter 区分表示与监督，不直接把失败解释成四格思想无效。

## 5. 模型与训练机制

### 5.1 核验监督矩阵

例如 Q1 为“打开门”，Q2 为“关闭门”；V1 只满足 Q1，V2 只满足 Q2。现实中视频可能同时包含两种动作，因此必须核验整段 absence，不能从一个标注 interval 推出其反动作不存在。可以使用其他动作/物体组合，不预设反义词必然互斥。

| | Q1 | Q2 |
|---|---:|---:|
| V1 | 1 | 0 |
| V2 | 0 | 1 |

只取已有标签或经可靠核验成立的四格。保持每格实际文本、视频和编码 hash；不能把“同 action group”的不同句子混成同一 query。按视频、query 与矩阵去重，交换行列不增加独立样本。

### 5.2 四个 margin

令 `sij=sθ(Vi,Qj)` 为未舍入 existence logit：

\[
d_1=s_{11}-s_{12},\quad d_2=s_{22}-s_{21},\quad
d_3=s_{11}-s_{21},\quad d_4=s_{22}-s_{12}.
\]

前两项固定视频，后两项固定 query。基础辅助损失为：

\[
L_{\mathrm{cross}}=\frac14\sum_{k=1}^{4}\operatorname{softplus}(m-d_k),\qquad
L=L_{\mathrm{natural\ BCE}}+\lambda_e L_{\mathrm{aux\ BCE}}+\lambda_r L_{\mathrm{cross}}.
\]

主 BCE 的全样本覆盖和自然比例保留。Auxiliary 四格是独立监督流，新增曝光在 comparator 中匹配。`m, λe, λr` 仅为待冻结超参数，首轮不堆叠更多损失。

**这一平均损失在数学上就是该四格上的双向 pairwise ranking。** 不把它包装成新损失。贡献首先来自闭合、可靠标签与同时排除两种单模态解释的训练/评价组织；其必要性需与已有关联排序比较。

若基础版有效，可单独研究最差约束的联合优化：

\[
L_{\mathrm{joint}}=\operatorname{softplus}\left[\tau\log\left(\frac14\sum_{k=1}^{4}\exp[(m-d_k)/\tau]\right)\right].
\]

该 smooth-max 让一个容易的 row 不能掩盖失败的 column。外层 softplus 保证 loss 非负并在全部 margin 足够大时趋于 0，避免裸 log-mean-exp 随 margin 增大趋于负无穷。它与平均 pairwise loss 的区别是梯度集中方式，需同四格、同预算消融；温度控制硬约束近似的平滑度。不是首轮成功的必要前提，也不预先认定其具有 D 类 novelty。

### 5.3 交互差分只作诊断

\[
D=s_{11}+s_{22}-s_{12}-s_{21}.
\]

对于任何 `s(V,Q)=a(Q)+b(V)`，D=0，且四个严格 margin 不可能全正。D>0 可以由少数很强的 margin 抵消失败 margin，不能只优化或报告 D 就声称四格正确。纯 query / video 模型在对称加权四格的 pooled AUC=0.5；任意自然 pooled set 并无该保证。

### 5.4 评分路径

第一轮仅训练同容量 existence head，冻结其对应的严格 inner baseline。所有四格 cell 都必须有真实 `(V,Q)` 条件下的表示。已有 feature bank 只可复用其中已经算过的组合，不能拿 Q1-conditioned decoder slots 作为 Q2 的输入。若未来加入新组合或 shuffled-video，必须在 fusion 前替换输入并重算。

## 6. 为什么可能改善 AUROC(U+ vs U-)

在自然 BCE 下，source query frequency 可以降低分类风险；语义改变后这些偏置可能不再保持正负排序。四格要求同一文本在不同视频上翻转，使语言频率不能解决 column ranking，同时要求同视频的 query 区分，使 video eventness 不能解决 row ranking。若 backbone 的表示含可迁移关系信号，辅助监督可把 head 的容量转向这部分信号。

但是，conditional ranking 不保证跨 query pooled AUROC。它只约束差值，不完整控制不同 query 的 scale/offset。因此主 AUROC 必须与条件指标一起验证；仅有条件改善时再考虑 02，不能把 group AUC 当作主指标替代。

## 7. 最小实验

完整计划见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。一至两张 GPU 上训练同容量小 head，比较 natural BCE、相同四格额外曝光的 BCE、same-video ranking、双向四格 ranking。先利用已标注的支持，再决定是否值得人工扩充。四个 inner folds，第一轮一个 seed；不联训 backbone。

## 8. 成功判据

Inner Novel pooled AUROC 与 same-query / quartet 指标同时改善；至少多数 fold 正向，Seen 保持；Cq/Cv 在对称支持集不能解释结果；四格版应优于同数据曝光 BCE 和单向 ranking。建议推进门槛见共用协议。若仅条件改善，记为条件兼容性成果及下一步打分问题，不宣布已解决 AUROC degradation。

## 9. 失败能排除什么

- 数据核验后、训练能拟合且 Novel 不改善：削弱“现有 pooled state 只缺这类条件监督就能迁移”的解释。
- 双向不优于 same-video：在当前支持集下，column 约束未带来新增价值；检查样本多样性与效应量。
- Train 都不改善：不能排除研究假设，先分清优化、表示容量、矛盾标签或支持不足。
- 只训练在四格上的指标提高：不能排除支持选择偏差，也不能宣布自然主 benchmark 得益。

## 10. 最大 confound

四格 rare subset 与自然数据分布不同，可能由少数场景/演员差别和语言模板决定。模型可以学习 scene×query interaction，虽然不是纯单模态函数，却未必识别真正事件。需要自然语言 negative、相近场景视频、相同数据曝光、以及输入前 shuffled/counterfactual 对照。小样本 quartet 共享节点，不能当作大量独立观测。

## 11. 文献与 novelty 定位

最近邻是 D-TSG 的双轴对比、MVMR/CroCs 的跨视频检索、Winoground 的交叉矩阵及 SHINE 的 semantic hard negatives，详见 [RELATED_WORK.md](RELATED_WORK.md)。基础排序形式已有；候选贡献定位 **B/C**：将经过核验的双条件翻转组织成 GMR existence 的训练和审计单元，并针对 task-semantic holdout 上的 pooled / conditional 分离建立统一验证。如果未来加入 smooth-max，仅在证明其相对均值双向排序有稳定增益后再主张新增机制。

## 12. 最强 reviewer objection 与回应证据

“这只是 Winoground-style 数据加双向 ranking；提升来自额外 negatives，并无方法贡献。”回应不能靠更名：必须给相同四格曝光 BCE、单向/双向/联合损失对照，证明 closed support 的条件可识别性与 semantic holdout 收益；再跨三个 backbone 验证。若联合损失无新增贡献，诚实定位为监督 formulation 和评价贡献，而不声称全新对比学习。
