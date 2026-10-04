# Idea 05：独立的视频—查询存在性适配器

状态：`draft_not_executed`。本目录只定义研究方案，没有新增训练、推理结果或已实现模块。对应原研究文档 Idea 8；优先级编号用于安排下一阶段，不表示版本继承。

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

**本方向先做什么：** 首轮比较 source-adapted pooled/slots 路径与 raw video sequence 的轻量存在性旁路，定位输出冻结。研究的是表示路径和专用兼容性学习，不是“更多 token 必然更好”。

| 边界 | 本 idea 的具体约定 |
|---|---|
| 复制来源及最小依赖 | 从 B 复制 Moment baseline 完整 models/training/configs，使用正确 inner checkpoint 做严格 Novel 对照；复制 V5 bank/metrics/readout helpers 作为本地 comparator。Raw input 从 dataset 加载原始 CLIP→SlowFast 及 query token features，不假设它们已在可直接 cosine 的同一空间。 |
| 在本目录新增/改造什么 | 新增 `code/idea_method/compatibility_adapter.py` 及本地 full-sequence dataset/训练入口。保留 baseline forward 的 spans/raw scores，新 branch 单独学习 video/text 投影与轻量交互；不改原 decoder、criterion、checkpoint 或 feature npz。 |
| 本方法特别要记录的数据 | 记录 pooled/slots/raw 输入形状、归一化、mask、TEF 是否进入、新参数量/有效容量/forward预算。新 sequence cache 写 `artifacts/`；记录 raw-mean/text-only/video-only controls，禁止把不同预算收益归为 representation 信息损失证明。 |
| Baseline 与机制对照 | 首轮使用 feature relative indices，不能当秒级对齐已成立；任何截断/对齐调整只在本地副本并单列。各路径 supervision、selection 一致，raw 路径不能额外获得更好 negative。独立 score 初轮不与 canonical logit 任意混合。 |
| 扩展到另外两个 backbone | 从 A 复制三模型用于保持各自定位输出并作对照；同 raw adapter 可在多个 backbone 外提供存在性 score。评估分开记原 baseline、旁路和 gate，不能将共用旁路同一份 score 视为三个独立训练的统计样本。 |

**实施阅读顺序：**先完成本节的来源与隔离记录，再读下方机制定义；依据本目录 `EXPERIMENT_PLAN.md` 建立 arms、支持与成功/失败判据；`RELATED_WORK.md` 用于最近邻方法适配和贡献边界。本次任务只补充文档；后续代码复制与实验依据计划在本 idea 内独立实施。单独复制本 idea 目录后，本节中的 A/B 绝对路径仍能定位本机资产；迁移机器时用 SOURCE/DATA manifest 映射路径，不能假定原视频、权重和缓存随 Markdown/Git 一起迁移。

## 1. 研究问题与定位

目标是解释并缓解 semantic novelty 下的 `AUROC(U+ vs U−)` 退化。U− 是完整视频中没有 query 描述的事件，不是视频没有任何事件。Moment-DETR-GMR、FlashVTG-GMR、QD-DETR-GMR 的共同退化说明，应优先检查共享监督、source adaptation 与存在性证据，而不是预设某一种 decoder 的 pooling 有问题。

本方向问一个具体问题：**在相同 Seen existence 监督下，直接读取冻结的原始视频特征序列，能否比读取 source-localization 适配后的 pooled/slot 表示，更稳定地支持新语义存在性判别？**

如果答案为是，才有理由发展独立于定位训练路径的 compatibility 分支。该问题不是“更大的 head 是否更好”，也不是“raw features 一定更泛化”。

## 2. 核心假设与三个 baseline 的共同问题

- H05.1：定位导向的表示变换、训练与 checkpoint 选择可能保留 source localization 有用的信息，却不利于新语义 present/absent discrimination。
- H05.2：原始 CLIP/SlowFast 序列可能仍保留可用于 query-conditioned existence 的信息；V4/V5 的 decoder readout 失败没有否定它。
- H05.3：一个小型 existence 专用 adapter 可能在不改变原定位链的条件下，学习更可靠的视频条件证据。

FlashVTG 的存在性路径与两种 DETR 的路径不同。因此跨三模型的主张应是“存在性需要专门的关系学习路径”，不能写成“三模型都因 max-pooled decoder slots 失去信息”。若同一个原始特征 adapter 可配接三个模型，它首先证明 backbone-independent 的旁路存在性建模可行，不直接证明三个模型内部都发生了同一种表征损失。

## 3. V1–V5 的支持、反证与未决项

| 证据 | 对本方向的约束 |
|---|---|
| V1：定位/phrase correspondence 增强未稳定改善 existence | 不能把存在性继续寄托于定位改善；但没有证明独立分支必然更好 |
| V2：generic visual residual/eventness 不稳 | adapter 必须读取 query 并学习兼容性；video-only eventness 不是本方向 |
| V3：联合 sampler/ranking 损害 Seen/localization | 主数据覆盖不变，定位路径冻结，避免重复梯度与分布混杂 |
| V4：pooled-state residual correction 约 +0.0004 且 CI 含 0 | 独立直接打分，不默认给 canonical scalar 加 residual；不能由 V4 推出 raw feature 无信息 |
| V5：R1/R2/R3 未稳定改善 Novel；R3 对 R2 不稳 | 更完整 decoder slots 不是已确立瓶颈；原始序列与 decoder 表示需要受控比较 |
| V5：Cq pooled Novel 提升、source PairAcc 下降 | 原始 adapter 的收益必须通过同 query 换视频与 crossed evaluation 证明，不能仅看 pooled |

原始证据入口：[主研究文档](../../semantic_novelty_GMR_next_stage_research.md)、[V5 readout 结果](../../experiments/moment_detr_gmr_evidence_v5/P3_READOUT_RESULTS.md)、[V5 readout 实现](../../models/moment_detr_gmr_evidence_v5/readouts.py)。

## 4. 输入、输出与冻结边界

输入为项目已有的冻结序列：`Xclip∈R^(T×512)`、`Xsf∈R^(T×2304)`、query token features `Q∈R^(L×512)`，附视频/text 有效 mask。维度由已有 readout 与 time-grid audit 支持；实际样本仍应按 feature manifest 核验。不同模态长度按现有经过核验的对齐约定处理，不能自行补插值并假定其时间准确。

定位 backbone、预训练特征、定位 head 与 canonical 输出全部冻结；只训练新投影、交互层与 existence scorer。输出一个独立的 raw existence logit `e(V,Q)`。原 baseline score 保留作为评价参照，首轮不把两个 score 任意混合，否则无法区分旁路的信息与混合权重的收益。

全序列可用 `t/max(T−1,1)` 的相对索引作为位置。它表达采样顺序，不是已经核实的秒级时间。当前 [time-grid audit](../../experiments/moment_detr_gmr_evidence_v5/time_grid_audit.json) 明确记录 local ROI gate 尚待 extraction timestamp provenance；因此首轮不裁 GT 秒级 ROI、不报告新的时间定位精度。

## 5. 具体机制：学投影后的轻量交互

建议 pilot 使用小维度 `d`，例如 128，单层 text-conditioned 交互，参数仅为待实施建议：

\[
z_t=P_v\operatorname{LN}([x_t^{clip};x_t^{sf}])+p_t,\qquad u_l=P_q\operatorname{LN}(q_l).
\]

视频与 text 经过各自学习投影。**不能因为某些输入都叫 CLIP features，就把 mean text tokens 与 512 维 video vector 直接 cosine 当作已经对齐的共享空间。** token hidden states、pooled text embedding、不同视觉提取阶段可能不同；本方案不依赖这一未经核实的等价关系。

\[
\tilde z_t=\operatorname{FFN}\left(\operatorname{LN}[z_t+\operatorname{Attn}(z_t,U,U)]\right),
\quad q_*={\sum_l m_lu_l\over\sum_lm_l},
\]

\[
r_t=\operatorname{MLP}([\tilde z_t;q_*;\tilde z_t\odot q_*;\tilde z_t-q_*]),
\quad e=\tau\log\left({1\over T_{valid}}\sum_{t\ valid}\exp(r_t/\tau)\right).
\]

首轮 pooling 和温度在 pooled、slots、raw-sequence 三臂保持一致；该聚合本身不是新增机制，V5 已有 normalized logsumexp。不要同时新增 witness parser、背景中心化或 hard-negative pipeline，以免一个 positive result 无法归因。

使用原自然样本覆盖的二元监督：

\[
\mathcal L_{exist}=\mathbb E_{(V,Q,y)\sim D_{inner\ train}}\operatorname{BCEWithLogits}(e,y).
\]

若前置四格方向已确定可信 pair supervision，可在三臂上加入完全相同的辅助 pair loss，并作为第二个预定试验层；不能只给 raw-sequence 臂增加更好负例。

## 6. 为什么可能改善 Unseen AUROC

如果 source-adapted 表示把不存在的细粒度条件抹平，raw-sequence interaction 可能重新利用这些条件区分 U+ 与 U−。冻结定位避免为新的存在性目标损害定位结果；专用分支避免 existence 与定位共享优化目标。

这只是可检验机制。新的 adapter 仍可能学习 query prior、视频背景或 feature 长度；独立分支本身不保证 semantic invariance。主目标是提高 U+ 对 U− 的排序，不是提高阈值拒绝率或降低所有 Novel scores。

## 7. 最小实验与推进条件

完整方案见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。四个既有 inner semantic folds 中，先比较同监督、同交互 head 家族的 pooled、slots、raw-sequence。query-only 与 video-only 为必要控制。

成功需要 raw-sequence 在 pooled Novel AUROC、same-query cross-video 判别和 crossed evaluation 上一致优于受控 comparator；same-video PairAcc 不明显下降；Seen 保持在共同协议容差内；shuffled-video 下不能完整复现收益；原定位输出保持一致。

“parameter matched”不等于计算量、有效容量、token 数量和优化条件 matched。输入投影维度差异需要单独记录，比较头家族与近似参数预算，同时报告 wall time/forward cost。即使 raw-sequence 获胜，也不能直接把所有差异归因为 decoder 丢信息。

## 8. 失败后可以排除什么

| 结果 | 合理结论 |
|---|---|
| raw-sequence 三类指标皆无收益 | 当前特征、轻量模型与监督下，“便宜旁路足以恢复泛化”解释被弱化 |
| pooled 提高、same-query/crossed 不提高 | 不能声称更好的视频条件证据，先排查 query/length/scene prior |
| raw-sequence 和 slots 同幅提高 | 可能是共同 head/训练改变，缺少 raw 信息优势证据 |
| 训练不拟合且 Seen 同样差 | 主要反映优化或容量不足，不可据此否定原始表示的信息 |
| 条件指标好、pooled 仍差 | 更可能有跨 query score comparability 问题，可转向参考中心化方向 |

任何失败都不能证明“CLIP/SlowFast 不含相关信息”或“所有独立存在性路径无效”。

## 9. 最大 confound 与控制

1. 序列长度、输入维度、位置编码与训练参数同时改变：分开报告，保持后三者尽可能相同，增加 raw mean 旁路对照。
2. raw mean 或 text-only 已提高 AUROC：不能把主模型收益归因于交互，要看同 query 与 shuffled-video。
3. teacher/定位 backbone 已见过 inner Novel semantics：必须用对应 inner-trained baseline，不能偷用全 Seen canonical checkpoint。
4. raw adapter 的训练预算更多：固定样本曝光与 checkpoint selection，记录真实优化步数。
5. background–query interaction shortcut：crossed tests 超过加性 prior 仍不是完整 grounding 证明，保留 counterfactual queries 与 nuisance-matched 对照。

## 10. 保留与放弃

保留 V4/V5 的冻结定位、干净 inner holdout、原始 logit、缓存身份核验、Cq/Cv controls 与 video-cluster uncertainty。放弃 V1 phrase losses 的默认继承、V2 generic eventness、V3 主 sampler 替换、V4 bounded scalar correction、V5 更大 slots 自动更好的假定。

## 11. 成本与研究贡献

成本为中：已有特征，无需重提视频或微调大模型；序列计算较向量 readout 多，GPU 时间需未来实测。1–2 GPU 可做顺序 pilot，不承诺固定小时数。

独立 head 本身已存在大量先例。可发展的贡献是：通过受控表示路径比较，确定 semantic-novel existence 需要什么证据；给出可保留定位输出的适配器；证明提升来自 label-changing 视频响应而非 Cq。若 raw-sequence 获胜后还能跨三 backbone 复现，才有理由形成通用 GMR 机制论点。[文献边界与 novelty](RELATED_WORK.md)

## 12. 最强 reviewer objection

“这是普通 cross-attention head，比 decoder readout 更大或输入更多，所以提高不说明新机制。”应对不是宣称结构原创，而是相同监督、容量/计算披露、raw mean/slot/text 控制、条件指标与跨 backbone 验证。若仍无法区分，需要把贡献限定为表示与监督的实证发现，而不是宣称因果证明。
