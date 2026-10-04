# Idea 10 交接：SHINE → GMR 与正式 Unseen 退化诊断

> 最新综合评估：当前A1/C1三模型50epoch训练/自动评测，以及v2/v3/v4均完成；本次补齐v4六组正式探索性Unseen评测，未启动新训练。主报告 `records/final_assessment/RESULTS.md` / `.json`，审计 `AUDIT.json`。相对匹配B0，Moment/QD/Flash的macro ΔU分别+.27/+.65/−2.45pp，Gap缩小分别−.19/+.36/−3.04pp。QD两折U点估计正向但CI均跨0；Flash C1 U −4.62pp，CI完全负向。v4低BCE=.01无pair（10ep）macro U+.75pp、Seen−.04pp、Gap缩小.78pp；A1 Seen−.42pp，U两折CI跨0，仍为探索候选。v4评测在 `records/ablation_test_v4/`。下一轮建议见综合报告，尚未授权/排程新训练；保持seed3407，不自动收尾原五split。以下为历史记录。

> 当前任务更新（2026-10-04）：v4低权重六组10epochs已完成，用户转向coarse/fine-only并要求检查Gap、训练50epochs。10epoch S1相对B0：A1 Gap32.15→32.55pp（扩大.40）；C1 18.19→17.55pp（缩小.64）；两折macro Gap25.17→25.05pp（缩小.12pp，相对.47%）。新saliency_v5已启动：从canonical重新训练总50epochs，B0与S1_saliency_only匹配forward和exposure，各split两任务并行，A1/GPU0、C1/GPU1。旧10epoch权重缺optimizer/RNG恢复状态，故不接续到60epochs。状态/日志见 `records/saliency_v5/`，计划/冻结见 `configs/SALIENCY_V5_PLAN.md` / `SALIENCY_V5_FREEZE.json`。新worker已安排完成后的自动正式探索性评测和汇总，勿另开重复评测。保持seed3407、Seen-val选模、不以U调参。以下为历史记录。

> 当前任务更新（2026-10-04）：v3 六组已全部完成；用户授权评测正式 A1/C1 Unseen 并进入下一步。七个10epoch适配方案+canonical各两折的独立探索性评测在 `records/ablation_test_v2_v3/`，checkpoint/选择规则/输入hash在读取test前已核验冻结。下一步v4权重依据v3 Seen结果预先固定并记录进评测gate：保留GMR+coarse/fine，Low_BCE_only=.01/0、Low_Pair_only=0/.02、Low_BCE_pair=.01/.02（BCE/pair有效权重）。v4已启动：A1/GPU0、C1/GPU1、每卡三任务、seed3407、10epochs、Seen-only。状态/日志见 `records/ablation_v4/`，计划/冻结见 `configs/ABLATION_V4_PLAN.md` / `ABLATION_V4_FREEZE.json`。不依据新Unseen结果更换v4系数。保持存活worker；A1/C1 U为已看过的探索集。以下为历史记录。

> 当前任务更新（2026-10-04）：第一步 v2 八组×10 epochs 已全部完成，checkpoint/hash/逐epoch exposure 已核验。S1 相对B0 Seen +.17/+ .57 pp，existence组合 −6.04/−3.07 pp（A1/C1）。用户已授权并启动第二步 v3：BCE_only、Pair_only、Weak_BCE_pair，各保留GMR+coarse/fine；BCE/pair有效权重分别1/0、0/.2、.1/.2。A1/GPU0、C1/GPU1，各三任务并行。当前状态和日志见 `records/ablation_v3/`，计划与冻结见 `configs/ABLATION_V3_PLAN.md` / `ABLATION_V3_FREEZE.json`；不要重启存活worker。只seed3407、10epochs、Seen-only，不读取formal U。完成后先读自动A1/C1汇总，再决定后续修复或泛化验证。以下保留历史记录。

> 2026-10-04 新指令：用户取消五 split 收尾优先级，授权直接针对 A1/C1 启动分项实验。旧训练快照和 P0 恢复指令已被此范围变更取代，不要重启旧 worker。新方案见 `../../configs/ABLATION_V2_PLAN.md`，运行入口 `code/train_ablation_v2.py`；状态/日志在 `records/ablation_v2/`。本次启动时两张 GPU 空闲，旧 formal worker 已不存活。以下为历史交接内容。

更新时间：2026-10-04 14:29（Asia/Shanghai）。本文是清空对话上下文后的首读入口。运行状态是此刻快照，恢复时必须重新读取 status，不能据此重启任务。

## 1. 用户目标、已授权事项与约束

用户要求在 `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/research_ideas/README.md` 增加第 10 个 idea，把 https://github.com/zxccade/SHINE.git 的思路迁入当前 GMR，并落实代码和训练。

最终目标是 [COMMON_PROTOCOL](../../../COMMON_PROTOCOL.md) 的正式 Seen→Unseen 退化缓解，不是 inner Novel-dev 提升。用户允许训练 50 epochs，明确不做多 seed，优先看效果。只用 seed 3407；当前只迁移 Moment-DETR-GMR，未迁移 FlashVTG/QD-DETR。

用户明确确认两项训练假设：batch 内轮换 query 可视为 absent；强制编辑层级距离链符合视频事件满足关系。不要重新索取这一授权，也不要把推翻这两项假设当成既定失败原因。有效标签仍可能在监督权重、难度组成和优化目标上与主任务不匹配。

用户曾说“现在任务完成”，随后重新要求评测已完成的 A1/C1，又要求分析原因。本轮最新要求是整理交接与未来任务文档。本轮没有启动新消融，也没有修改已有训练配方；已启动的其他 split 保持运行。

## 2. 当前必须记住的结论

Inner pilot 是正向的，但正式 A1/C1 没有复现。SHINE 相对同预算 baseline50：两 split macro Unseen −0.67 pp、Seen −4.06 pp。Gap 虽缩小，主要是 Seen 损伤，不能认定退化缓解。

最有证据的候选原因：本次额外加入的 rotated-negative existence BCE 在原 canonical checkpoint 上产生较大、部分冲突的初始梯度，损伤原任务；saliency 的 coarse/fine 代理目标没有稳定改善 decoder existence 的条件排序。唯一因果根因尚未确定，必须用分项消融验证。不要将本次联合方案失败直接写成 SHINE 原始思想无效。

未来任务执行顺序见 [NEXT_ACTIONS](NEXT_ACTIONS.md)，详细诊断见 [A1_C1_FAILURE_ANALYSIS](A1_C1_FAILURE_ANALYSIS.md)。

## 3. 代码、环境与隔离边界

工作目录 `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`，Idea 根目录 `research_ideas/10_shine_absence_aware_transfer`。

- `vendor/shine/`：上游物理快照，MIT LICENSE；上游 commit `dfbaab1cf8f6f88ca3c3a39bee2bbe8dd999f466`。
- `code/models/moment_detr_gmr/` 与 `code/training/moment_detr_gmr/`：当前 GMR 的独立物理复制。
- `code/shine_losses.py`：coarse 与 fine red ranking；处理 padding、空 GT 集。
- `code/prepare_bank.py`、`train_pilot.py`、`summarize_results.py`：已完成 inner pilot，保留历史冻结。
- `code/prepare_formal.py`、`train_formal.py`、`worker_formal.py`：正式五 split 的准备与训练。
- `code/evaluate_formal.py`：原冻结五 split evaluator，要求十个 run 均完成 50 epochs。
- `code/evaluate_formal_subset.py`：用户授权的 A1/C1 子集评测入口；要求对应四个 run 完成，不代表五 split 完成。
- `code/diagnose_gradient_conflict.py`：训练数据上的初始 checkpoint 梯度诊断，无 optimizer 更新。
- `code/clip/`、`code/_deps/`：本地 text-only CLIP 与依赖。text-only 模式绕开环境损坏的 torchvision nms 导入；编码数学未改。

训练/评测 Python：`/home/guoxiangyu/miniconda3/envs/gmr/bin/python`（torch 2.6.0+cu124）。语义解析 Python：`/home/guoxiangyu/miniconda3/envs/owvtg/bin/python`。两张 RTX3090 24GB。读取原始数据、特征与权重，未修改共享环境。

Canonical 权重：`/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2/<split>/moment/best.ckpt`。

Release 数据：`/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2/<split>/`。视频特征为 `/home/guoxiangyu/paper/新建文件夹/charades/vid_clip` 和 `vid_slowfast`；文本特征为原 GMR `features/semantic_existence_v2/<split>/clip_text`。视频输入 CLIP512 + SlowFast2304 + TEF2。

本迁移是训练机制迁移，保留原 GMR 架构；不是完整 SHINE-native global-token 模型复现。未 git commit/push。原已存在的 `plan.md` 删除和其他 ideas 的变更不属于本任务，不要恢复、覆盖或清理。没有用户要求的 sub-agent，不要自动启动多 agent。

## 4. 实际训练机制

原自然 GMR 数据流保留全部正负例，`mr_only=True`、原 criterion `lw_saliency=0`。SHINE arm 每 batch 对 text/mask roll(-1)，重新执行完整 pre-fusion forward。

总损失：`L_GMR + L_coarse + L_fine + L_rotated_BCE + 0.2 L_exist_pair`。

- coarse：GT 内 topK 均值与 GT 外最大值 margin 1；自然 query 与轮换 query 在 GT 内的 topK 均值 margin 2；q=8，仅正例。
- fine：每 batch 最多 4 条正例编辑链，三层 query 新鲜融合；距离链 GT→positive→level1→level2→level3→rotated，四个相邻 margin .25。
- fine 距离按 upstream 为单侧 `−stopgrad(target) log(sigmoid(prediction))`（soft target 先 sigmoid），不是 KL 或完整 BCE。
- rotated BCE 标签均为 0；existence pair margin .2、系数 .2，仅正例。
- 编辑三层只监督 saliency 距离，不统一赋 absent existence 标签。
- saliency：encoder 视频 token 的 linear head；existence：decoder slots pooling 后的 MLP，两者仅通过共享表示间接联系。
- 原 clip_length=1 的 GT mask 映射，无独立验证提取 timestamp。

Inference 仅使用自然 V/Q，不使用 GT、编辑 bank 或 batch 伙伴。前景类别是 softmax class 0；top1 span 按该前景分数选取。

## 5. 已完成 inner pilot（不要当作正式 U）

Strict-inner folds：A1_action_01、C1_composition_01。从各自 V5 checkpoint 初始化；每 arm 10 epochs，lr1e-5，bsz16，seed3407。按 Seen-val pooled AUROC 选最早最佳 epoch，Novel-dev 不选 epoch。

| Fold | baseline Novel | SHINE Novel | ΔNovel |
|---|---:|---:|---:|
| A1_action_01 | .6122 | .6541 | +4.18 pp |
| C1_composition_01 | .5455 | .5813 | +3.58 pp |

Macro Novel +3.88 pp，Seen +1.32 pp。固定单 seed 视频聚类 paired bootstrap 下，两折 ΔNovel 区间下界为正。定位与条件排序未一致改善。结果在 `records/RESULTS.json`、`PILOT_RESULTS.md`，冻结在 `configs/PILOT_FREEZE.json`。不得覆盖这些历史记录。

## 6. 正式五 split 方案与实时状态

方案见 `FORMAL_EXPERIMENT_PLAN.md`；冻结在 `configs/FORMAL_FREEZE.json`。A1、A2_alt、A3、C1、C2_alt 各自 canonical 初始化，baseline 与 SHINE 各 50 epochs，lr1e-5、bsz16、原 weight decay、grad_clip .1。只有 Seen-val AUROC 选模，平局最早；本配方不允许 epoch0 作隐式 fallback。不存在 U 选 checkpoint、threshold 或超参。

正式数据/编辑 bank 已全部准备，只排除 formal held semantics；正式 train / Seen-val 视频无重叠。代码冻结 SHA 在交接时校验全部通过。新诊断/子集脚本不在原冻结文件集合内；原冻结文件保持不变。

| Split | baseline50 | SHINE50 | 正式 test 评测 |
|---|---|---|---|
| A1 | completed 50 | completed 50 | 已完成 canonical/baseline/SHINE |
| A2_alt | completed 50 | running，最近写入 22 | 未评测 |
| A3 | 未开始 | 未开始 | 未评测 |
| C1 | completed 50 | completed 50 | 已完成 canonical/baseline/SHINE |
| C2_alt | completed 50 | running，最近写入 3 | 未评测 |

GPU0 队列：A1→A2_alt→A3；GPU1：C1→C2_alt。每 split 先 baseline 后 SHINE。worker 状态 `records/formal/worker_cuda_0.json`、`worker_cuda_1.json`；run 状态 `runs/formal/<split>/<arm>/seed3407/formal_v1_50ep/status.json`；日志 `records/formal/<split>_training_50ep.log` 与 `records/worker_formal_gpu{0,1}.log`。

原 exec sessions 23099/14471 可作历史线索，不应假设清空上下文后仍能 poll；以 status、日志及真实进程为准。旧监控 cell 78 已终止，仅停止监控，未停止训练。没有自动 evaluator 在等训练结束。

## 7. A1/C1 正式结果

| Split | Arm | Seen AUROC | Unseen AUROC | Gap |
|---|---|---:|---:|---:|
| A1 | canonical replay | .8034 | .4956 | .3078 |
| A1 | baseline50 | .8088 | .4858 | .3231 |
| A1 | SHINE50 | .7517 | .4799 | .2718 |
| C1 | canonical replay | .7611 | .5683 | .1928 |
| C1 | baseline50 | .7606 | .5747 | .1858 |
| C1 | SHINE50 | .7365 | .5672 | .1694 |

SHINE−baseline：A1 ΔU −.59 pp，95% CI [−3.98,+2.86] pp；C1 ΔU −.76 pp，CI [−2.94,+1.51] pp。两个 Seen Δ 分别 −5.71 / −2.40 pp，配对区间均低于 0。Macro 仅这两 split：Unseen −.67 pp、Seen −4.06 pp、Gap reduction +3.39 pp。

最佳训练 epoch：A1 baseline10 / SHINE22；C1 baseline1 / SHINE1。是训练满50后按 Seen-val 选出的模型，不是直接测试最终 epoch。

核心记录：`records/formal/A1_C1_FORMAL_RESULTS.{json,md}`、`A1_FORMAL_RESULTS.json`、`C1_FORMAL_RESULTS.json`，六组全精度 test_predictions.jsonl。评测前 gate：`A1_C1_EVALUATION_CHECKPOINT_FREEZE.json`；实际执行源码：`A1_C1_EVALUATOR_EXECUTED.py`，其 SHA 与 gate 匹配。

COMMON_PROTOCOL Moment 历史五 split macro 是 .7518/.5287/.2231；不可直接与两个 split macro 作增量比较。Canonical replay 与旧 diagnostics 也略有差异，旧分数精度/评测口径需单独说明，不计作方法收益。

五 split `FORMAL_RESULTS.json` 尚未生成；它与 A1_C1 子集汇总是不同文件，不能混用。子集 runner 后续修正为独立 gate 文件名；本轮实际已执行版本保留在 records 中。

## 8. 已完成的原因诊断与证据强度

详细论证见 `A1_C1_FAILURE_ANALYSIS.md`。以下检查都未更新模型。

1. 训练曲线：A1 SHINE Seen-val epoch0 .8367→epoch1 .7743，最佳 .7824；C1 .7883→.7599，最佳就是 epoch1。更多 epoch 未恢复。证据支持新增目标损伤既有能力，而非仅 U 难。
2. 原任务 train loss：epoch1 A1 baseline .947 vs SHINE1.551，C1 1.536 vs2.091。
3. 初始 train-only 梯度：每 split 三个固定 minibatch，关闭 dropout，只看共享 transformer/input projections。rotated BCE 范数为 GMR 的 A1 2.01–4.28 倍、C1 2.19–3.06 倍；6个batch中5个 cosine 为负。记录 `GRADIENT_DIAGNOSTICS.json`，日志 `gradient_diagnostics.log`。局部诊断不是完整训练因果证明。
4. 监督组成：原 train 负例比例 A1 17.43%、C1 14.26%；加同权重全负轮换 BCE 后，两个 existence BCE 的有效负例权重57–59%。标签有效也不保证强度/难度合理。
5. Score 下压与排序：A1 U+/U− 平均logit9.296/9.434→−1.004/−.815；整体下降并未恢复正负排序。阈值/共享单调校准不改变 AUC。
6. 条件指标：U source PairAcc A1 .6580→.5519、C1 .7126→.6693；same-query A1 .4825→.4381、C1 .4612→.4787，未一致改善。SHINE shuffled U AUC A1 .4895、C1 .5805，与自然输入 .4799/.5672 接近甚至更高；只是依赖性诊断，不证明完全不使用视频。
7. 负例 strata：A1 U− 964/1119是 action counterfactual，此类 AUROC .4525→.4571，仍低于.5；object类 .7474→.6416。C1 U− 267/270是object counterfactual，该类 .5744→.5669。不能仅凭 C1 名字把全部失败归因于composition；3条composition负例不足以强归因。记录 `FAILURE_SCORE_DIAGNOSTICS.json`。
8. 时间格：duration/feature_length 中位数约1秒，没有 GT start 被压到超长末尾的行；GT end 截断 A1 518/7108、C1 625/9016。无明显倍数尺度错误证据，但缺少独立提取timestamps。记录 `TEMPORAL_GRID_DIAGNOSTICS.json`。
9. Coarse/fine 值下降不等于existence收益；saliency和decoder existence目标间接相连。尚未做分项消融，不能确定fine链就是根因。

缺失机制证据：配套 Cq/Cv 重训、独立验证闭合四格、temporal-only/rotated-existence-only 消融。未启动这些任务；不得把它们写为已完成。

## 9. 清空上下文后怎样恢复

先读本文与 NEXT_ACTIONS，然后重查十个 run status、worker日志、进程和冻结 SHA。保持既有训练。十个 run 完成后，用 gmr Python 执行原 `code/evaluate_formal.py --device cuda:0`，它先核验再读取所有正式test，产生完整五 split报告。没有自动后台评测预约。

后续新消融需要独立入口、run ID、freeze与记录；不要直接改原train_formal.py或shine_losses.py，否则正在运行的旧队列及最终评测冻结校验会被破坏。用户清空上下文后可复制 NEXT_ACTIONS 中的恢复提示启动下一轮。此次请求只整理文档，不代表已启动这些未来实验。
