# Idea 10 下一轮执行任务

> 最新综合评估：当前A1/C1三模型50epoch训练/自动评测，以及v2/v3/v4均完成；本次补齐v4六组正式探索性Unseen评测，未启动新训练。主报告 `records/final_assessment/RESULTS.md` / `.json`，审计 `AUDIT.json`。相对匹配B0，Moment/QD/Flash的macro ΔU分别+.27/+.65/−2.45pp，Gap缩小分别−.19/+.36/−3.04pp。QD两折U点估计正向但CI均跨0；Flash C1 U −4.62pp，CI完全负向。v4低BCE=.01无pair（10ep）macro U+.75pp、Seen−.04pp、Gap缩小.78pp；A1 Seen−.42pp，U两折CI跨0，仍为探索候选。v4评测在 `records/ablation_test_v4/`。下一轮建议见综合报告，尚未授权/排程新训练；保持seed3407，不自动收尾原五split。以下为历史记录。

> 当前任务更新（2026-10-04）：v4低权重六组10epochs已完成，用户转向coarse/fine-only并要求检查Gap、训练50epochs。10epoch S1相对B0：A1 Gap32.15→32.55pp（扩大.40）；C1 18.19→17.55pp（缩小.64）；两折macro Gap25.17→25.05pp（缩小.12pp，相对.47%）。新saliency_v5已启动：从canonical重新训练总50epochs，B0与S1_saliency_only匹配forward和exposure，各split两任务并行，A1/GPU0、C1/GPU1。旧10epoch权重缺optimizer/RNG恢复状态，故不接续到60epochs。状态/日志见 `records/saliency_v5/`，计划/冻结见 `configs/SALIENCY_V5_PLAN.md` / `SALIENCY_V5_FREEZE.json`。新worker已安排完成后的自动正式探索性评测和汇总，勿另开重复评测。保持seed3407、Seen-val选模、不以U调参。以下为历史记录。

> 当前任务更新（2026-10-04）：v3 六组已全部完成；用户授权评测正式 A1/C1 Unseen 并进入下一步。七个10epoch适配方案+canonical各两折的独立探索性评测在 `records/ablation_test_v2_v3/`，checkpoint/选择规则/输入hash在读取test前已核验冻结。下一步v4权重依据v3 Seen结果预先固定并记录进评测gate：保留GMR+coarse/fine，Low_BCE_only=.01/0、Low_Pair_only=0/.02、Low_BCE_pair=.01/.02（BCE/pair有效权重）。v4已启动：A1/GPU0、C1/GPU1、每卡三任务、seed3407、10epochs、Seen-only。状态/日志见 `records/ablation_v4/`，计划/冻结见 `configs/ABLATION_V4_PLAN.md` / `ABLATION_V4_FREEZE.json`。不依据新Unseen结果更换v4系数。保持存活worker；A1/C1 U为已看过的探索集。以下为历史记录。

> 当前任务更新（2026-10-04）：第一步 v2 八组×10 epochs 已全部完成，checkpoint/hash/逐epoch exposure 已核验。S1 相对B0 Seen +.17/+ .57 pp，existence组合 −6.04/−3.07 pp（A1/C1）。用户已授权并启动第二步 v3：BCE_only、Pair_only、Weak_BCE_pair，各保留GMR+coarse/fine；BCE/pair有效权重分别1/0、0/.2、.1/.2。A1/GPU0、C1/GPU1，各三任务并行。当前状态和日志见 `records/ablation_v3/`，计划与冻结见 `configs/ABLATION_V3_PLAN.md` / `ABLATION_V3_FREEZE.json`；不要重启存活worker。只seed3407、10epochs、Seen-only，不读取formal U。完成后先读自动A1/C1汇总，再决定后续修复或泛化验证。以下保留历史记录。

> 2026-10-04 更新：用户要求跳过五 split 收尾，已授权 A1/C1 分项实验。当前任务为独立 ablation_v2 的四组×两个 split、10 epochs、seed3407、Seen-only 开发。先检查 `records/ablation_v2/worker_A1.json` / `worker_C1.json` 和新 run 状态，保持存活 worker；完成后分析 `A1_RESULTS.md` / `C1_RESULTS.md`，再按观察冻结更细消融或修复。下方 P0 与旧恢复提示不再是当前执行要求；其余条目保留为历史研究计划。

本文与 [WORK_HANDOFF](WORK_HANDOFF.md)、[诊断报告](A1_C1_FAILURE_ANALYSIS.md) 配套。用户约束：只 seed3407，不做多 seed；目标正式 Unseen，允许50 epochs。下列是待执行计划，尚未运行新消融。保持原轮换absence/层级链有效假设。

## P0：先收尾已授权的五 split 正式比较

1. 读取每个 `runs/formal/<split>/<baseline|shine>/seed3407/formal_v1_50ep/status.json`，再读两个worker状态和日志；运行中不得重启，完成run不得覆盖。
2. 原计划是A1/A2_alt/A3/C1/C2_alt各两个arms、50epochs。若进程消失或失败，定位原因，保护完成checkpoint与原freeze；不根据已看的U成绩取消某个split或修改配方。
3. 十个run均完成后执行下面入口；本轮没有自动启动它：

```bash
cd /home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3
/home/guoxiangyu/miniconda3/envs/gmr/bin/python research_ideas/10_shine_absence_aware_transfer/code/evaluate_formal.py --device cuda:0
```

4. 核验五 split结果、checkpoint和预测hash，报告canonical / baseline50 / SHINE50的Seen、U、Gap与配对区间；区分完整五split macro与已经得到的A1/C1 macro。更新root与Idea README、disposition、handoff。
5. 若冻结校验失败，先检查差异，不能通过重写旧freeze绕过。新增未列入冻结的诊断脚本不会改变原列出文件的hash。

完成标准：`records/formal/FORMAL_RESULTS.json`与`.md`包含五split×三arm；训练完整、选择规则未变；报告Seen保持和ΔU，不能以Seen下降带来的Gap缩小判断成功。

## P1：分离最可能的干扰源

研究问题：失败主要来自额外rotated existence监督，还是SHINE coarse/fine代理目标？现有局部梯度优先指向rotated BCE，但尚无因果消融。

| Arm（建议名） | 自然 GMR | coarse/fine | rotated BCE + .2 pair | 用途 |
|---|---|---|---|---|
| B0 | 是 | 否 | 否 | 原主任务对照 |
| S1_saliency_only | 是 | 是 | 否 | 原SHINE saliency机制能否保住主任务 |
| E1_rotated_exist_only | 是 | 否 | 是 | 新增existence监督是否单独造成损伤 |
| SE1_full | 是 | 是 | 是 | 完整方案，与前两项比较 |

既有baseline50/SHINE50可作历史参照；若新实验改变训练预算、选择规则或exposure，就不能直接冒充完全匹配对照。额外forward数尽量匹配或明确记录；若采用保留完整forward再关闭某loss的控制，也记录dropout/RNG差异，不能保证与旧run逐步同轨。原seed、起点、自然训练行、特征、lr、batchsize、weight decay、clip规则应一致。

实现前：新建独立 `train_ablation_v2.py` 等入口，可复用只读原模块；不修改冻结的train_formal.py、shine_losses.py、evaluate_formal.py及model代码。使用独立 `runs/ablation_v2/...`、`configs/ABLATION_V2_FREEZE.json`、`records/ablation_v2/`。任何新loss实现放新文件，记录完整hash与比较表。

建议先用原strict-inner A1_action_01 / C1_composition_01做Seen-only开发筛查，固定10epoch、seed3407；Novel-dev可以用于开发但必须记录已开发次数，不能称untouched。若目标是直接验证canonical初始化时的梯度损伤，也可在formal train/Seen-val做上述固定分项对照；全过程不读取formal test进行系数搜索。两条路线是待固定的设计选择，不要无记录混合比较。

先检查epoch0、epoch1、epoch3、最佳Seen-val、自然GMR loss和各项梯度。未来配方是否允许epoch0 fallback必须预先固定，旧50epoch结果不允许事后改成fallback。

完成标准：能回答“哪一项导致早期Seen下降”，而不仅是新增总loss是否下降；结果有配置、曲线、状态与checkpoint记录。若仍未消除Seen损伤，先报告失败，不通过U挑epoch救回结果。

## P2：仅在P1支持后检验缓和策略

候选措施为：降低rotated BCE权重、warmup、先训练saliency head再开放共享网络。它们目前是候选，不是已冻结超参，不要把建议值当历史配方。

优先依据P1选择一个方向做小范围Seen-only开发，控制开发次数。记录自然/轮换正负权重，分别记录coarse/fine与rotated BCE对共享网络的梯度，不只比较loss数值。保留模型原有判别能力后再讨论泛化。

如果S1本身破坏Seen，应优先检查saliency监督路径与权重；若S1保持Seen而E1破坏，优先处理额外BCE。若两者分别无害、联合有害，再看联合梯度与权重交互。不能在无结果时预设结论。

## P3：证明是否增强目标条件判别

在预先冻结的开发评价上报告pooled、same-video、exact-query支持与ranking、raw/gated定位、fresh pre-fusion shuffled-video。保持原始自然分布的主AUC，strata作为诊断，不替换主分数。

若需要宣称“事件证据增强”，补配套Cq/Cv与核验闭合四格；这些目前没有完成。若只优先看效果，可先报告pooled与条件指标，明确机制边界，不能以shuffle近似分数直接断言模型完全不用视觉。

A1的动作反事实、C1的object反事实是正式失败的观察；可据此提出新研究方向，但已看的A1/C1 U不再是未触碰确认集。继续在其上评价必须标明exploratory；不要在U上调阈值/epoch/权重。其他split没有据此变成天然独立确认集，若用于确认需提前固定用途并说明共享benchmark。

## P4：再做正式50epoch复制

只有开发结果支持新方案且方案/代码/数据/选择规则冻结后，才运行新正式比较。仍只seed3407，上限50epochs。完整报告Seen/U、ΔU和Gap；Unseen提高且Seen保持才支持缓解。也记录定位风险与条件结果。不自动扩展到Flash/QD或多seed。

## 可直接复制给下一轮的提示

> 请先阅读 `research_ideas/10_shine_absence_aware_transfer/records/formal/WORK_HANDOFF.md`、`NEXT_ACTIONS.md` 和 `A1_C1_FAILURE_ANALYSIS.md`，恢复Idea10任务。目标是COMMON_PROTOCOL正式Seen→Unseen退化缓解，只seed3407，允许50epochs。不要改原冻结代码，不要覆盖既有run，也不要重启存活worker。先检查并收尾原五split训练与正式评测；随后按文档落实coarse/fine-only、rotated-existence-only、完整方案的分项消融，在独立v2目录冻结。把“新增rotated BCE梯度过强”当待验证解释，继续接受用户确认的轮换absence和层级链。新参数只在Seen/inner开发上确定，禁止用正式U挑epoch、threshold或权重。保存完整结果并更新交接文档，不进行多seed。
