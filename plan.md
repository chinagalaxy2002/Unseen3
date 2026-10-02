你现在负责一次“先审计、再复制、再迁移、最后可训练验证”的代码工程任务。

不要从零写一个新项目，不要凭印象实现 TRM，也不要只看 README。
必须先完整阅读现有基础项目，再阅读真实 TRM 官方代码和论文，然后在基础项目副本上完成 Moment-DETR + TRM / TRM-PT 的迁移。

============================================================
0. 本任务的本地路径和外部参考
============================================================

基础项目，只读源目录：

/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen

目标工作目录：

/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3

TRM 官方真实代码仓库：

https://github.com/minghangz/TRM

TRM 论文：

Phrase-level Temporal Relationship Mining for Temporal Sentence Localization
AAAI 2023
论文入口：
https://ojs.aaai.org/index.php/AAAI/article/view/25478

同一官方仓库 README 还列出后续工作：

Large-Scale Pre-trained Models Empowering Phrase Generalization in Temporal Sentence Localization
TRM-PT / IJCV

重要事实：
当前 https://github.com/minghangz/TRM 的 main 分支主要公开的是 TRM 源码。
不要虚构一个仓库中不存在的独立 TRM-PT 实现。
凡是 TRM-PT 的模块，必须区分：
1. 官方仓库真实已有实现；
2. 论文明确描述、但仓库没有实现的部分；
3. 我们为了适配 Moment-DETR 必须做的工程改写。

在最终文档中必须明确标注这三类来源。

============================================================
1. 总体研究目标
============================================================

我要研究的是 unseen semantic generalization 下的 Video Moment Retrieval。

当前这一轮只做第一阶段：

“U+ localization 泛化到底能否通过 TRM/TRM-PT 式 phrase/composition reasoning 得到改善”。

这一阶段：

- 只研究 localization；
- 不研究 U−；
- 不优化拒绝率；
- 不使用 existence threshold；
- 不使用 hard gate；
- 不因为 query unseen 就拒绝；
- 不使用 U+ test 做模型选择；
- 不允许 U semantics 参与训练。

目标模型是：

Moment-DETR + TRM-style phrase-level temporal relationship mining

基础 Moment-DETR 必须来自本地基础项目：

/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen

禁止换成 TRM 自己的 2D-TAN/MMN 主干。

核心思想是：

保留 Moment-DETR 的 video encoder / text projection /
transformer / DETR decoder / span regression /
Hungarian matching。

把 TRM 的：

- phrase decomposition
- phrase encoding
- phrase importance weighting
- phrase-to-temporal-proposal matching
- phrase/sentence consistency
- phrase exclusiveness
- cross-sample negative phrase learning
- phrase evidence 对 sentence localization 的 refinement

迁移到 Moment-DETR decoder proposals 上。

Moment-DETR 的 decoder query slots 就视为 TRM 中的 temporal proposals。

不要把 TRM 的完整 2D proposal map 网络硬塞进 Moment-DETR。

============================================================
2. 第一阶段：完整审计基础项目，禁止立刻修改
============================================================

首先：

cd /home/guoxiangyu/VLMbasedIter_momentretrival/Unseen

完整检查 git 状态：

git status
git log -5 --oneline

记录当前 commit、dirty files、重要文件 SHA256。

然后系统阅读整个 Moment-DETR 训练链。

至少必须找到并阅读：

1. Moment-DETR model 定义
2. transformer
3. matcher
4. criterion/loss
5. dataset
6. collate
7. prepare_batch_inputs
8. train loop
9. optimizer
10. evaluation
11. inference/post-processing
12. config
13. feature loading
14. CLIP text feature生成方式
15. positive-only localization 实验脚本
16. semantic split A1/A2_alt/A3/C1/C2_alt

不要假设这些文件路径和 GMR_Unseen GitHub 项目完全一样，
必须以本机 Unseen 实际内容为准。

重点查清：

- src_txt shape
- src_txt_mask shape
- src_vid shape
- hidden_dim
- num_queries
- decoder hs shape
- pred_logits 的 foreground/background index
- pred_spans 格式
- matcher 如何处理正例
- 空 GT 如何处理
- CLIP text NPZ 保存什么
- query token feature 是不是 last_hidden_state
- text normalization 如何做
- video feature 来源和维度
- train/val/test split 路径
- localization-only 训练是否已经存在
- checkpoint 选择规则
- 当前 baseline 的正式训练 seed、epoch、batch size

把审计写到：

/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/
    experiments/trm_momentdetr_generalization/
        BASELINE_AUDIT.md

但是此时先不要创建 Unseen3；
先完成源目录检查，把需要记录的信息暂存。

============================================================
3. 完整审计官方 TRM 源码
============================================================

把官方仓库 clone 到临时只读参考目录，例如：

/home/guoxiangyu/VLMbasedIter_momentretrival/_external/TRM

如果已存在，则 git fetch 并记录 commit，不要随意覆盖本地修改。

必须阅读真实文件，而不是只看 README。

重点至少包括：

trm/modeling/trm/trm.py
trm/modeling/trm/loss.py
trm/modeling/trm/text_encoder.py
trm/modeling/trm/featpool.py
trm/modeling/trm/feat2d.py
trm/modeling/trm/proposal_conv.py

trm/data/datasets/charades.py
trm/data/collate_batch.py

trm/engine/trainer.py
trm/engine/inference.py

configs/charades.yaml
trm/config/defaults.py
scripts/charades_train.sh
README.md

必须逐项确认以下真实实现：

A. sentence encoder 和 phrase encoder 如何共享参数

B. AttentivePooling 实际代码是否真的用了 global sentence feature
不要只相信论文图。
如果源码参数传入但没有实际参与计算，要在迁移记录中写明。

C. phrase 最大数量

D. phrase dropout

E. sentence score 如何和 weighted phrase score 融合

F. phrase score 如何与 temporal proposals 计算

G. cosine normalization 和 sigmoid temperature

H. positive phrase consistency loss

I. negative video / negative phrase contrastive construction

J. exclusiveness loss

K. Charades 实际 config 中：
CONSIS_WEIGHT
EXC_WEIGHT
THRESH
CONTRASTIVE
DROP_PHRASE
RESIDUAL

尤其检查一个重要事实：

官方 Charades 配置如果确实是：

CONSIS_WEIGHT: 1.0
EXC_WEIGHT: 0.0

则必须记录：
论文描述包含 exclusiveness，
但官方 Charades 运行配置关闭 exclusiveness。

本项目要求“全量迁移”，因此这里不要做消融，
但必须提前决定：
是严格 source-faithful 配置，
还是启用论文完整 loss。

本任务采用：

“完整机制迁移”。

因此：
- consistency 开启
- negative phrase/video learning 开启
- exclusiveness 也实现并开启

但是：
不能偷偷把超参数调到 test 最优。

对论文没有给出、官方 Charades 配置又关闭的 loss weight，
选择一个事先固定的合理默认值并记录。
优先：
lambda_consistency = 1.0
lambda_exclusiveness = 1.0

如果源码或论文给出了更明确的 full-method 默认配置，
以论文/源码为准，并在文档中说明。

============================================================
4. 阅读论文，不允许只根据代码猜机制
============================================================

阅读 AAAI 2023 TRM 正文。

至少整理：

- 问题定义
- phrase generation
- sentence localization
- phrase localization
- MIL assumption
- consistency constraint
- exclusiveness constraint
- negative sentence/video construction
- final inference fusion
- loss equation
- Charades 设置
- unseen composition 实验设置

写入：

experiments/trm_momentdetr_generalization/
    TRM_SOURCE_AUDIT.md

文档中每个机制标注：

[TRM-PAPER]
[TRM-REPO]
[MOMENT-DETR-ADAPTATION]

不能把 adaptation 写成原论文方法。

============================================================
5. TRM-PT 的处理原则
============================================================

还要阅读：

Large-Scale Pre-trained Models Empowering Phrase Generalization in Temporal Sentence Localization

检查论文具体说明的：

- pretrained VLM
- phrase pseudo-label generation
- pseudo temporal supervision
- iterative refinement
- verb/state-change semantic augmentation

但是：

如果官方 minghangz/TRM 仓库中没有对应代码，
不得声称“照官方源码迁移”。

对于没有代码但论文定义足够明确的部分：

可以实现，
但必须放在独立模块中，
并在 METHOD_SPEC.md 写：

“paper-derived reproduction, not copied from released TRM source”.

如果论文无法明确确定某个实现细节，
不要臆造复杂算法。
采用最小、可解释、可关闭的实现，并记录假设。

非常重要：

TRM-PT 使用任何外部 pretrained model 时：

只能处理 S+ training data。

禁止使用正式 U+ test query 来：

- 生成训练 pseudo label
- 选择 prompt
- 调整 threshold
- 选择 checkpoint
- 调超参数

否则破坏 S→U 泛化协议。

============================================================
6. 在复制前冻结源项目
============================================================

完成以上两边源码审计后，记录：

- Unseen source absolute path
- source git commit
- source git status
- TRM git commit
- 关键代码 SHA256

然后复制整个基础项目：

SOURCE=/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen
TARGET=/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3

如果 TARGET 已存在：

不要直接 rm -rf。

先停止并报告，检查里面是否已有内容。
只有确认是本任务可覆盖目录时才能继续。

建议：

rsync -a \
  --exclude='.git' \
  "$SOURCE/" "$TARGET/"

然后：

cd "$TARGET"
git init
git add .
git commit -m "Baseline copy before Moment-DETR TRM migration"

绝对禁止回写 SOURCE。

从这一刻开始，所有修改只发生在：

/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3

============================================================
7. 新实现必须与 baseline 隔离
============================================================

不要直接覆盖现有 Moment-DETR baseline。

优先新建独立代码路径，例如：

models/moment_detr_trm/
training/moment_detr_trm/
configs/moment_detr_trm/

并尽量 import 原 baseline 中不需要修改的：

- transformer
- matcher
- position encoding
- span utils
- misc utilities
- postprocessing

最终必须保证：

旧的 Moment-DETR localization baseline
仍然可以按旧命令运行。

============================================================
8. Moment-DETR + TRM 的结构定义
============================================================

实现下面的模型。

--------------------------------
8.1 原 Moment-DETR 主链不变
--------------------------------

full query 和 video 仍走原模型：

video features
+
full-query CLIP token features
→ input projection
→ transformer
→ decoder slots h_j
→ class head
→ span head

保留：

pred_spans
pred_logits
Hungarian matching
span L1/GIoU
foreground/background classification

--------------------------------
8.2 phrase branch
--------------------------------

每个 query 提取最多 P 个 semantic phrases。

phrase 可以包括：

- predicate / verb phrase
- subject / agent
- object / patient
- prepositional or complement phrase
- 有明确语义贡献的其他 SRL constituent

不要简单按空格切词。

优先根据 TRM 的原 phrase 数据生成逻辑/数据定义恢复 phrase 规则。

如果原始 TRM annotations 已经带 phrase，
分析它们如何产生。

若需要外部 SRL：

把 phrase extraction 写成离线 preprocessing，
冻结结果。

不要训练过程中动态调用网络 API。

保存：

qid
original query
phrases
phrase roles
parser version
parser/source
parse success
fallback status

到类似：

features/.../trm_phrase_metadata.jsonl

--------------------------------
8.3 phrase text feature
--------------------------------

为了保持本项目输入预算一致，

优先使用和 full query 完全相同的 CLIP text encoder。

不要把 full query 用 CLIP、phrase 用 DistilBERT。

如果基础代码已有 CLIP ViT-B/32 权重和
last_hidden_state 提取代码，
直接复用。

phrase features 缓存在新目录，
绝不覆盖原 query features。

例如：

features/.../clip_phrase/

每个 qid 保存：

phrase_features
phrase_masks
phrase_count

phrase tokens 的 preprocessing、
normalization、
truncation
必须和原 query pipeline 一致。

--------------------------------
8.4 共享 text projection
--------------------------------

full-query token 和 phrase token
共享 Moment-DETR 的 input_txt_proj。

不要为 phrase branch 建一个独立大 encoder。

phrase token projection 后，
masked mean 得到：

phrase_repr
shape:
[B, P, hidden_dim]

--------------------------------
8.5 decoder slots = temporal proposals
--------------------------------

Moment-DETR：

hs[-1]
shape 应为：

[B, Q, hidden_dim]

把每个 decoder slot h_j 当作 temporal proposal representation。

新增轻量 projection：

phrase_proj
slot_proj

然后：

p_i = normalize(phrase_proj(phrase_repr_i))
h_j = normalize(slot_proj(h_j))

score_ij = p_i dot h_j

保持 TRM 源码的 score scale，
如果源码是：

sigmoid(10 * cosine)

则采用相同实现。

输出：

pred_phrase_scores
shape:
[B, P, Q]

--------------------------------
8.6 phrase importance
--------------------------------

参考 TRM 官方 AttentivePooling。

如果官方源码实际是：

Linear
→ tanh
→ Linear
→ masked softmax

则忠实实现。

不要为了“看起来更合理”
擅自变成另一个 Transformer attention。

得到：

alpha_i
shape:
[B,P]

计算：

phrase_support_j =
sum_i alpha_i * score_ij

--------------------------------
8.7 phrase → sentence proposal refinement
--------------------------------

参考 TRM：

sentence score
+
weighted phrase score

Moment-DETR 对应为：

base foreground logit_j
+
lambda_refine * phrase_support_j

background logit 保持原始逻辑。

即生成：

pred_logits_refined

最终训练 matcher / classification
和 inference candidate ranking
使用 refined logits。

同时保留：

pred_logits_base

方便之后审计，但这一轮不做 ablation。

lambda_refine 使用论文/源码默认值。
若只能推导，事先固定并记录，
不要根据 U test 调参。

============================================================
9. Phrase temporal relationship losses
============================================================

完整迁移，不做组件消融。

--------------------------------
9.1 sentence GT 与 decoder proposal IoU
--------------------------------

对每个 S+ training sample：

使用预测 pred_spans
与 GT relevant window
计算每个 decoder slot 的 IoU。

定义：

positive candidate set A_pos

优先采用 TRM 的阈值。

如果官方 Charades：

THRESH = 0.1

则初始采用 0.1。

但是 DETR 训练早期可能没有任何 slot 超过阈值，
因此增加 adaptation：

A_pos =
{slot | IoU >= threshold}
UNION
{Hungarian matched positive slot}

这样至少有一个正 proposal。

这条必须在 METHOD_SPEC.md 标注：

[MOMENT-DETR-ADAPTATION]

--------------------------------
9.2 positive phrase consistency / MIL
--------------------------------

对每个有效 phrase：

在 A_pos 中取：

max_j phrase_score[i,j]

要求至少一个正 proposal
对该 phrase 产生高支持。

按 TRM 源码的 BCE / focal-style 实现恢复。

不要自行换成 unrelated contrastive objective。

--------------------------------
9.3 negative phrase / negative video
--------------------------------

迁移 TRM 源码中的：

current phrase + negative video
以及
current video + negative phrases

策略。

优先使用 batch permutation，
不要 while random 死循环。

例如 deterministic cyclic permutation：

neg_idx = (idx + offset) % batch_size

但如果 TRM 原代码严格随机选其他 batch sample，
可以保持其逻辑，同时保证 seed 可复现。

必须处理 batch_size == 1。

--------------------------------
9.4 exclusiveness
--------------------------------

实现论文定义的 exclusiveness：

对 GT 外 / 低 IoU proposal，

不应该所有 phrase 都高匹配。

按论文/源码数学定义实现。

因为本任务要求完整迁移，
此 loss 开启。

设置：

lambda_exclusiveness

必须在正式运行前冻结。

--------------------------------
9.5 phrase dropout
--------------------------------

迁移 TRM 的 phrase dropout。

如果 Charades 源配置 DROP_PHRASE=True，
使用源码同等 keep probability。

不要根据 test 表现修改。

============================================================
10. 只对正例计算 phrase temporal losses
============================================================

第一阶段训练数据是 S+ positive-only。

因此正常情况下所有 train row 都有 GT。

但代码仍必须健壮：

如果 relevant_windows 为空：

- 不计算 span loss
- 不计算 phrase consistency
- 不计算 phrase exclusiveness
- 不计算 phrase temporal refinement supervision

不要制造虚假的 temporal window。

============================================================
11. 第一阶段的数据协议
============================================================

这一步非常重要。

当前实验只测试 localization generalization。

对于每个正式 split：

A1
A2_alt
A3
C1
C2_alt

训练：

只使用该 split 的 S+

验证：

只使用 seen positive validation

测试：

分别统计：

S+
U+

禁止训练：

S−
U−

禁止模型选择：

U+
U−

禁止：

existence head
existence BCE
existence gate
existence threshold
RR
FRR

如果原项目已有 positive-only localization 脚本，
优先复用其 exact split construction，
不要重新发明数据划分。

============================================================
12. 不改变底层特征预算
============================================================

第一轮完整训练要求和 Moment-DETR baseline
尽可能同信息预算。

保持原来的：

CLIP video
SlowFast video
CLIP text
clip_length
max_v_l
max_q_l
hidden_dim
num_queries
encoder layers
decoder layers

除非 TRM branch 必须新增参数。

不要同时：

- 换 VideoMAE
- 换更强 CLIP
- 换 LLM encoder
- 改视频 fps
- 改 Moment-DETR backbone
- 加额外 video pretraining

否则无法判断收益来自 TRM 还是 backbone。

TRM-PT 如果确实要求额外 pretrained model，
必须单独记录 external-information budget。

============================================================
13. 训练策略
============================================================

正式实验采用“全量迁移训练”。

不做 component ablation。

但是仍要保留一个 baseline reference：

原 Moment-DETR positive-only localization 结果。

不是重新做 ablation，
而是用于判断新模型有没有提升。

训练必须从头运行，
不要偷偷从 U-aware checkpoint 初始化。

使用和现有正式 Moment-DETR localization control
一致的：

- seed
- epochs
- optimizer
- batch size
- validation schedule
- video/text features

如果项目正式 multisplit 是固定 100 epoch，
则保持 100 epoch。

如果 baseline 使用 forced 100 epoch，
TRM 也 forced 100 epoch。

不要因为 TRM 收敛较慢
临时多训练几百 epoch。

============================================================
14. 模型选择
============================================================

只允许使用 seen validation。

绝不能看 U+ 选择 epoch。

对于 localization-only，
使用现有 baseline 的 checkpoint-selection metric。

例如：

seen validation MR-full-mAP
或者
seen R@1@0.5

必须先查基础项目实际上用的是什么，
然后保持一致。

不能给 TRM 换一个更有利的 selection metric。

============================================================
15. 正式输出指标
============================================================

每个 split：

A1
A2_alt
A3
C1
C2_alt

至少输出：

Seen positive:
- R@1 IoU 0.3
- R@1 IoU 0.5
- R@1 IoU 0.7
- mIoU

Unseen positive:
- R@1 IoU 0.3
- R@1 IoU 0.5
- R@1 IoU 0.7
- mIoU

并计算：

seen_to_unseen_gap_R1_05
seen_to_unseen_gap_mIoU

同时与原 Moment-DETR positive-only baseline 比较：

delta_seen_R1_05
delta_unseen_R1_05
delta_seen_mIoU
delta_unseen_mIoU

当前阶段不要输出：

AUROC
FRR
RR
gated R1

因为这一阶段没有做 existence。

============================================================
16. 后续 GMR 接口必须预留，但现在不要启用
============================================================

模型代码最好保持未来可重新接入：

pred_exist_logits

但这一轮：

use_exist_head = false

不要让 phrase score 直接当 existence score。

尤其禁止：

high uncertainty => absent
low phrase score => unseen/absent

“unseen”不等于“不存在”。

============================================================
17. 需要新增的文档
============================================================

在：

Unseen3/experiments/trm_momentdetr_generalization/

创建：

README.md
BASELINE_AUDIT.md
TRM_SOURCE_AUDIT.md
METHOD_SPEC.md
EXPERIMENT_PLAN.md
IMPLEMENTATION_LOG.md
SOURCE_MANIFEST.json

其中 METHOD_SPEC.md 必须逐条写：

TRM 原机制
→
Moment-DETR 对应实现

例如：

TRM 2D proposal
→
Moment-DETR decoder slot

TRM phrase-proposal cosine score
→
phrase-slot cosine score

TRM weighted phrase map
→
weighted phrase support per decoder slot

TRM sentence + phrase score
→
base foreground logit + phrase support

TRM consistency
→
phrase support on high-IoU/Hungarian matched slots

TRM exclusiveness
→
phrase support suppression on low-IoU slots

============================================================
18. 代码质量要求
============================================================

必须：

- type/shape 注释
- mask 正确
- batch_size=1 可运行
- phrase_count=0 有 fallback
- phrase_count=1 可运行
- 所有 padding phrase 不参与 softmax/loss
- 不产生 NaN
- GT 为空不崩
- CPU import 能通过
- CUDA forward 能通过
- checkpoint 可保存/恢复
- evaluate 能独立加载 checkpoint

不要：

- 到处复制旧代码造成两个版本漂移
- 修改 baseline 正式结果文件
- 覆盖旧 checkpoint
- 覆盖旧 prediction
- 使用同一个 results_dir

============================================================
19. 先做静态和最小运行验证
============================================================

在正式训练前必须执行：

A. Python import test

B. dataset smoke test

取至少：
2-4 个样本

检查：

query
phrases
phrase masks
query features
phrase features
video features
GT windows

C. model forward test

打印并 assert：

pred_spans
pred_logits_base
pred_logits
pred_phrase_scores
phrase_weights

D. loss test

确保：

loss_span finite
loss_giou finite
loss_label finite
loss_phrase_consistency finite
loss_phrase_negative finite
loss_phrase_exclusiveness finite

E. backward test

确认：

phrase_proj 有梯度
slot_proj 有梯度
phrase attention 有梯度
Moment-DETR 原 transformer 有梯度

F. 1 mini-batch optimizer step

G. 10~20 batch tiny overfit test

检查 loss 是否可下降。

这不属于消融，
只是工程正确性验证。

============================================================
20. 运行前冻结配置
============================================================

正式训练前生成：

EXPERIMENT_FREEZE.json

里面记录：

- source Unseen git commit
- source Unseen dirty-state
- copied baseline commit
- TRM git commit
- paper URLs
- code SHA256
- train annotation SHA256
- val annotation SHA256
- text feature source
- video feature source
- phrase metadata SHA256
- phrase feature source
- random seed
- all hyperparameters
- lambda_refine
- lambda_consistency
- lambda_negative
- lambda_exclusiveness
- phrase dropout
- IoU threshold
- max phrases
- total train rows
- total seen val rows
- forbidden U access declaration

冻结之后不能因为正式 U+ 结果不好修改参数。

============================================================
21. 正式训练顺序
============================================================

完成代码和 smoke test 后，

先只运行一个 split 的完整训练验证工程链，
建议 A1。

但注意：

不能用 A1 U+ 表现反过来改方法。

A1 完成后：

检查的是：

- 程序是否稳定
- checkpoint 是否正常
- evaluation 是否正常
- 没有数据泄漏
- metrics 文件是否完整

确认工程正确后，
使用完全相同被冻结的机制和超参数，
跑：

A1
A2_alt
A3
C1
C2_alt

每个 split 都独立训练。

不能共享 fine-tuned checkpoint。

============================================================
22. 如果加入 TRM-PT
============================================================

在 TRM core 全量迁移完成并可训练后，
再实现 TRM-PT。

不要混在第一次代码修改里导致无法定位问题。

但是本任务最终目标是包含 TRM/TRM-PT，
所以完成 TRM core 后继续完成 PT branch。

TRM-PT 必须：

1. 根据论文确定 pretrained model；
2. 仅处理 S+ train；
3. 产生 phrase-level pseudo temporal labels；
4. 缓存到独立目录；
5. 保留 confidence；
6. 实现论文说明的 iterative refinement；
7. 如果使用 verb before/after state，
   保存生成文本和生成模型信息；
8. 绝不使用 U+ test。

Moment-DETR-TRM-PT 仍保留原 full-query GT localization supervision，
phrase pseudo labels 只能作为辅助监督。

如果论文某个细节无法从正文/附录确定，
在 IMPLEMENTATION_LOG.md 写：

UNSPECIFIED_BY_PAPER

然后使用最小假设实现。

不要伪装成作者原代码。

============================================================
23. 禁止事项
============================================================

以下行为禁止：

1. 不读基础项目就开始写。
2. 不读 TRM 源码只根据论文摘要写。
3. 整个替换成 TRM 的 2D-TAN backbone。
4. 修改原 Unseen 源目录。
5. 用 U+ test 调参。
6. 用 U query 生成训练 pseudo label。
7. 把 phrase confidence 当 existence。
8. 把 unseen 当 absent。
9. 为了结果好临时打开/关闭 loss。
10. 看到 U+ 结果后改 phrase parser。
11. 多个 split 用同一个 fine-tuned checkpoint。
12. 覆盖已有实验结果。
13. 宣称仓库有实际上不存在的 TRM-PT 官方实现。
14. 隐藏 source 与 paper implementation 的差异。

============================================================
24. 执行方式
============================================================

现在开始工作。

第一个阶段不要直接改代码。

先完成：

1. 基础项目审计
2. TRM repo 审计
3. TRM/TRM-PT 论文审计
4. 给出准确的迁移映射表
5. 报告你计划修改/新增的文件列表

确认这些工作后，
再复制到 Unseen3 并实际修改。

整个过程中不要问我“是否继续”
这类无必要问题。

只要没有遇到：

- 目标目录已有不可覆盖数据
- 核心特征不存在
- 论文不可获取
- TRM repo 无法获取
- 基础项目结构与上述假设根本冲突

就继续执行到：

“代码完成 + smoke test 通过 + 可正式训练”

为止。

最后向我报告：

1. 实际读取了哪些基础文件
2. 实际读取了哪些 TRM 文件
3. TRM paper 与 repo 的差异
4. TRM-PT paper 与 repo 的差异
5. 新增/修改文件清单
6. Moment-DETR → TRM 的准确结构映射
7. loss 总公式
8. 数据流
9. phrase preprocessing
10. smoke-test 结果
11. 参数量变化
12. 正式训练命令
13. 正式评测命令
14. 输出目录
15. 已知风险/未决实现细节

不要只给方案。
最终需要实际完成 Unseen3 中的代码迁移。