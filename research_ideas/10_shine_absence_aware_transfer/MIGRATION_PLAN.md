# 已落地迁移与上游差异

上游 commit：`dfbaab1cf8f6f88ca3c3a39bee2bbe8dd999f466`。保留 MIT 来源与完整 vendor。改动局限本目录，不修改现有 baseline/其他 idea。

| 源码/文件 | 迁移内容 |
|---|---|
| SHINE `shine/ctf_ranking.py` | coarse topK、fine div_loss 与 red 链→`code/shine_losses.py`；保留数值公式，新增 padding/null 有效域 |
| SHINE `data/charades*/*gpt_train.jsonl` | 按 inner-train 视频+原句匹配三层编辑→`code/prepare_bank.py`；没有重新调用 LLM |
| 当前 `models/moment_detr_gmr/` | 完整物理副本，保留 original model/head；不搬 SHINE global token/reference decoder |
| 当前 `training/moment_detr_gmr/dataset.py` | 保留 mr_only=true、自然空窗口支持；单独计算 dense GT mask，绕开 mixed batch 字段交集导致的 saliency 丢失 |
| `code/train_pilot.py` | 共享 fusion 联训，轮换重 forward、三层重 forward、existence 连接、独立选模/预测/指标 |
| `code/clip/` | 复制现有文本提取实现；text_only 分支避开不使用的 torchvision 图像预处理，不改 CLIP 文本计算 |
| `code/check_transfer.py` | 上游公式 parity、padding 零梯度、空项、真实 mixed-batch backward 与 fusion/existence 梯度 |

上游 Charades loader 直接访问 `relevant_windows[0]`，不能直接搬来承接 GMR null 行。上游 query features 命名 `<qid>.npz` 与本项目 `qid<qid>.npz` 不同；新编辑由 hash manifest 对应。Fine 内部固定三层，不能只改 hn_num 就以为支持任意层数。

上游 div_loss 是 `mean(-stopgrad(sigmoid(target))*log(sigmoid(logits)+eps))`；GT 模式直接用二值 target。不是 KL，也不是包含负项的完整 BCE。五个距离 d1..d5 的 red loss 是相邻四个 `[.25+di−di+1]_+`。本实现对有效 token 长度归一化，合成无 padding 输入与上游对齐。

第一轮是**机制迁移**，不是 SHINE 原生架构复现；保持当前 CLIP+SlowFast 而非上游默认 I3D。后续原生 SHINE+GMR/Flash/QD 属于独立扩展，不混入当前结果。

用户确认 batch absence 与层级有效性已记录 SOURCE_MANIFEST；不重复要求核验。所有 queries 从 fusion 前重新输入，正式 U 不访问。当前名义时间网格与保存配置一致，结果注明没有额外 extraction alignment 证据。
