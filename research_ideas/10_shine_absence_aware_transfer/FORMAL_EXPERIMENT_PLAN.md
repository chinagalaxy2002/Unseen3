# 正式 Seen→Unseen 退化比较：50 epochs

用户明确要比较 COMMON_PROTOCOL 的正式 Unseen 退化，允许 50 epochs；不进行多 seed。之前两折 Novel-dev 结果仅是 pilot，不用于代替下面的结果。

固定 seed 3407；Moment-DETR-GMR 五个原始 split：A1、A2_alt、A3、C1、C2_alt。每个 split 只用其正式 S+/S- train 和 Seen-val，按该 split 的语义保留定义过滤辅助编辑。初始化对应 canonical moment/best.ckpt，不加载 inner 模型。

每 split 保留三个比较对象：原 canonical checkpoint；从该 checkpoint 继续训练 50 epochs 的 baseline；从同 checkpoint 联合 SHINE 训练 50 epochs 的方法。后两者完全同自然数据、起点、优化器、lr=1e-5、batch16、50 epochs、原 grad_clip，无早停，Seen-val existence AUROC 选 epoch，平局最早。多 seed 建议由用户明确覆盖；inner四折先行建议由用户指定正式目标覆盖。

SHINE loss/margins/三层 red 链沿用 pilot 已固定版本，不根据正式 U 改动。预计算编码器冻结，fusion/head 联训；每 batch 最多四条完整链。采用用户确认的 batch 轮换 absence 和强制编辑距离链。

`code/train_formal.py` 不加载 test 标注/特征，不评测 U。完成所有十个训练 run，并冻结 checkpoint 身份后，再由独立 evaluator 加载正式 test；U 不选 epoch、不选超参数、不调阈值。阈值仅从 Seen-val 选择。旧 formal 指标已被研究人员看到，因此这是 exploratory replication。

主表每 split/五 split 等权 macro 报 S+/S- AUROC、U+/U- AUROC、Gap、ΔU、ΔSeen、Gap 缩小量；canonical 重算完整精度结果与协议历史四位均值分列。同时报告继续训练 baseline→SHINE 的增量，不能将额外训练预算归因于 SHINE。

关键判据：真实 Unseen AUROC 上升，并保持 Seen；不能通过 Seen 下降缩小 Gap 作为成功。报告正负数、独立视频/query、full-precision score、matched same-video / exact same-query support、raw/gated localization、视频置换依赖性和配对视频 bootstrap。Cq/Cv 与四格若没有当前五 split 的等协议训练/可靠闭合支持，则如实写缺失，不用 inner 结果代替正式机制证据。

所有代码、标签、bank、新特征、配置、权重、日志和结果独立落在本 idea 的 formal 子目录；不覆盖 pilot、共享模型、release 或历史结果。配置、源码/数据/特征及输入 checkpoint hashes 保存在 `configs/FORMAL_FREEZE.json` 与 `records/formal/`。当前训练尚未完成，不预填正式收益。
