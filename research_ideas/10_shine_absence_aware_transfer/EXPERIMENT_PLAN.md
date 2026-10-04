# 首轮单 seed 效果筛查

用户指定：落实代码并训练、不做多 seed、优先效果。Seed 3407，A1_action_01 / C1_composition_01，两个 GPU worker 各负责一折；每折依次 baseline 与 shine，均同 strict-inner checkpoint 起点。

配置：10 epochs、AdamW lr=1e-5、原 wd、batch16、原 grad_clip=.1，固定预计算特征、共享 fusion 联训；不早停。每个 epoch 只看 Seen-val AUROC 选最好，平局最早。epoch0 与最终 selected checkpoint 才评价 Novel-dev；正式 U 不参与。基线也微调十轮，不能只与 untouched checkpoint 比较。

SHINE arm：原 GMR loss + coarse(1,2;q8) + fine(red;.25) + rotated existence BCE + .2*exist pair(.2)。每 batch 最多四条完整三层链；自然所有行保留。部分 chain 过滤以保护留出语义，覆盖存 QUERY_BANK_AUDIT。不把所有编辑 query 都标 absent，只给用户确认的 rotated query 加负 BCE。

输出自然全集 Seen/Novel AUROC、source PairAcc、exact same-query ranking/support、raw top1 R@IoU .5/.7、Seen-val 阈值的 gated R@IoU .5、loss breakdown、chain exposure、最佳 epoch、时间和显存。该 raw R@IoU 为直接 top1、不含官方 NMS/postprocessing 的筛查指标，不能冒充 official mAP。

单 seed 两折结果只用于效果筛查；不声称跨 backbone/全部语义稳定泛化。当前 baseline 和 shine 的额外 forward/数据预算不同，如有收益，后续可以在同 seed 做 rotated-only/coarse-fine-only 消融区分作用，不因尚未做全部消融阻止首次训练。

每 run `resolved_config.json/history.json/status.json/best.ckpt/predictions_*.jsonl/metrics.json`；records 保存固定来源、数据/文本 hash、imports、验证与每折 summary。完成后更新 README 与实际结果；运行中不预填收益。
