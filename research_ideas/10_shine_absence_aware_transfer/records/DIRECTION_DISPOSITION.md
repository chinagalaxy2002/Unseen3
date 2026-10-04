# Idea 10 disposition

状态：`exploratory_positive_pooled`。完成两个 folds × 两个 arms × 一个 seed（3407），每 run 10 epochs，共 40 training epochs。

SHINE 相对同起点微调 baseline 的 Novel-dev ΔAUROC：A1 +4.18 pp，C1 +3.58 pp，macro +3.88 pp；Seen macro +1.32 pp。固定模型的 paired video-cluster CI 两折下界均为正。

采用用户确认的 batch-rotation absence 与层级距离链；pilot 未进行多 seed、正式 U 或跨 backbone 训练。Novel 定位未提高，conditional 排序并非一致改善。可保留为 pooled 改善候选，不能声称根因或 saliency-only 贡献已确立。详见 [PILOT_RESULTS](PILOT_RESULTS.md)。

以上是此前 pilot 完成时状态；后续五 split 正式 50-epoch 队列已启动。后续如推进，可先在同 seed 做 temporal-only / rotated-existence-only 消融；这只是建议，本轮未自动启动。

## 正式 U 更新：A1/C1

状态：`formal_subset_no_observed_improvement`。两 split 的 baseline50 / SHINE50 均完成，checkpoint 仅按 Seen-val 选模。SHINE 相对 baseline 的 Unseen macro −0.67 pp、Seen macro −4.06 pp；Gap 缩小不能解释为退化缓解。两个 U 配对区间均含 0，两个 Seen 配对区间均低于 0。当前方向在正式 U 的这两个 split 未复现 inner pooled 增益。其他 split 继续原冻结训练，本次未调参或更换选模规则。详见 [正式结果](formal/A1_C1_FORMAL_RESULTS.md)。
