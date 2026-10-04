# Idea 10：正式 Unseen 退化缓解结果（50 epochs）

Moment-DETR-GMR，seed 3407，本次仅评测 A1、C1；macro 是这两个 splits 的等权平均，不代表五 split。新训练两 arms 各 50 epochs，从同 canonical checkpoint 初始化；只按 Seen-val AUROC 选模，正式 U 仅在训练完成且 checkpoint 冻结后评测。

| Split | Arm | Seen AUROC | Unseen AUROC | Gap |
|---|---|---:|---:|---:|
| A1 | canonical | 0.8034 | 0.4956 | 0.3078 |
| A1 | baseline | 0.8088 | 0.4858 | 0.3231 |
| A1 | shine | 0.7517 | 0.4799 | 0.2718 |
| C1 | canonical | 0.7611 | 0.5683 | 0.1928 |
| C1 | baseline | 0.7606 | 0.5747 | 0.1858 |
| C1 | shine | 0.7365 | 0.5672 | 0.1694 |
| **Macro** | **canonical** | **0.7823** | **0.5320** | **0.2503** |
| **Macro** | **baseline** | **0.7847** | **0.5302** | **0.2544** |
| **Macro** | **shine** | **0.7441** | **0.5235** | **0.2206** |

| Comparator | A1/C1 Macro ΔUnseen (pp) | A1/C1 Macro ΔSeen (pp) | Gap reduction (pp) |
|---|---:|---:|---:|
| SHINE−canonical | -0.84 | -3.81 | +2.97 |
| SHINE−baseline | -0.67 | -4.06 | +3.39 |

COMMON_PROTOCOL 的 Moment 历史参考为 Seen .7518 / Unseen .5287 / Gap .2231；canonical 行为同 checkpoint 的完整精度重算。历史分数精度/选模口径不同处不能当作新方法收益。

Gap 缩小若伴随 Seen 降低，不能单独算退化缓解；主判断是 ΔUnseen 与 Seen 保持。不同 split 共享视频，未给出假设 split 独立的 macro CI。逐 split 共享视频配对区间、条件排序、raw/gated top1 定位和 fresh pre-fusion shuffle 见 [A1_C1_FORMAL_RESULTS.json](A1_C1_FORMAL_RESULTS.json)。

这里只迁移 Moment，不能声称 Flash/QD 的 .5479/.5144 退化已缓解；Cq/Cv 与闭合四格对照仍缺，不能将 pooled 收益直接归因于真正事件证据。没有运行多 seed。

## 当前判断

A1、C1 正式 U 未观察到退化缓解：相对同预算 baseline，Unseen 分别 −0.59 / −0.76 pp，两者配对 95% CI 均含 0；Seen 分别 −5.71 / −2.40 pp，两者区间均低于 0。两 split macro Unseen −0.67 pp、Seen −4.06 pp。Gap 缩小源于 Seen 损失，不能作为成功证据。Inner Novel-dev 的正向结果未在这两项正式 U 复现。

本次只完成 A1/C1 评测，其他 split 的既定训练继续；没有依据 U 更换 checkpoint、参数或选择规则。执行源码存于 `A1_C1_EVALUATOR_EXECUTED.py`，评测前校验与 checkpoint hashes 见 `A1_C1_EVALUATION_CHECKPOINT_FREEZE.json`。
