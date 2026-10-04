# 来源与贡献边界

最近邻为 Cheng 等 **SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding**（ECCV 2024）：[论文](https://arxiv.org/html/2407.05118)、[用户指定代码库](https://github.com/zxccade/SHINE)。分层语义编辑与 coarse-to-fine saliency 思想全部继承。

具体源码固定到 [ctf_ranking.py](https://github.com/zxccade/SHINE/blob/dfbaab1cf8f6f88ca3c3a39bee2bbe8dd999f466/shine/ctf_ranking.py)，接口参见同 commit 的 model.py/start_end_dataset.py。本文为定向源码迁移审计，不是完整 survey 或绝对原创声明。

新增是当前 GMR null-set 接口、共享 fusion 训练与最终 existence BCE/ranking 的连接，以及 semantic holdout AUROC 效果筛查。暂定 B 候选；单 seed 初步收益即便出现也不能代替数据/预算/机制消融，不把已有层级 ranking 改名当创新。
