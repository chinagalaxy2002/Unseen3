# 09：Prior-controlled evidence increment（PEI-0）

状态：`implemented_not_executed`。Stage 0/1/2 代码均已实现；尚未运行真实数据审计或 PEI-0 实验。13 项协议检查通过，包含合成数据完整四折流程；这些检查不构成真实任务的性能结果。已有 V5 结果属于历史依据。

研究目标：检查冻结 CLIP 通道在控制 query-only logit、query 偏移和 video 偏移后，能否带来语义留出存在性判别的增量。遵守 [COMMON_PROTOCOL](../COMMON_PROTOCOL.md)，只读取四个 strict-inner folds；正式 U 不参与本实验。

- [冻结配置](records/FREEZE.json)：固定 folds、manifest hashes、seed、聚合选择、统计和推进门槛。
- [实验规范](EXPERIMENT_PLAN.md)：给出估计目标、中心化、交叉拟合和声明边界。
- [Stage 0 脚本](code/audit_stage0.py)：只读标注/原始特征；输出限制在本目录，并拒绝覆盖记录。

在仓库根目录运行标签层审计：

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/audit_stage0.py
```

可选的 train-only 特征诊断，包括 GT 内外相似度、三个聚合器的原始 AUC、100 次 shuffled-video 原始分数 AUC：

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/audit_stage0.py \
  --features \
  --output research_ideas/09_prior_controlled_evidence_increment/records/STAGE0_FEATURE_AUDIT.json
```

依赖 Python 3.10+；Stage 0 使用 NumPy，完整流程还使用 SciPy、scikit-learn、PyTorch 和 threadpoolctl，版本下限见 `requirements.txt`。默认标注来自 `experiments/moment_detr_gmr_evidence_v5/inner_folds/`；原始视频来自 `/home/guoxiangyu/paper/新建文件夹/charades/vid_clip`；文本来自 `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/features/semantic_existence_v2/{A1,C1}/clip_text`。可以用 `--clip-root` 和 `--text-base` 指定只读位置。输入文件逐一记录 SHA256。全部视频特征在单个 fold 内缓存，内存占用由视频长度决定；未测量时长和峰值内存。

GT 内外 AUC 默认采用 `(clip_index + 0.5) × 1 秒` 的名义中心，只是诊断，不会通过 alignment gate。若有可核验的提取时间来源，用 `--timestamps path.json` 提供：

```json
{
  "provenance": "实际提取脚本版本及时间坐标说明",
  "videos": {"video_id": [0.5, 1.5, 2.5]}
}
```

每个正例视频必须提供与原始视频特征长度一致、严格递增的 clip 中心秒数；不能把上面的示例或名义中心包装成真实提取记录。gate 要求输入完整且 train 正例 row-macro 内外 AUC 的 video-cluster 95% CI 下界 > 0.5。未通过仅限制该通道，不能证明事件证据不存在。

Stage 0 的 shuffled AUC 是依赖性诊断；实际标签在换视频后不再是对应事件的标签，所以 null AUC 不必为 0.5。完整流程另行计算重拟合 M2/M3 后的宏增量 null，并运行条件功效模拟。冻结 revision 2 保存全部代码 hash，revision 1 作为历史规范保留。

## Stage 1/2 运行

在有真实提取时间来源时先生成与当前冻结版本匹配的审计：

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/audit_stage0.py \
  --features --timestamps /path/to/extraction_clip_centers.json \
  --output research_ideas/09_prior_controlled_evidence_increment/records/STAGE0_VERIFIED.json
```

所有 fold alignment gate 通过后，运行纯零训练分析：

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/run_pipeline.py \
  --stage stage1 --run-id pei_stage1_v1 \
  --audit research_ideas/09_prior_controlled_evidence_increment/records/STAGE0_VERIFIED.json
```

完整流程包含每折 5 seed × 5 个 Cq、双中心化、M2/M3、Cv 历史原始分数对照、零拟合组合、单侧中心化、时间聚合对照、100 次 shuffle-refit、共享视频 bootstrap 和条件功效模拟：

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/run_pipeline.py \
  --stage full --run-id pei_full_v1 --device cpu \
  --audit research_ideas/09_prior_controlled_evidence_increment/records/STAGE0_VERIFIED.json
```

也可指定 `--device cuda:0`；执行身份记录 device 与库版本。所有输出在 `runs/<run-id>/`，包含 Cq 选模历史及训练/验证/OOF 行索引、输入 SHA256、全精度预测、null、条件 MDE、RESULTS 和输出 SHA256。运行中断后，相同命令复用已核验 Cq checkpoint；已完成的 run 拒绝覆盖。`runs/` 与本机 Stage 0 输出均不提交到 Git，避免发布模型和逐行数据。

Pipeline 会核对 freeze、Stage 0 audit、代码 hash、输入特征 hash 及所有标注 manifest；缺少门控或编码不一致时停止。Cv 只读四折 V5 已有 `Cv_P3_bce_seed3407/predictions.jsonl` 的 `learned_logit`，不会使用 fallback；文件 hash 和角色/行身份均检查。其训练 recipe 与交叉拟合 Cq 不同，结果明确记录这一限制。

功效模拟固定 b 和特征，按视频共享随机效应与同 video/query 的随机数，在生成标签后重新拟合 M2/M3。每个角色校准生成标签的期望正例率到审计中的自然比例。视频随机效应 SD 为 0、0.5、1，系数网格、200 次重复和 80% power 均在 FREEZE 中固定。MDE 取各敏感性设置的最大值；无法找到阈值时保留空值并输出 inconclusive。它是依赖假定生成模型的条件估计，不能替代无条件功效，也不包含 Cq 重训不确定性。

## 验证

```bash
python research_ideas/09_prior_controlled_evidence_increment/code/test_pei.py
```

检查使用合成数组和临时目录，不读取 benchmark 特征/标注，不访问正式 U。完整流程检查使用缩减的合成配置，不改变真实 FREEZE。检查覆盖精确 leave-self-out、可加先验消除、无固定点视频置换、OOF/内部验证隔离、固定 b 系数、零方差处理、跨折共享视频重采样、巨大连通分量限制、MDE 门控及相同 query 输入分数广播。

缺少时间来源时仍可运行默认 Stage 0 标签审计和名义特征诊断，但不能通过换用名义时间绕开完整流程门控。
