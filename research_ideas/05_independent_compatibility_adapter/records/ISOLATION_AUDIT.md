# Isolation audit

- Runtime import smoke test resolved the copied baseline module to `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/research_ideas/05_independent_compatibility_adapter/vendor/baseline_moment/models/moment_detr_gmr/moment_detr.py`.
- No original model/training code, other idea code, original data, V5 cache, or existing result was edited. The local vendor is a physical copy; hashes and lineage are in `SOURCE_MANIFEST.json`.
- The two strict-inner checkpoints and used bank-array hashes are recorded in `SOURCE_MANIFEST.json`. Each source row file was copied locally and matched to the feature-bank qid/vid/query/label identity before use. Feature source file hashes and per-role counts are in `SUPPORT_AND_FEATURE_AUDIT.json`.
- All new training, predictions, metrics, logs, and shuffle JSONL inputs are written under this idea's `runs/`. The baseline evaluator's temporary substituted rows are also inside the relevant run directory.
- Formal U was not read. GPU 0 was checked idle before queue startup and reserved per task; GPU 1 remains unused.
