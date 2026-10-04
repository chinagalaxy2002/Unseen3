# Isolation audit

- The copied baseline import resolves to `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/research_ideas/06_evidence_aware_mil_aggregation/vendor/baseline_moment/models/moment_detr_gmr/moment_detr.py`. Its complete Python closure is physically local and hash-tracked.
- Strict-inner rows and source pairs are local copies. Candidate banks, baseline checkpoints, shared video/text features, and source annotations are read-only. No original model, shared data, V5 cache, user document, or earlier result was changed.
- All method code, task configs, logs, checkpoints, predictions, metrics, and video-derangement JSONL files are under this idea.
- Candidate bank qid/vid/query/label identities and all 10 decoder slots were verified. No positive candidate labels are inferred from missing spans.
- Formal U was not accessed. GPU allocation will be recorded by the persistent queue; only a startup-confirmed idle GPU may be used.
