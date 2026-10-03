# Moment-DETR-GMR Residual-AUC-v4

**Status:** exploratory follow-up. Joint-v1/v2/v3 U results were already inspected, so this is not an untouched unseen evaluation.

GMR-AUC-v4 freezes each split's original Moment-DETR-GMR checkpoint and trains one small bounded residual existence adapter. Its primary KPI is pooled unseen AUROC (`U+` vs `U−`). U never participates in training, pair construction, checkpoint selection, threshold calibration, baseline selection, or hyperparameter selection. The same architecture and hyperparameters are used for all five splits.

The original baseline checkpoints are under `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2/<SPLIT>/moment/best.ckpt`. `BASELINE_AUDIT.md` and `baseline_audit.json` record checkpoint hashes, saved configs, seeds, feature order, pooling, training metadata and checkpoint rule. Original video feature order is CLIP then SlowFast, confirmed from each checkpoint's saved `v_feat_dirs` and the dataset's ordered concatenation.

Run `python scripts/audit_gmr_baseline_v4.py`, implement and smoke-check code, then `python scripts/freeze_gmr_auc_v4.py` and commit/push the freeze before any test U inference. Execute `bash scripts/run_gmr_auc_v4_multisplit.sh`; it assigns A1/A3/C2_alt to GPU0 and A2_alt/C1 to GPU1. The split runner extracts Seen train/val banks, trains 50 adapter epochs, freezes the best Seen-val adapter, calibrates Seen-val threshold, then performs final test inference and bootstrap metrics.

The global AUC term is a pairwise differentiable surrogate, **not exact AUROC differentiation**. Epoch 0 is a legal best checkpoint. If selected, it means all learned adapters scored worse than the frozen baseline on Seen validation.

Outputs live only in `results/moment_detr_gmr_auc_v4/<SPLIT>/`, including feature-bank audits, `adapter_training_history.json`, `best.ckpt`, `test_predictions.jsonl`, and `v4_summary.json`. Full-precision logits are canonical. Final aggregate files are `experiments/moment_detr_gmr_auc_v4/MULTI_SPLIT_RESULT.md` and `multi_split_summary.json`. Large feature banks/checkpoints are retained locally and are not committed.
