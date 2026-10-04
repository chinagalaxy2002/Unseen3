# Idea 07 isolation audit

- All executable model, loader, trainer, evaluator, support-audit, and job-preparation code used by this direction is under `code/`; 05 readout/evaluation dependencies were copied here and their source/copy hashes are in `SOURCE_MANIFEST.json`. No runtime import reaches another idea or the original baseline source tree.
- Inner train, Seen validation, and Novel-dev annotations were copied into this direction. The reviewed source-pair unit manifests and feature-bank arrays are local copies. Exact query feature files used for each fold were copied locally and byte-checked.
- Raw CLIP/SlowFast video feature files are shared inputs opened read-only; every train/Seen/Novel video-feature file hash is recorded in `VIDEO_FEATURE_MANIFEST.json`. Loading uses `np.load`; the code writes no sidecars to shared input locations.
- The model receives raw per-video sequences and query tokens; video shuffling replaces raw video sequences before the adapter's query cross-attention and score computation. Original labels are retained only for the dependence diagnostic.
- Frozen localization and canonical baseline existence scores are read from copied strict-inner feature-bank outputs. Inner checkpoint identities and source-array hashes are recorded.
- Logs, configs, checkpoints, predictions, metrics, and task-specific temporary products are directed to this idea's `runs/` or the shared queue's `specs/` and `records/`. No formal U rows or official test predictions were read.
- Original baseline project/source, other idea directories, shared labels, raw features, and prior results remain read-only.
