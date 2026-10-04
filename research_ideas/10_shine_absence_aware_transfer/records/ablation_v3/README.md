# Step2 execution: BCE/pair separation and weak-BCE repair

User authorized launch on 2026-10-04. All new arms keep GMR + coarse/fine. BCE_only uses BCE1/pair0; Pair_only uses BCE0/pair.2; Weak_BCE_pair uses BCE.1/pair.2. Canonical initialization, seed3407, 10 epochs, Seen-only. No formal test evaluation or further repairs automatically scheduled.

Plan: `../../configs/ABLATION_V3_PLAN.md`; freeze: `../../configs/ABLATION_V3_FREEZE.json` (23,244 files including 36 v2 reference artifacts). Existing v2 and original formal source hashes verified unchanged. V3 training differs only in arms/weights, output paths and configuration metadata; training and sampling mechanics are retained.

GPU0: A1 three simultaneous trainers. GPU1: C1 three simultaneous trainers. Status: `worker_A1.json`, `worker_C1.json`. Logs: `<split>_<arm>_training_10ep.log`; runs: `../../runs/ablation_v3/<split>/<arm>/seed3407/component_v3_10ep/`. Workers are detached; inspect real processes and statuses before any recovery action, and never overwrite live/completed runs.

When all three arms on a split finish, the worker verifies selected checkpoint hashes and per-epoch exposure against v2 B0/S1/SE1, then generates `<split>_RESULTS.json` and `.md`. Summary reports both ΔSeen vs B0 and vs S1. Original pair component is already multiplied by .2 in training; enabled pair multiplier1 means effective .2.
