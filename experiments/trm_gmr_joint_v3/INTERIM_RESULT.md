# Joint-v3 fixed-50-epoch restart

All five splits restart from scratch and train exactly 50 epochs, with early stopping disabled. `best_mAP.ckpt` is always saved after validation. `best.ckpt` prefers the constrained worst-semantic Seen AUROC checkpoint when available; otherwise, it is the Seen-val mAP best fallback and is evaluated with `localization_constraint_satisfied=false`. Calibration remains Seen-only.

Run ID: `run_02_fixed50_20261003T080623Z`.

GPU0: A1 → A3 → C2_alt. GPU1: A2_alt → C1. Each split automatically trains, calibrates, and evaluates, followed by aggregation. The restarted run has no final metrics yet.

The previous 30-stale-epoch run is archived separately. Its metrics are historical and are not combined with this restart.

- Results archive: `results/moment_detr_trm_gmr_joint_v3_archive/run_01_patience30_20261003T080623Z`
- Documents and prior freeze: [experiments/trm_gmr_joint_v3/archive/run_01_patience30_20261003T080623Z](archive/run_01_patience30_20261003T080623Z/INTERIM_RESULT.md)

See `RESTART_RECORD.json` and the current `EXPERIMENT_FREEZE.json` for the user-requested protocol amendment. Model architecture, losses, semantic groups, sampling, seed, and learning rate are unchanged.
