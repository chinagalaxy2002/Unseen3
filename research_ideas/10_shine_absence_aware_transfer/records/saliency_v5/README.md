# Coarse/fine-only: matched total-50-epoch comparison

User requested quantifying baseline Seen→Unseen gap and training longer. The active experiment trains from canonical for a fresh total50 epochs per arm; it does not append50 to an already selected 10-epoch checkpoint (those artifacts lack full optimizer/RNG continuation state). Both B0 and S1 are rerun for a fair matched-forward/exposure control. Existing checkpoints and results are preserved.

10-epoch reference: A1 B0 gap32.15pp → S1 gap32.55pp (worsens .40pp); C1 18.19pp →17.55pp (improves .64pp). Two-split macro25.17pp →25.05pp (reduction .12pp, relative .47%). S1 macro Unseen +.38pp and Seen +.26pp; no consistent two-split mitigation yet. Exact values in TEN_EPOCH_GAP_REFERENCE.json.

Plan: ../../configs/SALIENCY_V5_PLAN.md. Freeze: ../../configs/SALIENCY_V5_FREEZE.json. GPU0/A1 and GPU1/C1, two concurrent trainers per card (B0 and S1_saliency_only). Seed3407, total50 epochs, lr1e-5, batch16, original wd/clip, best trained Seen-val AUROC/earliest tie, epoch0 diagnostic only. No multi-seed, no BCE/pair auxiliary training objective.

Worker status/log: worker_A1.json/.log, worker_C1.json/.log. Training logs: <split>_<arm>_training_50ep.log. Runs: ../../runs/saliency_v5/<split>/<arm>/seed3407/saliency_v5_50ep/.

After both arms on each split complete, the worker automatically verifies checkpoint selection/hash and matched exposure, freezes the evaluation gate, then evaluates formal Seen/Unseen with Seen-val thresholds. Per-split reports/predictions: evaluation/<split>/. After both splits finish, aggregate evaluation/RESULTS.json and .md report ΔU, ΔSeen, gap reduction and relative reduction. Formal A1/C1 test remains exploratory. Automatic evaluation is scheduled in this worker, unlike old formal queues; do not start a duplicate evaluator.

Training is detached. Inspect real processes plus current statuses before recovery; do not overwrite/restart a live or completed run. Old v2/v3/v4 sources and artifacts remain unchanged.
