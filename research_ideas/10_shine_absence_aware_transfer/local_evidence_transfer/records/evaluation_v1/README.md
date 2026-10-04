# Moment local evidence: formal A1/C1 exploratory evaluation

User explicitly authorized Unseen evaluation after training finished. This is Moment-DETR-GMR, not QD-DETR. New arms Local_CF/Uniform_CF/Local_noCF,10epochs,seed3407. Checkpoints and thresholds selected by Seen-val only. A1/C1 test was previously examined and remains exploratory.

EVALUATION_FREEZE.json verifies all six new selected checkpoints, frozen training inputs, code and controls; all10 epoch exposure hashes match. Same-budget B0/S1 controls use immutable, hash-verified v2 predictions, copied into this evaluation folder and recalculated against the same test identities. No parent result files are overwritten.

A1_evaluation.log / C1_evaluation.log track evaluations on GPU0/GPU1. Per-split outputs: A1/ and C1/, containing RESULTS.json/.md, predictions and status. Combined report: RESULTS.json/.md after both complete. Evaluator snapshot: EVALUATOR_EXECUTED.py.

Comparisons: all new arms vs B0 and old S1; Local_CF also vs Uniform_CF and Local_noCF. Metrics include Seen/U, gap, conditions, localization, fresh shuffled-video with retained labels, base-head and local-residual inference decomposition.1000 paired video-cluster bootstrap intervals are conditional on one trained seed; no macro split-independence CI. Component score decomposition is not a retraining ablation.

No checkpoints, temperatures, branch coefficients or thresholds are changed using test; no further training is auto-started. All mutations stay within local_evidence_transfer.
