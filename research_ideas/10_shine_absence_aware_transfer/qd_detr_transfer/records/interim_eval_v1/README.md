# User-authorized interim Unseen evaluation

Training has not yet completed50 epochs. Parent QD training remains running; final automatic evaluation is unchanged. Separate frozen snapshots, histories and evaluator in this directory preserve checkpoint identities while live best.ckpt files continue updating.

Observed training epochs at snapshot: A1 B0=28/S1=26, C1 B0=16/S1=16. Selected trained epochs: A1 B0=24/S1=1, C1 B0=1/S1=1, all by earliest best completed Seen-val. Canonical included as replay reference. FREEZE.json was written before test-label evaluation. Test is exploratory, not used to change checkpoints, thresholds, loss weights or training plans.

Results: A1/RESULTS.json/.md, C1/RESULTS.json/.md; combined RESULTS.json/.md after both finish. Predictions and paired video-cluster intervals retained. Unequal currently observed A1 budgets and single seed limit interpretation. Compare final50-epoch outputs separately in records/evaluation/.
