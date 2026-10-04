# Exploratory formal test, v2/v3

User authorized formal Unseen evaluation and proceeding to next stage. Seven adapted arms plus canonical per split, A1/C1 only. All adapted arms trained 10 epochs with seed3407; selected checkpoints are the earliest best trained Seen-val epochs. Selected checkpoint hashes and matched exposure verified before reading formal test.

Gate: EVALUATION_FREEZE.json. Executed evaluator snapshot: EVALUATOR_EXECUTED.py. Logs: A1_evaluation.log / C1_evaluation.log. Per-split status/results/predictions: A1/ and C1/. Combined report: RESULTS.json / RESULTS.md after both finish. Paired video-cluster 1000-resample CI is conditional on the trained seed; no macro independence assumption.

V4 reduced-scale plan was written and hashed in this evaluation gate before evaluation. The new Unseen outcomes will not change its arms or weights. A1/C1 test is already examined and remains exploratory, not an untouched confirmation set. Old formal and training result files are preserved.
