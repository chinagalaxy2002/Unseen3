# Direct saliency evidence for existence: isolated Moment experiment

> Final status: all six 10-epoch training runs and A1/C1 Unseen evaluations completed. This is Moment-DETR-GMR. See [degradation report](records/evaluation_v1/DEGRADATION_REPORT.md) and [full results](records/evaluation_v1/RESULTS.md). The original plan and historical startup notes below are preserved.

Authorized in this side conversation on2026-10-04. All new code/config/records/runs live in this directory; no parent experiment source, run, freeze, git state or shared environment is modified. GPU0/GPU1 were free before launch. No sub-agents used.

The query-fused encoder video tokens are pooled by masked softmax saliency weights. A local MLP maps the pooled evidence to a residual logit added to the original decoder existence logit. Its last layer starts at zero, so canonical initialization predictions are exactly preserved. Natural existence BCE can then train the local branch and saliency selection directly. Neither GT nor query edits are needed in inference.

| Arm | Pool | coarse/fine | Additional existence BCE/pair |
|---|---|---|---|
| Local_CF | saliency weighted | 1/1 | none |
| Uniform_CF | uniform valid-token mean | 1/1 | none |
| Local_noCF | saliency weighted | disabled | none |

A1/GPU0 and C1/GPU1 each run these three arms concurrently. Seed3407, 10epochs, canonical initialization, lr1e-5/batch16/original wd and clip. Earliest best trained Seen-val AUROC checkpoint; epoch0 diagnostic only. All auxiliary forwards and batch/edit exposure are matched. Frozen v2 B0 and S1 are reference controls, not overwritten; summary requires equal exposure hashes.

Plan: [EXPERIMENT_PLAN](configs/EXPERIMENT_PLAN.md). Freeze: [EXPERIMENT_FREEZE](configs/EXPERIMENT_FREEZE.json), 23,260 files including physical local model copies and readonly parent input identities. Real A1/C1 train-only checks: [SMOKE_CHECK](records/SMOKE_CHECK.json), no optimizer updates or test access. Verified exact zero-init canonical replay, finite loss and local-head gradients, residual-to-saliency derivative, masked padding invariance, and uniform-control independence from saliency.

Workers are detached. State/log: records/worker_A1.json/.log and worker_C1.json/.log. Training logs: records/<split>_<arm>_training_10ep.log. Runs: runs/v1/<split>/<arm>/seed3407/local_evidence_v1_10ep/. Reports: records/A1_RESULTS.json/.md and C1_RESULTS.json/.md automatically after all three arms of a split finish.

Predictions contain original trained-head base_logit, new local_logit, combined logit and pool entropy. History adds residual magnitude and output-layer norm; all three arms have identical new parameter counts. Compare Local_CF/Uniform_CF for saliency selection vs extra capacity, Local_CF/Local_noCF for CF supervision, and Local_CF/old S1 for direct-branch increment.

This first stage is Seen-only development. No formal Unseen evaluator,50epoch run, extra weight search or other-backbone transfer is auto-scheduled. A1/C1 test remains previously explored if evaluated later. On recovery inspect actual processes and statuses first; do not overwrite/restart live or completed runs. Side experiment uses modest GPU memory and does not stop any parent process.
