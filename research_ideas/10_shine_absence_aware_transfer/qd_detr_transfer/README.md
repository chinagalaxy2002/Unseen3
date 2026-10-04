# QD-DETR-GMR + SHINE coarse/fine

This independent experiment was explicitly authorized in the side conversation: QD-DETR-GMR, coarse/fine saliency ranking, 50 epochs. It does not alter parent Moment sources, runs, freezes or handoff documents.

A1/C1 each compare B0 (original QD GMR objective) and S1 (GMR + coarse + fine), initialized from each split's canonical QD checkpoint. Seed3407; 50 total new adaptation epochs; lr1e-5, batch16, original wd1e-4/clip.1. QD native global-token saliency head and internal training-negative forwards are retained. All natural/rotated/three-level edit forwards and sampling are matched across arms. No extra rotated existence BCE or existence pair loss.

Real train-batch checks passed for both splits: strict canonical load, label consistency, finite main/coarse/fine loss, finite backward, nonzero saliency-head gradients. No optimizer updates in smoke checks. Evidence: records/SMOKE_CHECK.json. All code, checkpoints, train/Seen records, bank and natural/edit feature identities are frozen in configs/EXPERIMENT_FREEZE.json. Plan: configs/EXPERIMENT_PLAN.md.

Current startup status: queued behind existing parent-thread Moment GPU workloads. Detached queues `records/queue_A1.json` / `queue_C1.json` automatically launch each split once its assigned GPU has no compute processes for60 seconds. Main-thread processes are neither stopped nor modified. Resource-only prelaunch freeze revision is recorded; original freeze is retained in records/pre_queue_freeze/. No QD training steps have run yet.

GPU0: A1 B0/S1 concurrently when free. GPU1: C1 B0/S1 concurrently when free. Worker state then appears in records/worker_A1.json / worker_C1.json. Per-arm logs: records/<split>_<arm>_training_50ep.log. Runs: runs/formal50/<split>/<arm>/seed3407/qd_saliency_v1_50ep/.

After each split's two runs finish all50, the worker verifies earliest best trained Seen-val selection, selected checkpoint hashes and50 matched exposure hashes, freezes identities before loading test labels, then automatically evaluates canonical/B0/S1 on formal Seen/Unseen. It reports gap, relative gap reduction, ΔUnseen/ΔSeen, video-cluster paired CIs, same-video source pairs, same-query ranking, raw/gated localization and fresh shuffled-video controls. Epoch0 is diagnostic only; thresholds and selection use Seen-val. Formal test results remain exploratory.

Per-split outputs: records/evaluation/A1/RESULTS.json/.md and C1/RESULTS.json/.md. After both evaluations complete, records/evaluation/RESULTS.json/.md contains the descriptive two-split macro. Gap reduction caused by Seen damage is not success; no five-split or multiseed claim.

To recover, first read queue/worker/run statuses, logs and real processes. Do not rerun an existing queue, worker or run. Existing files are protected against overwrite. No separate parent-thread tasks, Flash transfer, extra hyperparameters or seeds are authorized by this experiment.

## Interim evaluation requested during training

User explicitly requested Unseen evaluation before50 completion. Independent immutable snapshots and exploratory results are in records/interim_eval_v1/. Snapshot times: A1 B0=28/S1=26, C1 both16; chosen epochs A1 B0=24/S1=1, C1 both1. Seen-only historical checks confirm snapshots equal earliest best under common budgets26 and16, respectively. Relative to B0: A1 ΔUnseen +.25pp/ΔSeen +.15pp/gap reduction +.10pp; C1 +1.05pp/+.44pp/+.61pp. Macro ΔUnseen +.65pp/ΔSeen +.30pp/gap reduction +.36pp. This does not replace final50 results. Training and automatic final evaluation remain running.
