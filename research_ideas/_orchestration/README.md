# Persistent GMR research queue

The queue is deliberately empty outside the two-fold Idea 01 pilot. It lives under this directory and does not write into the baseline repository, shared release, V5 results, or another idea.

## Runtime

- `queue.sqlite3` uses SQLite WAL, `synchronous=FULL`, atomic `BEGIN IMMEDIATE` claims, and an append-only transition/event table.
- Every GPU worker holds one database resource reservation for its lifetime. Each task runs training, validates and records the checkpoint, evaluates that checkpoint, validates prediction count/schema and metrics, then claims the next task.
- A task stores its idea, backbone, fold, seed, resolved config and hash, code hash, data-version identity, GPU, worker/child PID plus `/proc` start token, timestamps, logs, checkpoint, predictions, metrics, retry count, and error class.
- States are `planned`, `ready`, `running`, `trained`, `evaluating`, `succeeded`, `failed`, `blocked`, and `insufficient_evidence`. Exit code alone never produces `succeeded`.
- `recover` never sends signals. It checks the recorded worker PID/start token and, if the worker is gone, defers recovery only when the recorded child PID/start token, task environment, and argv still match that task's frozen train/eval command. Otherwise it reuses a validated checkpoint, resumes evaluation when possible, or returns an incomplete task to `ready` within its two-retry limit.
- GPU startup requires `nvidia-smi` to report no compute process. A failed device query blocks launch. The worker checks again before each task. This run uses only GPU 0; GPU 1 stays unreserved.
- Logs and outputs are inside each idea’s `runs/`. Source caches and the strict-inner checkpoint are read-only.
- CPU metrics run as part of the evaluation stage while that task retains its GPU reservation. Independent CPU-only task workers can be added without changing GPU ownership.

## CLI controller and recovery limits

The installed `codex exec --help` was inspected on 2026-10-04. The bounded repair path uses the supported `codex exec --cd <idea> --sandbox workspace-write --output-last-message <file> <prompt>` interface and an explicit subprocess timeout (600 seconds for this run). It runs only after two deterministic task retries fail with an engineering/validation error. Repairs are confined by prompt and post-run path checks to that idea’s `code/`; a repaired run gets a new task ID, config hash, and output directory. At most two repairs are allowed per failed task. The controller itself allows at most two GPU worker starts per persisted controller state; raise that limit explicitly for a later batch. It exits after the queue stays empty for its configured grace period, so later work requires starting a new controller session.

Example startup from the project root:

```bash
tmux new-session -d -s gmr_research_controller -c "$PWD" \
  'python research_ideas/_orchestration/controller.py --gpus 0 --max-worker-restarts 2 --max-repairs-per-task 2'
```

Inspect with:

```bash
python research_ideas/_orchestration/orchestrator.py status
python research_ideas/_orchestration/orchestrator.py recover
```

`controller_state.json`, SQLite events, worker logs, and each task’s run files survive shell disconnects. Do not delete/reset them to recover a task. Add new folds/seeds only after the current pilot decision is written and the next task config is frozen.

## Current research state (2026-10-04)

- Idea 01's exact-query pair fallback pilot and pre-fusion shuffled-video controls completed. Its disposition and limits are in `01_crossed_existence_supervision/records/`.
- Idea 02's two-fold rank-32 reference-centering screen completed. Small pooled changes have uncertainty spanning zero; the same-query mechanism signal is exploratory and query-feature-cache-sensitive. See its `records/`.
- Idea 03 is blocked for the explicit binding objective: strict-inner support has reviewed full-query absences but too few conservative independent atom-witness candidates and no validated substitute. No training was run.
- Idea 04's compute-matched M0/M1/M2 pilots and all six pre-fusion shuffled-video re-forwards completed on two strict-inner folds, one seed. Balanced-query M2 showed an A1 pooled gain that tracks cross-query ranking, with lower same-query PairAcc, a 1.72pp Seen drop, and a stronger query-only AUROC; C1 was near null. Disposition is `insufficient_evidence`; do not treat the action-fold gain as compatibility. See `04_query_marginal_matching/records/`.
- Idea 05 is now the active priority. Preliminary audit confirms strict-inner splits/checkpoints and raw CLIP/SlowFast sequence directories exist (9,848 files each); the shared time-grid audit reports modality length differences at most one and no embedded timestamps. The plan permits relative sequence indices and forbids unsupported second-level ROI claims. Identity and exact query-feature provenance still need a local audit before freezing a pilot.

The startup GPU check found GPU 0 and GPU 1 available; only GPU 0 has been assigned to this research, and GPU 1 has remained unused. Official U has not been accessed. Formal testing remains frozen until a candidate is selected.

- Idea 06's frozen A0/A2 candidate-scorer screen completed on two strict-inner folds, one seed. Reviewed-negative candidate BCE did not improve pooled Novel AUROC (mean −0.45 pp); A1 same-query PairAcc increased but C1 did not replicate it, while same-video PairAcc was unchanged and query-only AUROC exceeded the scorer on both folds. Disposition: `insufficient_evidence`; no grouping expansion. Full evidence: `06_evidence_aware_mil_aggregation/records/`.
- Idea 07 is next for a training-side complete-pair environment support audit. No training starts until the audit confirms multiple adequately supported environments and the pair-signal gate; current controller has exhausted 14 of its prior 16 worker-start allowance, so a new bounded controller state/cap will be needed for a later frozen batch. GPU 1 remains unused; official U remains unopened.

- Idea 07 passed its one-epoch end-to-end queue smoke after two isolated metadata/tuple fixes. The frozen source-linked pair ERM gate is now running: two strict-inner folds, one seed, B0 natural BCE + pair-row replay versus B1 with the additional same-video reviewed-pair margin loss, equal pair visits/optimizer updates, Seen-only checkpoint selection. `PAIR_GATE_FREEZE.json` and the environment/feature manifests record the exact plan. MLDG remains gated on conditional evidence from this pair screen. GPU 0 is the only assigned device; GPU 1 is untouched.

- Idea 08 time/label gate was audited without training: zero embedded timestamps, no reproducible per-feature extraction mapping, 532/4,918 videos with GT end beyond nominal feature support, no raw source videos in the accessible data tree, and no all-occurrence/transformed-video review labels. The documented pair substitute belongs to Idea 01 and is not an evidence intervention. Disposition: `blocked`; audit and reasons are in `08_evidence_intervention/records/`.

- Idea 07's source-linked pair ERM gate completed: pooled Novel AUROC changes were +0.00pp A1 and +1.72pp C1 (mean +0.86pp; both paired video-cluster intervals cross zero). Same-query conditional direction differed across folds; same-video source PairAcc did not improve. Disposition: `insufficient_evidence`; no MLDG/IRM/Group-DRO run is justified by this gate. Full results and decision are under `07_compatibility_domain_generalization/records/`. All Idea 07 queue tasks succeeded; GPU worker is idle.


## Superseding current state (2026-10-04)

The earlier interim bullets above accurately record when Ideas 05/06/07 were active; later entries supersede their “active/next” wording. All eight directions now have a disposition. Ideas 01, 02, 04, 05, 06, and 07 are `insufficient_evidence`; Idea 03 is `blocked` for a sufficiently large independent witness challenge; Idea 08 is `blocked` for time provenance and transformed labels. Idea 07's four frozen pair-gate tasks and its successful pipeline smoke are complete, and no GPU task remains active. No method is promoted, so formal U evaluation stays closed.

## Current queue and research state (2026-10-04, supersedes interim status above)

The two frozen Idea 05 Seen-selected query-prior correction evaluations completed and passed checkpoint, prediction-coverage, and metrics validation. Both tasks reused existing strict-inner raw-sequence checkpoints; no training was repeated. A1 selected λ=.25 and its Novel-dev AUROC fell 3.83 pp with a video-cluster paired interval excluding zero; same-video source-pair accuracy also declined. C1 selected λ=0. This does not support the correction or promote the adapter. The queue has no running/ready tasks; GPU 0 and GPU 1 are unreserved, and the detached controller exited after its idle grace period. Historical failed attempts remain archived. See [`RESULT_SYNTHESIS.md`](RESULT_SYNTHESIS.md) and Idea 05's [full disposition](../05_independent_compatibility_adapter/records/DIRECTION_DISPOSITION.md).
