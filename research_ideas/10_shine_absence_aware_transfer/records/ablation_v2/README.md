# A1/C1 ablation v2 execution

2026-10-04: user authorized starting the A1/C1 four-arm experiment, replacing five-split completion as the current task.

- Plan: `../../configs/ABLATION_V2_PLAN.md`
- Freeze: `../../configs/ABLATION_V2_FREEZE.json`
- GPU0 worker: `worker_A1.json`; log: `A1_training_10ep.log`
- GPU1 worker: `worker_C1.json`; log: `C1_training_10ep.log`
- Runs: `../../runs/ablation_v2/<split>/<arm>/seed3407/component_v2_10ep/`
- Automatic per-split reports when all four arms complete: `A1_RESULTS.json/.md`, `C1_RESULTS.json/.md`

Current workers were verified executing optimizer steps: A1 B0 reached epoch1 step200; C1 B0 reached epoch1 step100. Initial checkpoint Seen-val reproduced A1 .8366800535 and C1 .7882616211. First-batch gradient records exist on both splits and B0 active auxiliary gradient is zero, as intended. No formal test accessed.

An initial logging failure occurred before the first optimizer update because an integer criterion diagnostic was treated as a tensor. Original attempts, executed source and initial freeze are preserved in `startup_failure_01/`. The active freeze records the scalar-logging repair; objectives and design were unchanged.

Training is detached and continues after this response. Each worker stops on failure. To resume, first inspect real processes plus worker/run status and logs; do not rerun live workers or overwrite run directories. No automatic repair-stage experiment is scheduled: interpret these Seen-only results before freezing the next stage.

## Parallel scheduling update

2026-10-04: user authorized up to three simultaneous trainers per GPU. Independent E1/SE1 launchers now run alongside adopted S1; the previous waiting workers were retired, and their state is archived in `serial_scheduler_handoff/`. S1 was briefly suspended and resumed in-place after independent run records were created, without losing model/optimizer state. The old all-arm trainer is terminated only after S1 completion; its existing-run guard may also exit it before attempting a duplicate E1. Such old-launcher exit is an expected scheduling handoff, not an experiment failure.

Scheduling freeze: `../../configs/ABLATION_V2_PARALLEL_FREEZE.json`. Original training freeze remains unchanged. Independent logs: `<split>_<arm>_training_10ep.log`. Per-arm import audits and result files are isolated. Supervisors still generate the same per-split reports after all four arms finish. A1 S1 completed during the transition, so A1 now has two remaining active jobs; C1 runs S1/E1/SE1 concurrently. No additional arms were invented to occupy freed capacity.
