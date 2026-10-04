# FlashVTG-GMR + SHINE coarse/fine: A1/C1 50 epochs

> 更新：用户明确授权利用显存余量与现有任务共用 GPU，已停止仅等待的 Flash quiet queues 并直接启动两卡各 B0/S1 worker。训练配方与原冻结输入保持不变；旧等待队列状态保存在 records/quiet_queue_superseded/，调度变更见 records/IMMEDIATE_LAUNCH_OVERRIDE.json。下方空闲等待规则为历史初始方案。

Independent experiment authorized in this side conversation. Canonical initialization per split, seed3407, total50 epochs, matched B0 baseline vs S1_saliency_only. Original GMR architecture/criterion retained; coarse/fine weights1/1; no rotated existence BCE or pair supervision. Adaptation lr1e-5/batch16/wd1e-4/clip.1. Earliest best trained Seen-val AUROC selects checkpoint; formal Unseen never selects epoch/weight/threshold.

Native Flash dataset/criterion/forward inputs and native second-based top1 proposal evaluation integrated in isolated launchers. Model and native dataset modules are unmodified physical copies. CPU smoke passed finite forward/backward and native inference on actual A1/C1 train batches; no optimizer/test access. Process-local nncore.Config compatibility shim only; shared environment unchanged.

- Plan: [EXPERIMENT_PLAN](configs/EXPERIMENT_PLAN.md)
- Freeze: [EXPERIMENT_FREEZE](configs/EXPERIMENT_FREEZE.json), 46,381 input files including code/checkpoints/data/features
- Integration check: [SMOKE_CHECK](records/SMOKE_CHECK.json)
- GPU0/A1 and GPU1/C1: independently wait for no compute processes for60 seconds, then launch B0/S1 concurrently
- Queue status: records/queue_A1.json, queue_C1.json; worker status: records/worker_A1.json, worker_C1.json
- Runs: runs/formal50/<split>/<arm>/seed3407/flash_saliency_v1_50ep/
- Training logs: records/<split>_<arm>_training_50ep.log
- After both arms complete50: auto-freeze selected checkpoints/exposure, evaluate canonical/B0/S1 Seen/Unseen with paired video-cluster CI, conditionals/localization/fresh shuffled-video, generate per-split and combined records/evaluation/RESULTS.json/.md

Queues are detached and do not stop/interfere with current QD/Moment tasks. Inspect live GPU processes and queue/worker/run statuses before recovery; refuse duplicate or overwriting runs. Current A1/C1 benchmark is exploratory; a smaller Gap caused by Seen damage alone is not success. No multi-seed or five-split expansion. No sibling or parent experiment files modified.
