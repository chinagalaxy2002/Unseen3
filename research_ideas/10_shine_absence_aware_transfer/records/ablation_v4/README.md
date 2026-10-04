# Step3 execution: reduced BCE and pair supervision

User authorized next stage after v3. Design and weights were written before v2/v3 formal Unseen evaluation and hashed in that evaluation gate. Observed v3 Seen damage motivated reducing both auxiliary supervision strengths, with separate controls.

All arms retain GMR + coarse/fine. Effective weights:

| Arm | BCE | pair |
|---|---:|---:|
| Low_BCE_only | .01 | 0 |
| Low_Pair_only | 0 | .02 |
| Low_BCE_pair | .01 | .02 |

Canonical initialization, seed3407, 10 epochs, lr1e-5, batch16, original wd and clip. A1/GPU0 and C1/GPU1, three concurrent trainers per card. No formal test loaded by training, no epoch0 selection fallback. No further coefficient search or 50-epoch validation automatically scheduled.

Plan: ../../configs/ABLATION_V4_PLAN.md. Freeze: ../../configs/ABLATION_V4_FREEZE.json (23,285 files, including 72 historical v2/v3 reference artifacts). Status: worker_A1.json / worker_C1.json. Logs: <split>_<arm>_training_10ep.log. Runs: ../../runs/ablation_v4/<split>/<arm>/seed3407/component_v4_10ep/. Summary: A1_RESULTS.json/.md and C1_RESULTS.json/.md once each split completes.

Summaries require selected checkpoint hashes and all epoch exposure to match v2/v3 controls. Pair component in code has base coefficient .2; enabled v4 multiplier .1 yields effective .02. All previous frozen sources and runs remain untouched.
