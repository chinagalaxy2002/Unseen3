# Step3 prespecified before v2/v3 formal test evaluation

2026-10-04 user requested Unseen evaluation followed by next stage. Based only on completed v3 Seen development results: BCE-only damages best Seen vs B0 by A1 −6.25pp/C1 −1.68pp; pair-only by −1.25/−1.77pp; weak BCE.1 + pair.2 by −3.48/−1.55pp. Thus reducing BCE alone was insufficient, and pair cannot yet be treated as safe.

Prespecify one further reduced supervision scale with separate controls, rather than selecting coefficients from formal test results. Keep GMR+coarse/fine unchanged:

| Arm | rotated BCE | effective pair |
|---|---:|---:|
| Low_BCE_only | .01 | 0 |
| Low_Pair_only | 0 | .02 |
| Low_BCE_pair | .01 | .02 |

A1/C1, seed3407, canonical initialization, 10 epochs, lr1e-5, batch16, existing wd and clip. Identical forwards/sampling; verify exposure against existing v2 B0/S1/SE1 controls. Epoch0 diagnostic only; best trained Seen-val AUROC, earliest tie. Three jobs per GPU. Save same curves, predictions, gradients, checkpoints and metrics as v3. Independent v4 sources/freeze/runs/records. These are new exploratory Seen development trials, not a proven fix. No extra coefficients, warmup or 50-epoch runs automatically selected from formal U.

Separately evaluate all eight completed v2/v3 arms and canonical on A1/C1 formal Seen/Unseen, using already selected best checkpoints and Seen-val thresholds. Freeze identities before evaluation; report paired video-cluster intervals and conditionals. All A1/C1 test results remain exploratory. The planned v4 arms and weights are fixed before seeing these new Unseen results.
