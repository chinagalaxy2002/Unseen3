# Step2: BCE/pair separation and weak-BCE repair

Authorized 2026-10-04 after all eight v2 runs completed. A1/C1 only, seed3407, 10 epochs, canonical initialization, lr1e-5, batch16, original weight decay and grad clip .1. Three independent trainers per GPU: A1/GPU0, C1/GPU1. No formal test access.

All new arms retain original GMR + coarse/fine at weights1/1, identical auxiliary forwards and editing exposure. Natural train, Seen-val, feature bank, margins and chain sampling are unchanged from v2.

| Arm | rotated BCE | existence pair (effective) |
|---|---:|---:|
| BCE_only | 1 | 0 |
| Pair_only | 0 | .2 |
| Weak_BCE_pair | .1 | .2 |

The pair component in code is already multiplied by .2; ARM_WEIGHTS uses multiplier1 for enabled pair. Weak BCE .1 is one prespecified Seen development repair candidate, not a validated coefficient. Do not add unrecorded coefficient trials using formal U.

V2 B0, S1_saliency_only and SE1_full are historical same-design references. S1/SE1 and new BCE-only/Pair-only provide the component factorial on the saliency setting. Preserve and hash reference metrics, histories, predictions and selected checkpoints. Require all 10 epoch exposure hashes to match before treating them as matched references.

Best trained epoch by Seen-val pooled AUROC, earliest tie; no early stop, no epoch0 fallback. Save per-epoch predictions/metrics, original criterion losses, positive/negative score distributions, epoch1/3/10 gradients and checkpoints, selected checkpoint and hash. Read pooled AUC alongside source-pair, same-query ranking and raw/gated localization; conditional metrics can improve despite pooled damage.

Question1: is BCE alone sufficient for damage? Question2: can pair be retained safely? Question3: does one-tenth BCE avoid early damage in the combined saliency+pair setting? These runs diagnose and develop a repair; they do not establish Unseen improvement. Further warmup, coefficient changes, longer-budget or formal test evaluation are not automatically scheduled.

Independent sources/run paths/freeze: code/train_ablation_v3.py, code/worker_ablation_v3.py, code/summarize_ablation_v3.py; runs/ablation_v3/<split>/<arm>/seed3407/component_v3_10ep; configs/ABLATION_V3_FREEZE.json; records/ablation_v3/. V2 and original formal frozen artifacts remain untouched.
