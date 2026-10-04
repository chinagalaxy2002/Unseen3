# A1/C1 component ablation v2

Authorized 2026-10-04. Scope supersedes five-split completion: do not restart old formal workers. Existing formal artifacts stay preserved.

Research question: does early Seen damage originate in coarse/fine saliency supervision, added rotated existence supervision, or their interaction?

A1 and C1, canonical initialization per split, seed3407 only, 10 epochs, lr1e-5, batch16, original weight decay and clip .1. Natural train and Seen-val only; no formal test access. Best trained epoch selected by Seen-val pooled AUROC, earliest tie. Epoch0 is diagnostic only, never a fallback. No early stop.

| Arm | coarse | fine | rotated BCE | pair |
|---|---:|---:|---:|---:|
| B0 | 0 | 0 | 0 | 0 |
| S1_saliency_only | 1 | 1 | 0 | 0 |
| E1_rotated_exist_only | 0 | 0 | 1 | .2 |
| SE1_full | 1 | 1 | 1 | .2 |

All arms compute the natural, rotated and three-level edit forwards with identical sampling and order. Inactive terms are omitted from backward, avoiding artificial zero gradients and AdamW weight decay on unused heads. B0 is a matched-forward control; do not assume its trajectory equals historical baseline50. Per-epoch exposure hashes track batch qids and selected edit indices. Existing user-confirmed rotated absence and hierarchy assumptions remain accepted.

Save epoch0 and every epoch Seen predictions, losses including original criterion components, positive/negative score distributions, pooled/same-video source-pair/same-query metrics, and raw/gated localization. Preserve checkpoints at epochs1/3/10 and best. Probe first training batch at epochs1/3/10: component gradient norms and cosine to GMR on shared transformer/input projections, in train mode, before clipping. These probes do not update parameters or consume RNG; they remain local geometry rather than AdamW causal attribution.

GPU0: A1; GPU1: C1. Each runs B0 → S1 → E1 → SE1 and stops on failure. Summary generated automatically per split. No repair parameters will be selected using formal Unseen. Subsequent finer ablation and repair designs require a new recorded freeze based on these Seen results.
