# Step1 conclusion and proposed Step2

All eight seed3407 runs completed 10 epochs. Selected checkpoint hashes and all frozen inputs verified. Within each split, all four arms have matching exposure hashes across all 10 epochs (natural qid order and edit indices). No new formal test access.

| Arm | A1 best Seen | C1 best Seen | A1 Δ vs B0 (pp) | C1 Δ vs B0 (pp) |
|---|---:|---:|---:|---:|
| B0 | .8456 | .7844 | 0 | 0 |
| S1 saliency only | .8473 | .7901 | +.17 | +.57 |
| E1 BCE + pair | .7852 | .7536 | −6.04 | −3.07 |
| SE1 full | .7810 | .7599 | −6.46 | −2.44 |

Observation: existence auxiliary bundle is sufficient to reproduce substantial Seen damage in this fixed initialization/seed/10-epoch setting; saliency alone avoids this collapse. Small S1 improvements are descriptive, not evidence of Unseen gain or statistical significance. BCE and pair are not individually separated yet. Early gradient probes prioritize BCE, but do not prove it is the sole cause. Some conditional metrics improve while pooled AUC falls; therefore do not conclude all conditional ranking is damaged uniformly.

Proposed Step2 (not launched): retain GMR + coarse/fine and compare three new arms on both splits:

1. BCE_only: + 1.0 rotated BCE, pair disabled.
2. Pair_only: + 0.2 existence pair, BCE disabled.
3. Weak_BCE_pair: + 0.1 rotated BCE + 0.2 existence pair.

S1 (neither) and SE1 (both at original weights) provide existing same-design controls, forming a component factorial with the first two new arms. The third is a single prespecified repair candidate. Maintain canonical initialization, seed3407, 10 epochs, lr1e-5, batch16, weight decay, clipping, exact same forwards and sampling. Use a new independent v3 entry, freeze, run directories, and records; verify exposure matches before using v2 as matched references. Three new runs per GPU, A1/GPU0 and C1/GPU1.

Read only train/Seen-val. Assess epoch1 collapse, epochs3/10, trained best Seen AUROC, GMR loss, gradient geometry, score spread, conditional ranking and localization. Epoch0 remains diagnostic only. Weight .1 is a new Seen-only development candidate chosen before Step2, not a historical coefficient or a proven solution. No coefficient search using formal Unseen.

If Pair_only preserves Seen and BCE_only damages it, BCE is sufficient for the harm under the saliency setting. If weak joint supervision still damages Seen, consider a separately frozen warmup or removing BCE while retaining useful pair supervision. If both isolated terms are benign but full is harmful, investigate interaction. Formal Unseen and longer budget validation follow only after a supported recipe is frozen; A1/C1 test remains exploratory because already examined.
