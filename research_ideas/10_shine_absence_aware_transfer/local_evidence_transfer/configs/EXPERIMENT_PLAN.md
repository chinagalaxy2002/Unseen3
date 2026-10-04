# Direct local event evidence for existence: v1

Authorized in side conversation on 2026-10-04. All files and state are isolated within local_evidence_transfer; parent experiments, runs, freezes and configuration remain unchanged. No sub-agents. GPU0 andGPU1 were idle before launch.

Hypothesis: coarse/fine supervises query-fused encoder token saliency while the original existence head pools decoder slots; an explicit saliency-weighted encoder evidence path may improve transfer to natural existence classification. This is a candidate mechanism, not a proven cause.

For valid video tokens h_t and saliency s_t, a_t=masked_softmax(s_t), e=sum_t a_t h_t, and z=z_original+MLP_local(e). Temperature1. Local MLP has hidden width equal to encoder width, ReLU, no dropout; its final linear layer starts at exactly zero. All original canonical keys load strictly before attaching the new branch. The bridge preserves post-build CPU RNG so parent dropout/auxiliary sampling can remain matched. Natural existence BCE supervises the combined logit; gradients flow through both saliency weighting and encoder evidence. GT, edits and rolled queries are used only for training auxiliary saliency losses, never for inference pooling.

| Arm | Evidence pooling | coarse/fine | rotated BCE/pair auxiliary |
|---|---|---|---|
| Local_CF | saliency weighted | 1/1 | 0/0 |
| Uniform_CF | uniform valid-token mean | 1/1 | 0/0 |
| Local_noCF | saliency weighted | 0/0 | 0/0 |

All three have identical new-head parameter counts and initialization, and perform identical natural/rotated/edit forwards. Uniform_CF controls the extra capacity and direct encoder path; Local_noCF controls whether CF supervision adds value. Frozen v2 B0 and S1 serve as historical10epoch references only after exposure/code/data validation. No optional new weights or warmup chosen from test.

A1/C1, canonical Moment-DETR-GMR initialization per split, seed3407, 10 epochs, lr1e-5, batch16, original wd/grad_clip. Best trained epoch by Seen-val AUROC, earliest tie; epoch0 diagnostic only. No formal test accessed by training or summary. Three independent jobs per GPU, A1/GPU0, C1/GPU1, stopped-on-failure diagnostics while other healthy children finish. Do not interrupt parent tasks. Newly unused GPU slots are not filled with invented experiments.

Save curves, natural criterion components, CF losses, raw/gated localization, conditions, score distributions, first-batch shared gradients at epochs1/3/10, evidence entropy, residual magnitude and learned head norm, all per-epoch Seen predictions with original-head and local-residual logits, checkpoints at1/3/10 and selected best. Require exposure match with all new arms and v2 controls. Record parent input and selected reference hashes; model/code copies are physical isolated copies. Local copy adds video_memory output but leaves native span and saliency mathematics unchanged.

Success checks for this first phase: no Seen collapse; nonzero learned branch contribution; Local_CF compared with capacity control and noCF control. Positive Seen results alone do not prove Unseen generalization or attention faithfulness. Formal A1/C1 evaluation remains exploratory and is not auto-scheduled; 50epoch validation and stop-gradient attention ablation are possible followups, not already authorized experiments.
