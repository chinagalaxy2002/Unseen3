# GMR-AUC-v4 method specification

This is an exploratory follow-up motivated by already inspected Joint-v1/v2/v3 U results; it is not an untouched unseen evaluation. The experiment asks whether an adapter-only global pooled-AUC objective can reduce unseen AUROC degradation while preserving the original Moment-DETR-GMR anchor.

Each split starts from its canonical original Moment-DETR-GMR checkpoint. Every baseline parameter is frozen. The original existence head receives final decoder slot states `hs[-1]`, max-pools over the slot axis for these baselines, then applies its original linear-ReLU-linear head. The pooled vector is `z_base` and the exact replayed original logit is `s0`.

The residual adapter input is `concat(stopgrad(z_base), stopgrad(s0))`. Its fixed architecture is `LayerNorm(D+1) -> Linear(D+1,64) -> GELU -> Dropout(0.1) -> Linear(64,1)`. The last layer is initialized to zero. `delta = 2*tanh(raw_delta/2)` and `s = s0 + delta`, so initialization reproduces the base logit and `delta` is bounded in `[-2,2]`.

Training uses only complete S+/S− train feature banks. Each step combines four independently sampled streams: empirical BCE minibatch (shuffle traversal, batch 256), 128 uniform positives and 128 uniform negatives for global pairwise AUC, 256 positive-anchor same-semantic pairs using query/composition/action priority, and the squared anchor penalty. `L_global_auc = mean(softplus(1 - s_pos[:,None] + s_neg[None,:]))`; this is a differentiable pairwise ranking surrogate, not exact AUROC differentiation. Total loss is `1*BCE + 1*global_auc + 1*same_semantic + 0.1*anchor`.

AdamW trains adapter parameters only (lr 1e-3, wd 1e-4), seed 3407, for 50 epochs. Each epoch is `ceil(N_train/256)` steps and the BCE stream visits every Seen row once. Same-semantic pair pools are made from Seen train metadata only. Positives with no eligible negative remain in BCE and global AUC. U is excluded from all training and selection.

The selected adapter maximizes full-precision pooled Seen-validation AUROC. Epoch 0 is included, ties choose the earlier epoch, and a zero adapter can be the winner. Thresholds are computed after selection using Seen validation balanced accuracy only. Final canonical AUROC uses full-precision logits on common test qids. Paired stratified bootstrap uses 2000 resamples and seed 3407; it describes uncertainty and never drives selection.

The baseline model is evaluated once per query at final inference. V4 only changes existence logit. Raw spans and localization class logits are emitted from the same frozen forward pass and copied as identical baseline/V4 localization outputs; direct equality is asserted in the inference path. Any actual baseline parameter drift or localization change is an implementation failure.
