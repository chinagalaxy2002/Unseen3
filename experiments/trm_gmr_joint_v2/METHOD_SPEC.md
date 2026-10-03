# Joint-v2 method and protocol

## Inherited model path

Moment-DETR localization and transformer are unchanged. Joint-v1's phrase-slot matcher, candidate-conditioned phrase attention, candidate-wise gate and refinement, semantic evidence head, smooth logsumexp pooling, TRM phrase losses, empty-GT S− behavior, and official localization postprocessing are retained. The semantic candidate score remains the existing MLP over `[h_j, base_margin_j, refined_margin_j, phrase_support_j, phrase_gate_j]`; its pooled value is `pred_exist_logits_semantic`.

## Visual evidence and fusion

The dataset's per-modality normalized SlowFast+CLIP visual input is exposed before TEF concatenation. Background PCA is independently fitted per split from S+ training clips outside GT windows; the fixed rank is `min(256,D-1)`. Candidate spans are detached before temporal pooling. A candidate's mean visual feature is centered by `mu_bg`, projected onto `components_bg`, and the orthogonal residual norm is standardized by the train-background residual mean and standard deviation. `tanh` gives bounded `candidate_visual_support`. This branch reads no text/query, phrase, slot, semantic score, or label. Fusion adds `softplus(alpha_raw) * candidate_visual_support.detach()` to semantic evidence, with alpha initialized to 1.0; logsumexp pooling produces the final fused logit.

## Objective

The Joint-v1 objective is unchanged, including BCE. Joint-v2 adds `loss_auc_pairwise` with margin 1.0 and coefficient 0.5. It averages `softplus(1 - (positive_logit - negative_logit))` over all within-batch positive-negative pairs. If either class is absent, the output is a differentiable zero. This is a ranking surrogate for AUROC, not exact AUROC optimization. No external AUC library, balanced sampling, parameter sweep, or extra seed is used.

## Evaluation and reporting

Train on S+ and S− only. Fit PCA only on same-split train S+ background clips. Select `best.ckpt` by Seen validation MR-full-mAP. Calibrate threshold from Seen validation S+/S− using maximum balanced accuracy. Only after checkpoint and threshold are frozen, run S+, S−, U+, U− test inference. Canonical AUROC is based on full-precision fused and semantic logits; probability AUROC is cross-checked. Report semantic-only and fused AUROC, fusion deltas, Joint-v1 unseen AUROC deltas, threshold metrics, and raw localization metrics.

## Exploratory status

Joint-v2 is an exploratory follow-up motivated by Joint-v1 failure modes. The Joint-v1 U results were viewed before v2 was designed, so Joint-v2's results on these same U splits must not be described as untouched unseen evaluation. Nevertheless, U does not participate in this run's PCA, training, checkpoint selection, threshold calibration, or hyperparameter selection. The five-split architecture and all settings are frozen in `EXPERIMENT_FREEZE.json` before the first Joint-v2 U inference.
