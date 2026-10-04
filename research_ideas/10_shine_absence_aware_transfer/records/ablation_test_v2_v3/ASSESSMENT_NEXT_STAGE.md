# Assessment and next stage

All selected checkpoints and sixteen prediction files passed hash checks. Evaluation: A1/C1, canonical + seven 10-epoch arms, seed3407, Seen-val-only selection/threshold. No result-based checkpoint change. This is exploratory test evaluation, not an untouched confirmation.

Saliency-only (S1) vs B0: A1 ΔUnseen −.40pp, paired video-cluster 95% CI [−.88,+.09]pp; C1 +1.16pp, CI [+0.0025,+2.31]pp. C1 lower bound is extremely close to zero; intervals are conditional on one trained seed and exploratory multiple comparisons. Test Seen: A1 approximately unchanged, C1 +.52pp. Two-split macro ΔUnseen +.38pp/ΔSeen +.26pp, descriptive and not a five-split result. No robust across-split mitigation claim.

Pair-only raises A1 U by +1.45pp (CI [+.11,+2.81]pp) while test Seen drops −1.87pp; C1 U −.30pp and Seen −1.97pp. Macro pair-only ΔU +.57pp/ΔSeen −1.92pp. Weak BCE.1+pair.2 macro ΔU +.51pp/ΔSeen −2.49pp. The Unseen changes do not rescue substantial Seen damage. BCE-only and full losses similarly damage Seen.

Mechanism limitations: S1 U source-pair accuracy decreases on both splits, while exact-query ranking improves; raw U localization improves on A1 and declines on C1. S1 fresh shuffled-video U is A1 .4814 vs natural .4805, C1 .5853 vs natural .5896. These mixed conditionals and dependence checks do not support a strong event-evidence mechanism claim.

Next stage: v4 already launched with effective weights .01/0, 0/.02, .01/.02 for BCE/pair, retaining GMR+coarse/fine. These arms/weights were fixed from v3 Seen findings in ABLATION_V4_PLAN before the new test evaluation, and remain unchanged after these outcomes. Three trainers per GPU, A1/GPU0 and C1/GPU1, 10 epochs/seed3407, Seen-only. Determine whether reducing both auxiliary strengths preserves primary-task ranking, before considering warmup or longer validation. No new U-based coefficient/epoch search.
