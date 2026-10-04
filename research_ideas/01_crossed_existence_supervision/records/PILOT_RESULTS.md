# Idea 01 pilot results

Exploratory one-seed, two-fold pilot. This does not test complete closed quartets. Formal U was not accessed.

## A1_action_01

Novel pooled AUROC: BCE 0.6277; BCE+pair 0.6157; Δ -1.20 pp (paired video-cluster bootstrap 95% [-2.83, +0.38] pp).
Same-query cross-video PairAcc: 0.449 → 0.427 (Δ -0.022, 178 pairs; connected query-video component bootstrap 95% [-0.024, +0.000]). Same-video PairAcc: 0.713 → 0.706 (paired video-cluster Δ interval not estimable from saved pair identities).
Query-only AUROC 0.671, query-only same-query PairAcc 0.534; video-only AUROC 0.500. Shuffled-input retained-label AUROC: BCE 0.628 → 0.613; pair 0.616 → 0.599. Shuffle is an input-dependence diagnostic only.
Seen AUROC 0.8719 → 0.8735; frozen raw Seen localization R@1 IoU .5 0.348; 50,049 head parameters; training time 9.5s / 9.3s.

## C1_composition_01

Novel pooled AUROC: BCE 0.5536; BCE+pair 0.5546; Δ +0.11 pp (paired video-cluster bootstrap 95% [-0.28, +0.48] pp).
Same-query cross-video PairAcc: 0.507 → 0.516 (Δ +0.009, 213 pairs; connected query-video component bootstrap 95% [+0.000, +0.010]). Same-video PairAcc: 0.671 → 0.671 (paired video-cluster Δ interval not estimable from saved pair identities).
Query-only AUROC 0.583, query-only same-query PairAcc 0.883; video-only AUROC 0.494. Shuffled-input retained-label AUROC: BCE 0.554 → 0.535; pair 0.555 → 0.535. Shuffle is an input-dependence diagnostic only.
Seen AUROC 0.8117 → 0.8112; frozen raw Seen localization R@1 IoU .5 0.350; 50,049 head parameters; training time 12.3s / 12.5s.

## Decision

insufficient_evidence for this simple pair fallback: action fold regressed on pooled AUROC and same-query PairAcc; composition fold was near zero. This does not refute complete-quartet supervision.
Proceed to priority 02 reference-centered interaction pilot; revisit Idea 01 only if a closed-quartet audit shows enough independent, trusted matrices.
