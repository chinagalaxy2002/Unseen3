# Idea 04 pilot results

One seed, two strict inner folds. Natural pooled and conditional metrics are shown together. Bootstrap resamples videos for pooled AUROC and connected query-video components for same-query PairAcc. Shuffled-input AUROC retains original labels and is only a dependence diagnostic.

## A1_action_01

Novel AUROC M0/M1/M2: 0.6169/0.6229/0.6472; M2−M0 paired video-cluster 95% CI [+0.32,+5.88] pp.
Same-query PairAcc M0/M1/M2: 0.494/0.455/0.455; M2−M0 connected-component 95% CI [-0.054,+0.286].
Within-query / cross-query AUC: M0 0.494/0.618; M1 0.455/0.624; M2 0.455/0.648. Same-video PairAcc M0/M1/M2: 0.713/0.713/0.713; video-cluster 95% CI for M2 [0.604,0.808].
Query-only AUROC 0.671; video-only 0.500; empirical query-only same-query PairAcc 0.534 is cache-sensitive and must not be read as compatibility. Shuffled retained-label AUROC M0/M1/M2: 0.603/0.604/0.622.
Seen AUROC M0/M1/M2 0.8726/0.8731/0.8554; frozen localization R@1 IoU .5 0.348; 50,049 parameters; training seconds M0/M1/M2 5.5/7.1/6.9; per-arm reforward shuffle seconds 4.5/4.5/4.6.

## C1_composition_01

Novel AUROC M0/M1/M2: 0.5471/0.5482/0.5498; M2−M0 paired video-cluster 95% CI [-0.36,+1.05] pp.
Same-query PairAcc M0/M1/M2: 0.502/0.516/0.521; M2−M0 connected-component 95% CI [+0.000,+0.286].
Within-query / cross-query AUC: M0 0.502/0.548; M1 0.516/0.549; M2 0.521/0.550. Same-video PairAcc M0/M1/M2: 0.671/0.658/0.658; video-cluster 95% CI for M2 [0.524,0.784].
Query-only AUROC 0.583; video-only 0.494; empirical query-only same-query PairAcc 0.883 is cache-sensitive and must not be read as compatibility. Shuffled retained-label AUROC M0/M1/M2: 0.540/0.533/0.534.
Seen AUROC M0/M1/M2 0.8093/0.8084/0.8066; frozen localization R@1 IoU .5 0.350; 50,049 parameters; training seconds M0/M1/M2 7.4/9.8/9.9; per-arm reforward shuffle seconds 5.1/5.2/5.1.

## Decision

Exploratory only. Disposition remains insufficient_evidence unless pooled gain reaches the frozen screen and video-conditional metrics support compatibility rather than query-side gain; a single seed and two folds cannot confirm it.
