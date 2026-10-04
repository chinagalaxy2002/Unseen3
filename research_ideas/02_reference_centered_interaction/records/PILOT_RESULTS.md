# Idea 02 pilot results

Exploratory one-seed, two-fold inner Novel-dev screen. Official U was not accessed. Same-video pairing intervals are clustered by video; same-query pairing intervals resample connected query-video components. Shuffled-video scores retain original labels and are dependence diagnostics only.

## A1_action_01

Novel pooled AUROC B0 raw 0.6798, B1 same-checkpoint post-hoc 0.6778, B2 centered-trained 0.6811.
Same-query PairAcc: B0 0.708, B1 0.702, B2 0.753; B2−B0 component-bootstrap 95% [+0.000,+0.274].
Positive-negative AUC decomposition (within-query / cross-query): B0 0.708/0.680; B1 0.702/0.678; B2 0.753/0.681 (178 within and 27572 cross pairs).
Same-video PairAcc: B0 0.734, B1 0.741, B2 0.727. Query-only AUROC 0.637, video-only 0.502. Shuffled retained-label AUROC: B0 0.540, B1 0.544, B2 0.541.
Seen AUROC B0/B1/B2: 0.6774/0.6815/0.6849; frozen Seen localization R@1 IoU .5 0.348; 106,497 parameters; train seconds raw/centered 0.6/0.6.

## C1_composition_01

Novel pooled AUROC B0 raw 0.5427, B1 same-checkpoint post-hoc 0.5432, B2 centered-trained 0.5478.
Same-query PairAcc: B0 0.493, B1 0.512, B2 0.535; B2−B0 component-bootstrap 95% [+0.000,+0.045].
Positive-negative AUC decomposition (within-query / cross-query): B0 0.493/0.544; B1 0.512/0.544; B2 0.535/0.548 (213 within and 9963 cross pairs).
Same-video PairAcc: B0 0.553, B1 0.553, B2 0.539. Query-only AUROC 0.559, video-only 0.523. Shuffled retained-label AUROC: B0 0.525, B1 0.526, B2 0.522.
Seen AUROC B0/B1/B2: 0.6128/0.6140/0.6176; frozen Seen localization R@1 IoU .5 0.350; 106,497 parameters; train seconds raw/centered 0.8/0.9.

## Decision

Disposition remains insufficient_evidence for a cross-semantic method claim: one fold can show a conditional gain while the composition fold may not; report paired intervals and do not use this one seed to select a final method. The result tests a rank-32 masked-mean bilinear scorer and cannot establish the same mechanism for canonical GMR backbones.
