# Idea 06 pilot results

One seed (3407), two strict-inner folds, five epochs, fixed normalized log-mean-exp bag score (temperature 1, K=10). Inner Novel-dev is exploratory; official U was not accessed. The screen compares bag BCE (A0) with identical bag BCE plus reviewed-negative candidate BCE (A2). Pooled AUROC uses video-cluster bootstrap; same-query pairs use connected query-video component resampling; same-video source pairs use video clusters. Shuffled-video scores retain original labels solely as input-dependence diagnostics.

## A1_action_01

Novel AUROC A0/A2: 0.6434/0.6346; A2−A0 -0.87 pp, paired video-cluster 95% CI [-2.968374412831451, 1.1684870835685945].
Same-query PairAcc A0/A2: {'A0_bag_BCE': 0.38202247191011235, 'A2_verified_negative_local': 0.449438202247191} over 178 pairs; paired component CI [0.0, 0.07171314741035857].
Same-video PairAcc A0/A2: {'A0_bag_BCE': 0.6993006993006993, 'A2_verified_negative_local': 0.6993006993006993} over 143 reviewed source-linked pairs; paired video CI [0.0, 0.0].
Max aggregation AUROC {'A0_bag_BCE': 0.6374774774774775, 'A2_verified_negative_local': 0.6346666666666667}; query-only {'A0_bag_BCE': 0.6714954954954955, 'A2_verified_negative_local': 0.6714954954954955}; video-only {'A0_bag_BCE': 0.5004324324324325, 'A2_verified_negative_local': 0.5004324324324325}; baseline {'A0_bag_BCE': 0.616036036036036, 'A2_verified_negative_local': 0.616036036036036}; shuffled retained-label {'A0_bag_BCE': 0.6272432432432433, 'A2_verified_negative_local': 0.6227747747747747}.
Seen AUROC {'A0_bag_BCE': 0.8864574833716587, 'A2_verified_negative_local': 0.8768142818287465}; frozen localization R@1 IoU .5 {'A0_bag_BCE': 0.34841628959276016, 'A2_verified_negative_local': 0.34841628959276016}; parameters {'A0_bag_BCE': 232449, 'A2_verified_negative_local': 232449}; training seconds {'A0_bag_BCE': 9.546722888946533, 'A2_verified_negative_local': 14.78201961517334}; evaluation seconds {'A0_bag_BCE': 7.922260046005249, 'A2_verified_negative_local': 7.956609010696411}; shuffle seconds {'A0_bag_BCE': 1.5444841384887695, 'A2_verified_negative_local': 1.5652036666870117}.

## C1_composition_01

Novel AUROC A0/A2: 0.5492/0.5490; A2−A0 -0.02 pp, paired video-cluster 95% CI [-0.9259379509379547, 0.8822009520539711].
Same-query PairAcc A0/A2: {'A0_bag_BCE': 0.5070422535211268, 'A2_verified_negative_local': 0.49765258215962443} over 213 pairs; paired component CI [-0.21428571428571427, 0.0].
Same-video PairAcc A0/A2: {'A0_bag_BCE': 0.6710526315789473, 'A2_verified_negative_local': 0.6710526315789473} over 76 reviewed source-linked pairs; paired video CI [0.0, 0.0].
Max aggregation AUROC {'A0_bag_BCE': 0.5498231132075472, 'A2_verified_negative_local': 0.5479559748427674}; query-only {'A0_bag_BCE': 0.5829402515723271, 'A2_verified_negative_local': 0.5829402515723271}; video-only {'A0_bag_BCE': 0.49415290880503143, 'A2_verified_negative_local': 0.49415290880503143}; baseline {'A0_bag_BCE': 0.5553262578616353, 'A2_verified_negative_local': 0.5553262578616353}; shuffled retained-label {'A0_bag_BCE': 0.5433372641509434, 'A2_verified_negative_local': 0.5354756289308176}.
Seen AUROC {'A0_bag_BCE': 0.804986791934115, 'A2_verified_negative_local': 0.8093078017236228}; frozen localization R@1 IoU .5 {'A0_bag_BCE': 0.34985835694050993, 'A2_verified_negative_local': 0.34985835694050993}; parameters {'A0_bag_BCE': 232449, 'A2_verified_negative_local': 232449}; training seconds {'A0_bag_BCE': 16.78182554244995, 'A2_verified_negative_local': 19.702803373336792}; evaluation seconds {'A0_bag_BCE': 8.89449667930603, 'A2_verified_negative_local': 8.83785343170166}; shuffle seconds {'A0_bag_BCE': 1.0568606853485107, 'A2_verified_negative_local': 0.9759073257446289}.

## Disposition

**Insufficient evidence.** Mean A2−A0 Novel AUROC change was −0.45 pp across two folds; both paired video-cluster intervals include zero. A1 same-query PairAcc rose, but C1 did not replicate it; query-only AUROC exceeded the candidate scorer in both folds. The local candidate penalty is not promoted. Official U was not accessed.
