# Idea 05 pilot results

One seed (3407), two strict-inner folds, five epochs, natural BCE. Novel-dev is exploratory; official U was not accessed. Natural pooled AUROC and conditional metrics are shown together. Bootstrap units preserve video clusters for AUROC, connected query-video components for same-query pairs, and video clusters for source-linked pairs. Shuffled-video results retain labels and are input-dependence diagnostics only.

## A1_action_01

Natural Novel AUROC strict-inner baseline / pooled-arm / raw-sequence: 0.6160 / 0.6707 / 0.6335; raw−pooled -3.72 pp (paired video-cluster 95% CI [-8.97084622558031, 1.6672440174824155]).
Within-query / cross-query AUC (pooled arm): {'within_query_auc': 0.48314606741573035, 'within_query_pairs': 178, 'cross_query_auc': 0.6718772667923981, 'cross_query_pairs': 27572}; raw sequence: {'within_query_auc': 0.6179775280898876, 'within_query_pairs': 178, 'cross_query_auc': 0.6336138111127231, 'cross_query_pairs': 27572}.
Same-query PairAcc baseline / pooled / raw: 0.5112359550561798 / 0.483 / 0.618 over 178 pairs; paired component-bootstrap delta CI [0.10946745562130178, 0.6666666666666666].
Same-video PairAcc baseline / pooled / raw: 0.7062937062937062 / 0.7272727272727273 / 0.6433566433566433 over 143 trusted source-linked pairs; paired video-bootstrap delta CI [-0.14189298169136877, -0.026841163310961993].
Query-only AUROC {'P_pooled': 0.6478558558558558, 'R_raw_sequence': 0.6478558558558558}; video-only AUROC {'P_pooled': 0.5061621621621621, 'R_raw_sequence': 0.5061621621621621}; pre-fusion shuffled retained-label AUROC {'P_pooled': 0.6563243243243242, 'R_raw_sequence': 0.5922162162162162}.
Seen AUROC pooled/raw {'P_pooled': 0.8901169547264052, 'R_raw_sequence': 0.8826249289122964}; frozen localization R@1 IoU .5 {'P_pooled': 0.34841628959276016, 'R_raw_sequence': 0.34841628959276016}; parameters {'P_pooled': 33665, 'R_raw_sequence': 75649}; training seconds {'P_pooled': 32.85102367401123, 'R_raw_sequence': 51.38242268562317}; evaluation seconds {'P_pooled': 22.588478565216064, 'R_raw_sequence': 19.363860368728638}; shuffle seconds {'P_pooled': 1.342846393585205, 'R_raw_sequence': 0.20199275016784668}.

## C1_composition_01

Natural Novel AUROC strict-inner baseline / pooled-arm / raw-sequence: 0.5553 / 0.5529 / 0.5925; raw−pooled +3.96 pp (paired video-cluster 95% CI [-3.9968979916484426, 11.637432326176556]).
Within-query / cross-query AUC (pooled arm): {'within_query_auc': 0.5164319248826291, 'within_query_pairs': 213, 'cross_query_auc': 0.5536484994479575, 'cross_query_pairs': 9963}; raw sequence: {'within_query_auc': 0.5492957746478874, 'within_query_pairs': 213, 'cross_query_auc': 0.5933955635852655, 'cross_query_pairs': 9963}.
Same-query PairAcc baseline / pooled / raw: 0.5258215962441315 / 0.516 / 0.549 over 213 pairs; paired component-bootstrap delta CI [-0.35294117647058826, 0.04950495049504951].
Same-video PairAcc baseline / pooled / raw: 0.6578947368421053 / 0.6710526315789473 / 0.6710526315789473 over 76 trusted source-linked pairs; paired video-bootstrap delta CI [0.0, 0.0].
Query-only AUROC {'P_pooled': 0.5718356918238994, 'R_raw_sequence': 0.5718356918238994}; video-only AUROC {'P_pooled': 0.5137087264150944, 'R_raw_sequence': 0.5137087264150944}; pre-fusion shuffled retained-label AUROC {'P_pooled': 0.5379323899371069, 'R_raw_sequence': 0.5742924528301886}.
Seen AUROC pooled/raw {'P_pooled': 0.8021180717419109, 'R_raw_sequence': 0.8362498655287409}; frozen localization R@1 IoU .5 {'P_pooled': 0.34985835694050993, 'R_raw_sequence': 0.34985835694050993}; parameters {'P_pooled': 33665, 'R_raw_sequence': 75649}; training seconds {'P_pooled': 53.1603057384491, 'R_raw_sequence': 69.22727990150452}; evaluation seconds {'P_pooled': 33.39481067657471, 'R_raw_sequence': 28.389227151870728}; shuffle seconds {'P_pooled': 0.9271914958953857, 'R_raw_sequence': 0.11678528785705566}.

## Disposition

`insufficient_evidence`, with a within-query mechanism clue on the action fold. Mean raw-minus-pooled Novel AUROC is +0.12 pp, effects reverse across folds, and the paired video-cluster intervals include zero. On A1, same-query PairAcc improves while cross-query AUC and same-video PairAcc fall; the pooled objective remains unresolved. The raw adapter also has more than twice the pooled arm's parameters. Do not treat this as a semantic-novel solution or cross-backbone claim. Formal U remains reserved for a later frozen candidate.
