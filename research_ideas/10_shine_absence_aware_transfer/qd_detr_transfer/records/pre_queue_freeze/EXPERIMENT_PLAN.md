# QD-DETR-GMR + SHINE coarse/fine, A1/C1 50 epochs

Authorized explicitly in this side conversation: QD-DETR, SHINE coarse/fine saliency losses, 50 epochs. Independently rooted under qd_detr_transfer; no parent Moment code, run, freeze or record modifications. Source QD model/training code physically copied from current generalized-moment-retrieval; strict canonical checkpoint loading.

A1 and C1, seed3407 only, 50 total new adaptation epochs from canonical QD checkpoints. Original checkpoint architecture and GMR criterion retained; original lw_saliency=0 and mr_only=True. Constant adaptation lr1e-5, batch16, original wd1e-4, clip.1. No scheduler beyond the constant-lr adaptation design; no early stop. Select earliest best trained Seen-val AUROC, exclude epoch0 fallback. Formal train retains all natural positive/negative rows. Prepared Moment formal Seen/train partitions and held-semantic-filtered edit banks reused as physical JSON copies; CLIP edit arrays and original video/text feature arrays read only and hash frozen.

| Arm | objective |
|---|---|
| B0 | original QD-DETR-GMR loss |
| S1_saliency_only | original loss + 1 coarse + 1 fine |

No added rotated existence BCE or existence pair loss. Coarse: margins1/2, q8; fine red: four margins.25, upstream one-sided stop-gradient distance, maximum4 positive chains per batch. Original QD global-token saliency projections are used; no model/head replacement. Explicit rotated and edit-query forwards occur before fusion, matching the Moment transfer mechanism. QD native internal training-negative transformer forwards are retained on all model calls in both arms. Thus B0 matches auxiliary-forward/RNG exposure; it is not claimed to equal historical no-aux baseline trajectory.

Per-epoch natural qid/edit indices exposure hash; natural GMR components, coarse/fine, score distributions and Seen predictions; diagnostics and checkpoints at epochs1/3/10/25/50 plus best. Gradient probes in train mode, first batch, shared transformer/input projections, without optimizer updates.

GPU0: A1 B0 + S1 concurrently. GPU1: C1 B0 + S1 concurrently. Start only after integration smoke and freeze; no existing main-thread training processes were running at inspection. Detached independent workers. After both arms on a split finish all50, evaluator verifies selected checkpoint hashes, selection and all50 exposure hashes, freezes before test label load, and computes canonical/B0/S1 formal Seen/Unseen, gap and relative gap reduction, paired video-cluster intervals, conditionals, raw/gated localization and fresh pre-fusion shuffled-video. Thresholds only from Seen-val. Formal evaluation is exploratory because benchmark/test has already been inspected in this research workflow. No U-based checkpoint/parameter search.

Both splits completed: descriptive two-split macro auto-generated; do not assume independent splits or report this as five-split result. Success requires Unseen improvement with Seen preservation; gap shrinkage from Seen damage alone is not success. No multiseed, no Flash transfer, no other parent experiments started or altered by this side conversation.
