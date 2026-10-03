# Joint-v3 interim training status

Snapshot UTC: 2026-10-03T07:12:01.192020+00:00

Joint-v3 is an exploratory follow-up motivated by inspected Joint-v1/v2 U results. All values below use Seen validation only. No Joint-v3 U result is available for these two splits.

| Split | Training status | Epochs completed | mAP stale epochs | Highest Seen mAP | Epoch at highest mAP | Frozen mAP floor | Eligible checkpoint | U test run |
|---|---|---:|---:|---:|---:|---:|---|---|
| A1 | completed | 40 | 30 | 22.64 | 10 | 23.70 | False | False |
| A2_alt | running | 53 | 17 | 22.38 | 36 | 25.49 | False | False |

A1 stopped after 30 consecutive epochs without Seen-val MR-full-mAP improvement. Its highest mAP (22.64) is below the frozen Joint-v1 reference minus 1pp floor (23.70). This is a localization-constraint failure; it has no selected best checkpoint and no U evaluation. The constraint was not relaxed.

A2_alt continues in the automatic GPU1 queue. It will be evaluated only if an eligible checkpoint exists at completion; otherwise, the queue records the same selection-failure status. GPU0 has moved on to A3. Later jobs remain A3 → C2_alt on GPU0 and C1 after A2_alt on GPU1.

No NaN/Inf or missing-feature fallback was reported in the recorded training metadata. Validation uses full-precision logits for conditional AUROC. Detailed A1 Seen-only diagnostics are in `A1_training_summary.json`; A2_alt values are time-stamped progress, not final results.
