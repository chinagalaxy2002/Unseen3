# A1 component ablation: Seen-only development

seed3407; 10 epochs; matched auxiliary forwards; no formal test read. Epoch0 is a reference, excluded from checkpoint selection.

| Arm | Epoch0 | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δbest vs B0 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8367 | 0.8397 | 0.8423 | 0.8439 | 5 / 0.8456 | +0.00 |
| S1_saliency_only | 0.8367 | 0.8412 | 0.8454 | 0.8452 | 5 / 0.8473 | +0.17 |
| E1_rotated_exist_only | 0.8367 | 0.7738 | 0.7744 | 0.7826 | 7 / 0.7852 | -6.04 |
| SE1_full | 0.8367 | 0.7743 | 0.7810 | 0.7715 | 3 / 0.7810 | -6.46 |

These results diagnose Seen damage; they do not establish Unseen improvement. Gradient files describe local pre-clipping gradients.
