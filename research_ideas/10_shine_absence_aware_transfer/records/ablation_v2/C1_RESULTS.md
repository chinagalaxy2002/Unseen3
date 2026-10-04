# C1 component ablation: Seen-only development

seed3407; 10 epochs; matched auxiliary forwards; no formal test read. Epoch0 is a reference, excluded from checkpoint selection.

| Arm | Epoch0 | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δbest vs B0 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.7883 | 0.7826 | 0.7808 | 0.7796 | 5 / 0.7844 | +0.00 |
| S1_saliency_only | 0.7883 | 0.7901 | 0.7845 | 0.7846 | 1 / 0.7901 | +0.57 |
| E1_rotated_exist_only | 0.7883 | 0.7536 | 0.7360 | 0.6979 | 1 / 0.7536 | -3.07 |
| SE1_full | 0.7883 | 0.7599 | 0.7390 | 0.6764 | 1 / 0.7599 | -2.44 |

These results diagnose Seen damage; they do not establish Unseen improvement. Gradient files describe local pre-clipping gradients.
