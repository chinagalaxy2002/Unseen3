# Experiment Plan: Moment-DETR-TRM / TRM-PT Localization Generalization

This document defines the execution strategy, file architecture, hyperparameter freeze, and evaluation plan for Phase 1 of the TRM generalization investigation.

---

## 1. Core Research Objective

> **Does phrase-level temporal relationship mining (TRM / TRM-PT) improve unseen semantic moment localization ($U^+$) without sacrificing seen performance ($S^+$) in Moment-DETR?**

### Phase 1 Guardrails:
- **Task**: Localization-only.
- **Data**: $S^+$ only for training; Seen validation ($S^+$) only for model selection.
- **Forbidden**: No $S^-$, no $U^-$, no existence heads, no rejection thresholds, no $U^+$ peeking.

---

## 2. File Modification & Creation Manifest

All changes are strictly confined to `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3`.

### 2.1 Preserved Baseline (Isolated & Untouched)
- `models/moment_detr_gmr/`: Preserved for regression testing and comparative runs.
- `training/moment_detr_gmr/`: Original baseline training scripts.

### 2.2 New Moment-DETR-TRM Modules
- `models/moment_detr_trm/`:
  - `__init__.py`
  - `moment_detr_trm.py`: Implements MomentDETR-TRM with phrase projection, attentive pooling, phrase-slot matching, and refined logits.
  - `trm_modules.py`: AttentivePooling and phrase-to-slot similarity heads.
  - `trm_loss.py`: Consistency loss, negative contrastive loss, exclusiveness loss, and proposal IoU matching.
- `training/moment_detr_trm/`:
  - `__init__.py`
  - `dataset_trm.py`: Dataset loader providing both full-query CLIP features and phrase token features.
  - `train_trm.py`: Main 100-epoch training entry point for Moment-DETR-TRM.
  - `evaluate_trm.py`: Standalone evaluation supporting both $S^+$ and $U^+$ test splits.
- `scripts/`:
  - `extract_phrases.py`: Offline phrase extraction combining TRM official annotations with SRL/POS parser.
  - `extract_phrase_clip_features.py`: CLIP feature extraction for cached phrases using ViT-B/32.
  - `smoke_test_trm.py`: Comprehensive 7-step smoke test suite.
  - `train_trm.sh`: Training shell runner per split.
  - `infer_trm.sh`: Evaluation shell runner per split.

### 2.3 TRM-PT Reproduction Module (Isolated)
- `models/moment_detr_trm_pt/`:
  - `moment_detr_trm_pt.py`: Phrase pseudo-temporal supervision module.

---

## 3. Frozen Hyperparameter Specification

| Parameter | Frozen Value | Source / Rationale |
| :--- | :--- | :--- |
| `seed` | `3407` | Baseline standard across GMR splits |
| `n_epoch` | `100` | Forced 100 epochs, matching baseline protocol |
| `max_es_cnt` | `-1` | Early stopping disabled |
| `bsz` | `16` | Standard Moment-DETR batch size |
| `eval_bsz` | `16` | Evaluation batch size |
| `lr` | `1e-4` | Standard AdamW learning rate |
| `wd` | `1e-4` | Weight decay |
| `lr_drop` | `400` | Constant LR over 100 epochs |
| `hidden_dim` | `256` | Moment-DETR hidden dim |
| `num_queries` | `10` | Moment-DETR proposal slots |
| `max_phrases` ($P$) | `10` | Hardcoded in official TRM source |
| `phrase_drop_prob` | `0.1` | Keep prob = 0.9, matching official TRM |
| `iou_thresh` ($\theta$) | `0.1` | Matching Charades config in official TRM |
| `lambda_refine` ($w$) | `1.0` | `RESIDUAL: 1.0` in official TRM config |
| `lambda_con` | `1.0` | `CONSIS_WEIGHT: 1.0` in official TRM config |
| `lambda_neg` | `0.5` | `trainer.py` negative loss scaling (0.5) |
| `lambda_exc` | `1.0` | Pre-fixed full mechanism weight |

---

## 4. Evaluation Matrix (5 Splits)

For each split (`A1`, `A2_alt`, `A3`, `C1`, `C2_alt`), report:
- **Seen Positive ($S^+$)**: R@1@0.3, R@1@0.5, R@1@0.7, mIoU.
- **Unseen Positive ($U^+$)**: R@1@0.3, R@1@0.5, R@1@0.7, mIoU.
- **Gap & Degradation Analysis**:
  - $\text{Gap}_{\text{R1@0.5}} = S^+ - U^+$
  - $\text{Gap}_{\text{mIoU}} = S^+ - U^+$
  - $\Delta_{\text{unseen}} = \text{TRM}_{U^+} - \text{Baseline}_{U^+}$
