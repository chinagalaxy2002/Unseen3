# Baseline Audit: Moment-DETR and Grounding Benchmark

This document provides a comprehensive audit of the baseline codebase and training pipeline in accordance with Section 2 of `plan.md`.

---

## 1. Source Environment and Git State

- **Primary Source Path (ReadOnly)**: `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen`
- **Git Commit**: `ddc78aa910a29445c162512fd45cbedc61e2f2cf`
- **Git Branch**: `main` (clean working directory, tracking `origin/main`)
- **Git Log (-5)**:
  - `ddc78aa` docs: Remove EVALUATION_REPORT.md to keep repository purely factual and data-driven
  - `6f71904` docs: Update README with factual experimental results and add raw metrics JSONs for all 5 splits
  - `f9e49de` docs: Add full 5-split evaluation report and diagnostic comparison
  - `9aa398e` feat: Add dynamic dual-GPU worker queue for A2_alt, A3, and C2_alt
  - `6574dad` docs: publish A1 and C1 experiment logs and results
- **Canonical Moment-DETR Implementation Reference**:
  - `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen2` (commit `22418be7bb2b781190bc4dff8a8ff8ce901bc09a`)
  - `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval` (commit `bc88348e0e6b7b629a77d884fe66efe8c87b2774`)
  - *Note*: As verified during the source audit, `Unseen` hosts the benchmark dataset, features, evaluation harness, and `flash_vtg_gmr`, whereas `moment_detr_gmr` was developed and maintained in the upstream `generalized-moment-retrieval` repository and mirrored in `Unseen2`. In `Unseen3`, we bring together the complete Moment-DETR implementation with the benchmark dataset and the TRM migration.

---

## 2. Model Architecture & Tensor Interface Checklist

| Item | Audited Implementation Specification | Source Code Evidence |
| :--- | :--- | :--- |
| **`src_txt` shape** | `[B, L_txt, 512]`, float32 | `models/moment_detr_gmr/moment_detr.py:76`, `dataset.py:230-234` |
| **`src_txt_mask` shape** | `[B, L_txt]`, float/bool (1 = valid token, 0 = pad) | `models/moment_detr_gmr/moment_detr.py:77-78`, `dataset.py:265-270` |
| **`src_vid` shape** | `[B, L_vid, 2818]` (SlowFast 2304 + CLIP 512 + TEF 2) | `models/moment_detr_gmr/moment_detr.py:79`, `dataset.py:160-167, 247` |
| **`src_vid_mask` shape** | `[B, L_vid]`, float/bool (1 = valid frame, 0 = pad) | `models/moment_detr_gmr/moment_detr.py:80`, `dataset.py:265-270` |
| **`hidden_dim`** | 256 | `configs/moment_detr_gmr/base.yml:24`, `moment_detr.py:41` |
| **`num_queries`** | 10 decoder query slots | `configs/moment_detr_gmr/base.yml:28`, `moment_detr.py:49` |
| **`decoder hs` shape** | `[num_layers, B, num_queries, hidden_dim]` (e.g. `[2, B, 10, 256]`) | `moment_transformer.py:65`, `moment_detr.py:101-106` |
| **`pred_logits` indexing** | 2 classes: **Index 0 = Foreground**, **Index 1 = Background (no-object)** | `moment_detr.py:46, 149-153` (`foreground_label=0, background_label=1`) |
| **`pred_spans` format** | `[B, num_queries, 2]` normalized `(cx, w)` in `[0, 1]` | `moment_detr.py:84-87, 103-106` (`outputs_coord.sigmoid()`) |
| **Matcher positive handling** | Hungarian bipartite matching via Scipy `linear_sum_assignment`. Cost = $4 \times C_{\text{class}} + 10 \times C_{\text{L1}} + 1 \times C_{\text{GIoU}}$. $C_{\text{class}} = -\text{softmax}(logits)[:, 0]$. | `matcher.py:64-100` |
| **Empty GT handling** | When `targets["spans"]` is empty: returns empty index tuples; `loss_span` and `loss_giou` return `outputs["pred_spans"].sum() * 0.0` (finite 0, preserves graph). | `moment_detr.py:165-167`, `matcher.py:59-62` |
| **CLIP text NPZ format** | Compressed `.npz` containing key `'last_hidden_state'`, shape `[seq_len, 512]`. | `dataset.py:230-234`, verified via `prepare_charades_semantic_existence.py:60` |
| **Query token feature type** | `last_hidden_state` from CLIP ViT-B/32 text transformer. | `configs/moment_detr_gmr/feature/clip_slowfast.yml`, `dataset.py:230` |
| **Text feature normalization** | Token-wise L2 normalization via `l2_normalize_np_array`. | `dataset.py:233`, `models/moment_detr_gmr/utils/basic_utils.py` |
| **Video feature sources** | `features/charades_video/vid_slowfast` (2304) + `features/charades_video/vid_clip` (512) | Symlinked to `/home/guoxiangyu/paper/新建文件夹/charades` |
| **Train/val/test split paths** | `data/release/semantic_existence_v2/{split}/{train,val,test}.jsonl` | Symlinked to `/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2` |
| **Localization-only scripts** | `scripts/run_semantic_localization_controls.sh` extracts $S^+$ for train and val, and evaluates on $S^+$ and $U^+$ without existence head. | `run_semantic_localization_controls.sh:23-33, 76-96` |
| **Checkpoint selection rule** | Best checkpoint strictly selected by Seen Validation `MR-full-mAP` (`val_seen.jsonl` or `val_positive.jsonl`). | `train.py:118-124` |
| **Formal baseline training parameters** | Seed: `3407`, Epochs: `100`, Batch Size: `16`, Optimizer: `AdamW`, LR: `1e-4`, Weight Decay: `1e-4`, LR Drop: `400` (constant LR), Max Early Stop: `-1` (forced 100 epochs). | `run_semantic_localization_controls.sh:76-85`, `EXPERIMENT_PLAN.md` |

---

## 3. Data Partitions and Benchmark Protocol

- **Dataset**: `Charades-STA` derived Semantic Novelty $\times$ Event Existence (`semantic_existence_v2`).
- **Splits**:
  - `A1`: Action holdout (`put` / `take`)
  - `A2_alt`: Action holdout (`drink` / `pour`)
  - `A3`: Action holdout (`run` / `walk`)
  - `C1`: Composition holdout (`sit | bed/chair/couch`)
  - `C2_alt`: Composition holdout (`open/close | box/cabinet`)
- **Localization-Only Phase 1 Protocol**:
  - **Train**: Strictly $S^+$ (seen semantics with ground-truth temporal windows).
  - **Validation**: Strictly Seen validation ($S^+$).
  - **Test**: Separate evaluation on $S^+$ and $U^+$.
  - **Forbidden**: No $S^-$, no $U^-$, no existence heads, no existence BCE, no gate thresholds, no model selection on $U^+$.
