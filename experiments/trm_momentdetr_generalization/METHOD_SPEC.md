# Method Specification: Moment-DETR-TRM & TRM-PT

This document provides the definitive specification of the **Moment-DETR-TRM** architecture, data flow, loss formulations, and the exact mapping from TRM to Moment-DETR, complying with Sections 8, 9, 10, 16, 17, and 24 of `plan.md`.

---

## 1. Structural Mapping Table: TRM → Moment-DETR

| TRM Original Mechanism `[TRM-REPO]` | Moment-DETR Corresponding Implementation `[MOMENT-DETR-ADAPTATION]` | Rationale / Mathematical Definition |
| :--- | :--- | :--- |
| **2D Temporal Proposal Map** $F_{\text{iou}}^V \in \mathbb{R}^{B \times C \times T \times T}$ | **Transformer Decoder Slots** $h \in \mathbb{R}^{B \times Q \times d}$ ($hs[-1]$, $Q=10, d=256$) | In Moment-DETR, the $Q$ decoder query representations directly represent temporal candidate moments/proposals. |
| **Phrase Text Encoding** DistilBert + Linear -> $f_i^P \in \mathbb{R}^{B \times P \times d}$ | **CLIP Token Projection + Masked Mean** `input_txt_proj` -> $p_i \in \mathbb{R}^{B \times P \times d}$ | Reuses the identical frozen CLIP text representation budget; tokens for each phrase are projected through the shared input projection and mean-pooled over valid tokens. |
| **Phrase-Proposal Cosine Score** $S_{ij}^P = \sigma\left(10 \cdot \frac{f_i^P \cdot v_j}{\|f_i^P\| \|v_j\|}\right)$ | **Phrase-Slot Cosine Matching** $S_{ij}^P = \sigma\left(10 \cdot \frac{\text{proj}_p(p_i) \cdot \text{proj}_s(h_j)}{\|\text{proj}_p(p_i)\| \|\text{proj}_s(h_j)\|}\right)$ | Computes fine-grained alignment between the $i$-th phrase ($i \in \{1,\dots,P\}$) and the $j$-th decoder proposal slot ($j \in \{1,\dots,Q\}$). |
| **Attentive Pooling (Phrase Importance)** $\alpha = \text{softmax}(W_2 \tanh(W_1 P_{\text{feat}}))$ | **Attentive Pooling over Phrase Repr** $\alpha = \text{softmax}(W_2 \tanh(W_1 p_i))$ with padding mask | Replicates official TRM `AttentivePooling` to compute relative phrase importance $\alpha \in \mathbb{R}^{B \times P}$, where $\sum_{i} \alpha_i = 1$. |
| **Weighted Phrase Support** $S_{\text{phrase}}[j] = \sum_{i=1}^P \alpha_i S_{ij}^P$ | **Slot Phrase Support** $\text{support}_j = \sum_{i=1}^P \alpha_i S_{ij}^P$ | Aggregates phrase-level evidence supporting candidate slot $j$. Shape: $[B, Q]$. |
| **Proposal Score Refinement** $S[j] = \sigma\left(10 \cdot (S^s[j] + w S_{\text{phrase}}[j])\right)$ | **Refined Foreground Logits** $\text{logits}_{\text{refined}}[j, 0] = \text{logits}_{\text{base}}[j, 0] + \lambda_{\text{refine}} \cdot \text{support}_j$ | Refines the foreground logit (class index 0) while keeping background logit (class index 1) unchanged. Matcher and inference use refined logits. |
| **Positive Candidate Set** $A_1 = \{(s,t) \mid \text{IoU}(s, t, GT) \ge \theta\}$ | **Adapted Candidate Set** $A_{\text{pos}} = \{j \mid \text{IoU}(\text{span}_j, GT) \ge \theta\} \cup \{j_{\text{matched}}\}$ | **Critical adaptation**: DETR slots in early epochs may have IoU $< \theta$. Ensuring $A_{\text{pos}}$ includes the Hungarian matched positive slot guarantees gradient flow. |
| **Consistency Constraint (MIL)** $\max_{j \in A_1} S_{ij}^P \to 1$ | **Positive Slot Support** $\max_{j \in A_{\text{pos}}} S_{ij}^P \to 1$ for each valid phrase $i$ | Every semantic phrase in the sentence must be grounded in at least one positive temporal candidate slot. |
| **Negative Video / Phrase Contrastive** Cross-sample pairing $\max_{j} S_{ij}^{\text{neg}} \to 0$ | **Cross-Batch Slot Contrastive** $\max_{j \in \{1..Q\}} S_{ij}^{\text{neg}} \to 0$ with cyclic permutation | Phrases from sample $A$ with video slots of sample $B \to 0$; phrases from sample $B$ with video slots of sample $A \to 0$. |
| **Exclusiveness Constraint** $\min_{i=1}^P S_{ij}^P \to 0$ for $j \in A_2$ ($\text{IoU} < \theta$) | **Negative Slot Exclusiveness** $\min_{i=1}^P S_{ij}^P \to 0$ for $j \notin A_{\text{pos}}$ | For slots outside the true moment, not all phrases should match simultaneously. |

---

## 2. Mathematical Formulations

### 2.1 Model Forward
1. **Full Query & Video Input**:
   - $V_{\text{in}} \in \mathbb{R}^{B \times L_{\text{vid}} \times 2818}$, $T_{\text{in}} \in \mathbb{R}^{B \times L_{\text{txt}} \times 512}$.
   - $V_{\text{proj}} = \text{input\_vid\_proj}(V_{\text{in}})$, $T_{\text{proj}} = \text{input\_txt\_proj}(T_{\text{in}})$.
   - $\text{TransformerEncoderDecoder}(V_{\text{proj}}, T_{\text{proj}}) \to hs \in \mathbb{R}^{\text{layers} \times B \times Q \times d}$.
   - $\text{pred\_spans} = \sigma(\text{span\_embed}(hs[-1])) \in \mathbb{R}^{B \times Q \times 2}$.
   - $\text{pred\_logits\_base} = \text{class\_embed}(hs[-1]) \in \mathbb{R}^{B \times Q \times 2}$.

2. **Phrase Branch**:
   - For batch sample $b$, phrases $p_{b, 1}, \dots, p_{b, P}$ (where $P \le 10$).
   - Phrase token features $T_{b, i}^{\text{phr}} \in \mathbb{R}^{L_i \times 512}$ projected via shared $\text{input\_txt\_proj}$ and masked-mean pooled $\to \bar{p}_{b, i} \in \mathbb{R}^{d}$.
   - Importance weights via Attentive Pooling:
     $$\tilde{\alpha}_{b, i} = W_2 \tanh(W_1 \bar{p}_{b, i})$$
     $$\alpha_{b, i} = \frac{\exp(\tilde{\alpha}_{b, i})}{\sum_{k=1}^{P} m_{b, k} \exp(\tilde{\alpha}_{b, k})}$$
   - Phrase-slot cosine matching:
     $$\hat{p}_{b, i} = \frac{\text{phrase\_proj}(\bar{p}_{b, i})}{\|\text{phrase\_proj}(\bar{p}_{b, i})\|_2}, \quad \hat{h}_{b, j} = \frac{\text{slot\_proj}(h_{b, j})}{\|\text{slot\_proj}(h_{b, j})\|_2}$$
     $$\text{raw\_cosine}_{b, i, j} = \hat{p}_{b, i} \cdot \hat{h}_{b, j} \in [-1, 1]$$
     $$S_{b, i, j}^P = \sigma\left(10 \cdot \text{raw\_cosine}_{b, i, j}\right) \in (0, 1)$$
   - Weighted phrase support per proposal slot (raw-cosine weighted support):
     $$\text{support}_{b, j} = \sum_{i=1}^P \alpha_{b, i} \cdot \text{raw\_cosine}_{b, i, j} \in [-1, 1]$$
   - Refined foreground logit (Moment-DETR adaptation of TRM proposal score fusion):
     $$\text{pred\_logits}[b, j, 0] = \text{pred\_logits\_base}[b, j, 0] + \lambda_{\text{refine}} \cdot 10 \cdot \text{support}_{b, j}$$
     $$\text{pred\_logits}[b, j, 1] = \text{pred\_logits\_base}[b, j, 1]$$
     *(Enables bidirectional evidence: positive alignment boosts foreground confidence, while negative cosine suppresses foreground proposal)*.

### 2.2 Losses
Total training loss:
$$\mathcal{L} = \mathcal{L}_{\text{span}} + \mathcal{L}_{\text{giou}} + \mathcal{L}_{\text{label}} + \lambda_{\text{con}} \mathcal{L}_{\text{con\_pos}} + \lambda_{\text{neg}} \mathcal{L}_{\text{con\_neg}} + \lambda_{\text{exc}} \mathcal{L}_{\text{exc}}$$

1. **Base Moment-DETR Losses**:
   - $\mathcal{L}_{\text{span}} = \lambda_{\text{L1}} \|\text{span}_{\hat{\sigma}} - \text{GT}\|_1$
   - $\mathcal{L}_{\text{giou}} = \lambda_{\text{giou}} (1 - \text{GIoU}(\text{span}_{\hat{\sigma}}, \text{GT}))$
   - $\mathcal{L}_{\text{label}} = \text{CrossEntropy}(\text{pred\_logits}, \text{targets}_{\text{classes}}, \text{weights}=[1.0, 0.1])$

2. **Phrase Consistency Loss ($\mathcal{L}_{\text{con\_pos}}$)**:
   For valid phrases ($m_i = 1$):
   $$p_i^{\max} = \max_{j \in A_{\text{pos}}} S_{ij}^P$$
   $$\mathcal{L}_{\text{con\_pos}} = \frac{1}{\sum_i m_i} \sum_{i=1}^P m_i \cdot \left(-\log(p_i^{\max}) \cdot (1 - p_i^{\max})^2\right)$$

3. **Negative Contrastive Loss ($\mathcal{L}_{\text{con\_neg}}$)**:
   For cyclic permutation offset $k = 1$:
   - Current phrases against negative video slots:
     $$p_{\text{neg\_vid}, i}^{\max} = \max_{j \in \{1..Q\}} S_{i, j}^{\text{neg\_vid}}$$
     $$\mathcal{L}_{\text{neg\_vid}} = \frac{1}{\sum_i m_i} \sum_{i=1}^P m_i \cdot \left(-\log(1 - p_{\text{neg\_vid}, i}^{\max}) \cdot (p_{\text{neg\_vid}, i}^{\max})^2\right)$$
   - Negative phrases against current video slots:
     $$p_{\text{neg\_phr}, i}^{\max} = \max_{j \in \{1..Q\}} S_{i, j}^{\text{neg\_phr}}$$
     $$\mathcal{L}_{\text{neg\_phr}} = \frac{1}{\sum_i m_{\text{neg}, i}} \sum_{i=1}^P m_{\text{neg}, i} \cdot \left(-\log(1 - p_{\text{neg\_phr}, i}^{\max}) \cdot (p_{\text{neg\_phr}, i}^{\max})^2\right)$$
   - $\mathcal{L}_{\text{con\_neg}} = \frac{1}{2} (\mathcal{L}_{\text{neg\_vid}} + \mathcal{L}_{\text{neg\_phr}})$. (When batch size $= 1$, $\mathcal{L}_{\text{con\_neg}} = 0$).

4. **Exclusiveness Loss ($\mathcal{L}_{\text{exc}}$)**:
   For slots $j \in A_{\text{neg}} = \{j \notin A_{\text{pos}}\}$:
   $$p_j^{\min} = \min_{i: m_i=1} S_{ij}^P$$
   $$\mathcal{L}_{\text{exc}} = \frac{1}{|A_{\text{neg}}|} \sum_{j \in A_{\text{neg}}} -\log(1 - p_j^{\min}) \cdot (p_j^{\min})^2$$
   If $|A_{\text{neg}}| = 0$, $\mathcal{L}_{\text{exc}} = 0$.

---

## 3. TRM-PT Branch Design `[PAPER-DERIVED-REPRODUCTION]`

- **Isolation**: Defined in `models/moment_detr_trm_pt/` and `training/moment_detr_trm_pt/`.
- **Functionality**:
  Generates phrase pseudo temporal intervals for $S^+$ training sentences using phrase-video token similarity.
  Auxiliary phrase-level L1 / GIoU supervision is applied with a calibrated confidence weight.
  Strictly restricted to $S^+$ training set; never touches $U^+ / U^-$ evaluation splits.
