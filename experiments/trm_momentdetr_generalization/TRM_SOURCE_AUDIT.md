# TRM Official Repository and Paper Audit

This document provides a detailed, evidence-based audit of the official TRM repository (`minghangz/TRM`) and the AAAI 2023 / IJCV 2026 research papers, fulfilling Sections 3, 4, and 5 of `plan.md`.

---

## 1. Official Repository Provenance

- **Repository URL**: `https://github.com/minghangz/TRM`
- **Local Read-Only Clone**: `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen3/_external/TRM`
- **Commit Hash**: `a0fc1f97927bc1a5e53ae30d1032e343f862911f` (Branch: `main`)
- **Key Files Audited**:
  - `trm/modeling/trm/trm.py`
  - `trm/modeling/trm/text_encoder.py`
  - `trm/modeling/trm/loss.py`
  - `trm/modeling/trm/proposal_conv.py`
  - `trm/modeling/trm/feat2d.py`
  - `trm/modeling/trm/featpool.py`
  - `trm/data/datasets/charades.py`
  - `trm/data/collate_batch.py`
  - `trm/engine/trainer.py`
  - `trm/engine/inference.py`
  - `configs/charades.yaml`

---

## 2. Detailed Technical Audit of TRM Mechanisms (Items A through K)

### A. Sentence Encoder & Phrase Encoder Parameter Sharing `[TRM-REPO]`
- In `trm/modeling/trm/text_encoder.py` (`DistilBert.encode_sentences`, lines 120-136):
  Both full sentence queries (`stnc_query`) and extracted phrases (`phrase_query`) pass through the exact same method:
  `self.encode_single(query, word_len)`.
- The DistilBERT backbone and projection layers `fc_out1` and `fc_out2` are 100% shared between sentences and phrases.

### B. AttentivePooling and the Global Sentence Feature `[TRM-REPO]` vs `[TRM-PAPER]`
- In `trm/modeling/trm/text_encoder.py` (`AttentivePooling`, lines 10-51):
  ```python
  def forward(self, feats, global_feat, f_masks=None):
      # ...
      # The cross-attention lines with global_feat are commented out in the official code:
      # self.fc_phrase = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
      # self.fc_sent = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
      # alpha = torch.bmm(feats, global_feat) / math.sqrt(self.att_hid_dim)
      
      # The actual active code is self-attentive pooling over phrase features alone:
      attn_f = self.feat2att(feats)
      dot = torch.tanh(attn_f)
      alpha = self.to_alpha(dot)
      if f_masks is not None:
          alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
      attw = F.softmax(alpha, dim=1).squeeze(-1)
      return attw
  ```
- **Factual Audit Finding**: Although `global_feat` is passed into `forward()`, it does **not** participate in the calculation. The phrase importance weight $\alpha_i$ is computed purely from the phrase embeddings via `Linear -> Tanh -> Linear -> Masked Softmax`.

### C. Maximum Phrase Budget `[TRM-REPO]`
- In `text_encoder.py` (lines 132-133):
  `out, out_iou = self.encode_single(query[:10].cuda(), word_len[:10].cuda())`
  `pad_tensor = torch.zeros(10-len(out), self.joint_space_size)`
- The maximum number of phrases per sentence is hardcoded to $P = 10$. Sentences with more than 10 phrases are truncated, and sentences with fewer are zero-padded.

### D. Phrase Dropout `[TRM-REPO]`
- In `text_encoder.py` (lines 145-152):
  During training when `DROP_PHRASE=True`, each phrase is independently kept with probability $0.9$:
  `phrase_keep_weight[i, j] = 0.9; drop_mask = torch.bernoulli(phrase_keep_weight); phrase_mask = phrase_mask * drop_mask`.
- Keep probability = 0.9 (dropout probability = 0.1).

### E. Proposal Score Refinement via Phrase Support `[TRM-REPO]` vs `[MOMENT-DETR-ADAPTATION]`
- In `trm.py` (lines 80-97):
  The weighted phrase support for proposal $j$ is computed as:
  $S_{\text{phrase}}[j] = \sum_{i=1}^{P} \alpha_i \cdot S_{\text{phrase}}[i, j]$.
  The sentence score is refined as:
  $S_{\text{refined}}[j] = \sigma\left(10 \times \left(S_{\text{sent}}[j] + w \cdot S_{\text{phrase}}[j]\right)\right)$, where $w = \text{RESIDUAL} = 1.0$.
- In Moment-DETR:
  $h_j$ is the $j$-th decoder slot (acting as the temporal proposal).
  $S_{\text{phrase}}[i, j] = \cos(\text{proj}_p(p_i), \text{proj}_s(h_j))$.
  $S_{\text{phrase\_support}}[j] = \sum_{i=1}^{P} \alpha_i \cdot S_{\text{phrase}}[i, j]$.
  The base foreground logit is refined:
  $\text{logit}_{\text{refined}}[j] = \text{logit}_{\text{base}}[j] + \lambda_{\text{refine}} \cdot S_{\text{phrase\_support}}[j]$ `[MOMENT-DETR-ADAPTATION]`.

### F. Phrase-to-Proposal Score Computation `[TRM-REPO]`
- L2-normalized phrase representation $\hat{p}_i$ and L2-normalized visual proposal feature $\hat{v}_j$:
  $S_{\text{raw}}[i, j] = \hat{p}_i \cdot \hat{v}_j$.
  Scaled cosine with sigmoid:
  $S[i, j] = \sigma(10 \times S_{\text{raw}}[i, j])$.

### G. Cosine Normalization and Sigmoid Temperature `[TRM-REPO]`
- Temperature $\tau = 0.1$, corresponding to multiplication by 10 before sigmoid: $\sigma(S / 0.1)$.

### H. Positive Phrase Consistency Loss `[TRM-PAPER]` & `[TRM-REPO]`
- For positive training samples, let $A_{\text{pos}}$ be the set of proposals with IoU $\ge \theta$ (with threshold $\theta = 0.1$ in Charades).
- In TRM:
  $\max_{j \in A_{\text{pos}}} S_{\text{phrase}}[i, j] \to 1$.
  Focal loss formulation:
  $L_{\text{con\_pos}} = -\log(p_{\max}) \cdot (1 - p_{\max})^2$.
- In Moment-DETR:
  To avoid empty $A_{\text{pos}}$ in early DETR training, $A_{\text{pos}} = \{j \mid \text{IoU}(pred\_span_j, GT) \ge \theta\} \cup \{\text{Hungarian matched slot}\}$ `[MOMENT-DETR-ADAPTATION]`.

### I. Negative Contrastive Learning (Negative Videos & Negative Phrases) `[TRM-REPO]`
- In `trm.py` (lines 137-221):
  For each sample $i$ in batch, an unmatched sample `neg_idx` is selected.
  - Negative video: current phrases against proposals of `neg_idx`.
    $\max_j S_{\text{neg\_vid}}[i, j] \to 0$. Loss $= -\log(1 - p_{\max}) \cdot p_{\max}^2$.
  - Negative phrase: phrases of `neg_idx` against current video proposals.
    $\max_j S_{\text{neg\_phr}}[i, j] \to 0$. Loss $= -\log(1 - p_{\max}) \cdot p_{\max}^2$.
  - Total negative loss: $L_{\text{neg}} = (L_{\text{neg\_vid}} + L_{\text{neg\_phr}}) / 2$.

### J. Exclusiveness Loss `[TRM-PAPER]` & `[TRM-REPO]`
- For proposals outside GT temporal window ($A_{\text{neg}} = \{j \mid \text{IoU} < \theta\}$):
  Not all phrases should match simultaneously!
  $\min_{i=1}^P S_{\text{phrase}}[i, j] \to 0$.
  Loss:
  $L_{\text{exc}} = \frac{1}{|A_{\text{neg}}|} \sum_{j \in A_{\text{neg}}} -\log(1 - p_{\min, j}) \cdot p_{\min, j}^2$.

### K. Charades Configuration & Loss Weight Policy
In `configs/charades.yaml`:
```yaml
CONSIS_WEIGHT: 1.0
EXC_WEIGHT: 0.0
THRESH: 0.1
CONTRASTIVE: True
DROP_PHRASE: True
RESIDUAL: 1.0
```
- **Discrepancy Documented**:
  The AAAI paper describes exclusiveness as an essential pillar of TRM (`L = Liou + Lcont + Lcon + Lex`). However, in the released `configs/charades.yaml`, `EXC_WEIGHT` is set to `0.0`.
- **Project Policy for Unseen3**:
  In accordance with `plan.md` Section 3 ("完整机制迁移"), we implement the complete theoretical mechanism including exclusiveness.
  We freeze the default loss weights in advance:
  - $\lambda_{\text{refine}} = 1.0$
  - $\lambda_{\text{consistency}} = 1.0$
  - $\lambda_{\text{negative}} = 0.5$ (matching the $0.5 \times (L_{pos} + L_{neg})$ weighting in `trainer.py:89`)
  - $\lambda_{\text{exclusiveness}} = 1.0$
  - $\theta_{\text{IoU}} = 0.1$
  - Phrase dropout keep probability $= 0.9$

---

## 3. TRM-PT Paper Audit & Scope Clarification

- **Paper Title**: *Large-Scale Pre-trained Models Empowering Phrase Generalization in Temporal Sentence Localization* (IJCV 2026, Liu et al.).
- **Repo Reality**: The released `minghangz/TRM` repository (`main` branch) contains the AAAI 2023 TRM codebase. There is **no** official release of TRM-PT code in the repository.
- **Paper Mechanism**:
  TRM-PT uses large-scale multimodal models (e.g. CLIP / video-language models) to generate phrase-level pseudo temporal labels and auxiliary iterative refinement for phrases during training on $S^+$.
- **Designation in Unseen3**:
  Any TRM-PT module is categorized as `[PAPER-DERIVED-REPRODUCTION]`, completely isolated from core TRM, and strictly restricted to $S^+$ training data.
