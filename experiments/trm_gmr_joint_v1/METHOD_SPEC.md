# Method Specification: Moment-DETR-TRM-GMR-Joint-v1

## 1. Overview and Rationale

The baseline `Moment-DETR-TRM` achieved strong seen retrieval performance, but demonstrated a severe generalization drop on unseen action queries (e.g. on Split A1: Seen R1@0.5 was 43.87%, while Unseen R1@0.5 dropped to 23.01%, a gap of 20.86%). Furthermore, previous modular approaches to existence verification suffered from threshold sensitivity and false refusal of unseen positive queries.

`Moment-DETR-TRM-GMR-Joint-v1` unifies candidate-conditioned phrase attention, candidate-wise gated refinement, and an evidence-aware existence head into a single end-to-end differentiable architecture trained from scratch without stop-gradients or modular staging.

---

## 2. Architectural Formulation

### 2.1 Backbone & Slot Representations
Given video features $V \in \mathbb{R}^{L_v \times D_v}$ and query text features $T \in \mathbb{R}^{L_t \times D_t}$:
$$H_{\text{enc}} = \text{TransformerEncoder}([W_v V, W_t T])$$
$$\{h_j\}_{j=1}^Q = \text{TransformerDecoder}(H_{\text{enc}}, \text{query\_embed})$$
where $h_j \in \mathbb{R}^D$ ($D=256$) represents the latent representation of decoder slot $j \in \{1,\dots,Q\}$ ($Q=10$).
Each slot outputs an initial bounding span $\hat{s}_j \in [0, 1]^2$ and classification logits $\text{base\_logits}_j = [\text{base\_fg}_j, \text{base\_bg}_j]$.

### 2.2 Phrase Projection & Slot Matching
Each decomposed query phrase $k \in \{1,\dots,P\}$ ($P \le 10$) with token-level features is mapped via shared $W_t$ and masked average pooling to $p_k \in \mathbb{R}^D$:
$$p'_k = \frac{W_p p_k}{\|W_p p_k\|_2}, \quad s'_j = \frac{W_s h_j}{\|W_s h_j\|_2}$$
$$r_{kj} = p'_k \cdot s'_j \in [-1, 1]$$
$$S_{kj} = \sigma(10.0 \cdot r_{kj})$$

### 2.3 Candidate-Conditioned Phrase Attention
Instead of unconditioned global phrase pooling, each candidate slot $j$ dynamically attends across phrases:
$$a_{kj} = \text{MLP}_{\text{att}}([p_k, h_j, r_{kj}]) \in \mathbb{R}$$
$$\alpha_{kj} = \frac{\exp(a_{kj})}{\sum_{k' \in \mathcal{P}_{\text{valid}}} \exp(a_{k'j})}$$
$$e_j = \sum_{k \in \mathcal{P}_{\text{valid}}} \alpha_{kj} r_{kj} \in [-1, 1]$$
where $e_j$ is the candidate-conditioned phrase support for slot $j$.

### 2.4 Candidate-Wise Gated Refinement
To prevent noisy phrase matches from corrupting confident proposals:
$$\text{base\_margin}_j = \text{base\_fg}_j - \text{base\_bg}_j$$
$$g_j = \sigma\left(\text{MLP}_{\text{gate}}([h_j, \text{base\_margin}_j, e_j])\right) \in [0, 1]$$
$$\text{refined\_fg}_j = \text{base\_fg}_j + \lambda_{\text{refine}} \cdot 10.0 \cdot g_j \cdot e_j$$
$$\text{refined\_bg}_j = \text{base\_bg}_j$$

### 2.5 Evidence-Aware Existence Head
Existence is computed from candidate-level evidence pooled smoothly across all slots:
$$\text{refined\_margin}_j = \text{refined\_fg}_j - \text{refined\_bg}_j$$
$$z_j = [h_j, \text{base\_margin}_j, \text{refined\_margin}_j, e_j, g_j] \in \mathbb{R}^{D+4}$$
$$q_j = \text{MLP}_{\text{exist\_candidate}}(z_j) \in \mathbb{R}$$
$$\text{exist\_logit} = \log\left(\sum_{j=1}^Q \exp(q_j)\right) - \log Q$$
$$\text{exist\_score} = \sigma(\text{exist\_logit}) \in [0, 1]$$

---

## 3. End-to-End Joint Objectives

Training minimizes:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{span}} + \mathcal{L}_{\text{giou}} + \mathcal{L}_{\text{label}} + 1.0 \cdot \mathcal{L}_{\text{con}} + 0.5 \cdot \mathcal{L}_{\text{neg}} + 1.0 \cdot \mathcal{L}_{\text{exc}} + 1.0 \cdot \mathcal{L}_{\text{exist}}$$

- For positive samples ($S^+$): All loss terms are active.
- For negative samples ($S^-$): $\mathcal{L}_{\text{span}}$ and $\mathcal{L}_{\text{giou}}$ are skipped; $\mathcal{L}_{\text{label}}$ supervises all slots to background; positive phrase consistency and exclusiveness are skipped; cross-batch negative contrastive loss and existence BCE ($\text{label}=0$) supervise negative rejection.
- $\mathcal{L}_{\text{exist}}$ backpropagates directly into $z_j \to g_j \to e_j \to \alpha_{kj} \to r_{kj} \to h_j \to \text{Transformer}$, establishing unified gradient feedback.

---

## 4. Evaluation Protocol

1. **Checkpoint Selection**: Strictly based on Seen validation MR-full-mAP ($\text{best.ckpt}$).
2. **Threshold Calibration**: On Seen validation predictions using `choose_threshold(method="f1_optimal")` to yield $\tau^*$.
3. **Four-Quadrant Evaluation**:
   - $S^+$ (Seen Positive): Localization R1@0.3, R1@0.5, R1@0.7, mIoU; False Refusal Rate (FRR).
   - $S^-$ (Seen Negative): Rejection Rate (RR).
   - $U^+$ (Unseen Positive): Localization R1@0.3, R1@0.5, R1@0.7, mIoU; FRR; Raw-correct rejection rate.
   - $U^-$ (Unseen Negative): Rejection Rate (RR).
   - Existence AUROC: Seen AUROC ($S^+ \text{ vs } S^-$) and Unseen AUROC ($U^+ \text{ vs } U^-$).
   - Matched Pair Accuracy: On paired $U^+ / U^-$ queries.
