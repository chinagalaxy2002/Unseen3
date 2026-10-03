"""
Modules for Moment-DETR-TRM-GMR-Joint-v2:
1. PhraseSlotMatcher: Cosine similarity matching between phrases and slots.
2. CandidateConditionedPhraseAttention: Slot-specific phrase attention alpha_{ij}.
3. CandidateWiseGatedRefinement: Learned gate g_j for phrase support refinement.
4. EvidenceAwareExistenceHead: Smooth existential pooling over candidate evidence z_j.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhraseSlotMatcher(nn.Module):
    """
    Computes cosine similarity between phrase representations and proposal slots.
    Optionally applies linear projections W_p and W_s before cosine similarity.
    """
    def __init__(self, hidden_dim: int = 256, proj_dim: int | None = None, scale: float = 10.0):
        super().__init__()
        self.scale = scale
        self.proj_dim = proj_dim or hidden_dim
        self.phrase_proj = nn.Linear(hidden_dim, self.proj_dim)
        self.slot_proj = nn.Linear(hidden_dim, self.proj_dim)

    def forward(
        self, phrase_repr: torch.Tensor, slot_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phrase_repr: [B, P, D] phrase embeddings
            slot_repr:   [B, Q, D] decoder slot states h_j
        Returns:
            scores:      [B, P, Q] sigmoid-scaled scores for phrase loss computation
            raw_cosine:  [B, P, Q] raw cosine similarities r_{ij} in [-1, 1]
            p_norm:      [B, P, proj_dim] normalized projected phrase vectors
            s_norm:      [B, Q, proj_dim] normalized projected slot vectors
        """
        p_proj = self.phrase_proj(phrase_repr)
        s_proj = self.slot_proj(slot_repr)

        p_norm = F.normalize(p_proj, p=2, dim=-1)
        s_norm = F.normalize(s_proj, p=2, dim=-1)

        # raw_cosine r_{ij} = cos(W_p(p_i), W_s(h_j)): [B, P, Q]
        raw_cosine = torch.bmm(p_norm, s_norm.transpose(1, 2))

        # Sigmoid-scaled matching score for contrastive/consistency losses
        scores = torch.sigmoid(self.scale * raw_cosine)
        return scores, raw_cosine, p_norm, s_norm


class CandidateConditionedPhraseAttention(nn.Module):
    """
    Candidate-conditioned phrase attention:
    For each candidate slot j, compute attention weights alpha_{ij} over phrases i:
      a_{ij} = MLP_att(concat(p_i, h_j, r_{ij}))
      alpha_{ij} = softmax_i(a_{ij})
      e_j = sum_i alpha_{ij} * r_{ij}
    """
    def __init__(self, hidden_dim: int = 256, att_hid_dim: int = 128):
        super().__init__()
        # Input dimension: p_i (hidden_dim) + h_j (hidden_dim) + r_{ij} (1) = 2 * hidden_dim + 1
        in_dim = 2 * hidden_dim + 1
        self.mlp_att = nn.Sequential(
            nn.Linear(in_dim, att_hid_dim),
            nn.ReLU(),
            nn.Linear(att_hid_dim, 1),
        )

    def forward(
        self,
        phrase_repr: torch.Tensor,
        slot_repr: torch.Tensor,
        raw_cosine: torch.Tensor,
        phrase_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phrase_repr: [B, P, D]
            slot_repr:   [B, Q, D]
            raw_cosine:  [B, P, Q] r_{ij} in [-1, 1]
            phrase_mask: [B, P] binary mask (1 for valid phrase, 0 for pad)
        Returns:
            candidate_phrase_support:   [B, Q] e_j in [-1, 1]
            candidate_phrase_attention: [B, P, Q] alpha_{ij}
        """
        B, P, D = phrase_repr.shape
        _, Q, _ = slot_repr.shape

        # Broadcast to [B, P, Q, D]
        p_exp = phrase_repr.unsqueeze(2).expand(B, P, Q, D)
        h_exp = slot_repr.unsqueeze(1).expand(B, P, Q, D)
        r_exp = raw_cosine.unsqueeze(-1)  # [B, P, Q, 1]

        pair_feat = torch.cat([p_exp, h_exp, r_exp], dim=-1)  # [B, P, Q, 2D + 1]
        a_ij = self.mlp_att(pair_feat).squeeze(-1)            # [B, P, Q]

        # Masked softmax across phrase dimension P (dim=1)
        mask = phrase_mask.unsqueeze(2).expand(B, P, Q).bool()  # [B, P, Q]
        a_ij_masked = a_ij.masked_fill(~mask, -1e9)
        alpha_ij = F.softmax(a_ij_masked, dim=1)                # [B, P, Q]
        alpha_ij = alpha_ij * mask.float()

        # Handle samples where all phrases are masked (safety guard)
        denom = alpha_ij.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha_ij = alpha_ij / denom

        # Candidate-specific phrase support: e_j = sum_i alpha_{ij} * r_{ij}
        # [B, P, Q] * [B, P, Q] -> sum over P (dim 1) -> [B, Q]
        candidate_phrase_support = (alpha_ij * raw_cosine).sum(dim=1)
        return candidate_phrase_support, alpha_ij


class CandidateWiseGatedRefinement(nn.Module):
    """
    Candidate-wise gated phrase refinement:
    base_margin_j = base_fg_j - base_bg_j
    gate_input_j = concat(h_j, base_margin_j, e_j)
    g_j = sigmoid(MLP_gate(gate_input_j))
    refined_fg_j = base_fg_j + lambda_refine * phrase_scale * g_j * e_j
    refined_bg_j = base_bg_j
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        gate_hid_dim: int = 128,
        lambda_refine: float = 1.0,
        phrase_scale: float = 10.0,
    ):
        super().__init__()
        self.lambda_refine = float(lambda_refine)
        self.phrase_scale = float(phrase_scale)
        # Input: h_j (hidden_dim) + base_margin (1) + e_j (1) = hidden_dim + 2
        in_dim = hidden_dim + 2
        self.mlp_gate = nn.Sequential(
            nn.Linear(in_dim, gate_hid_dim),
            nn.ReLU(),
            nn.Linear(gate_hid_dim, 1),
        )

    def forward(
        self,
        slot_repr: torch.Tensor,
        pred_logits_base: torch.Tensor,
        candidate_phrase_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slot_repr:                 [B, Q, D] h_j
            pred_logits_base:          [B, Q, 2] class 0=fg, class 1=bg
            candidate_phrase_support:  [B, Q] e_j
        Returns:
            pred_logits:               [B, Q, 2] refined logits
            candidate_phrase_gate:     [B, Q] g_j in [0, 1]
            base_margin:               [B, Q] base_margin_j
        """
        # base_margin_j = base_fg_logit_j - base_bg_logit_j
        base_margin = pred_logits_base[..., 0] - pred_logits_base[..., 1]  # [B, Q]

        gate_input = torch.cat(
            [slot_repr, base_margin.unsqueeze(-1), candidate_phrase_support.unsqueeze(-1)],
            dim=-1,
        )  # [B, Q, D + 2]
        g_j = torch.sigmoid(self.mlp_gate(gate_input)).squeeze(-1)  # [B, Q]

        pred_logits = pred_logits_base.clone()
        # refined_fg_j = base_fg_j + lambda_refine * phrase_scale * g_j * e_j
        pred_logits[..., 0] = (
            pred_logits_base[..., 0]
            + self.lambda_refine * self.phrase_scale * g_j * candidate_phrase_support
        )
        pred_logits[..., 1] = pred_logits_base[..., 1]

        return pred_logits, g_j, base_margin


class EvidenceAwareExistenceHead(nn.Module):
    """
    Evidence-aware existence head with smooth existential pooling:
    z_j = concat(h_j, base_margin_j, refined_margin_j, e_j, g_j)
    q_j = MLP_exist_candidate(z_j)
    exist_logit = logsumexp(q_j, dim=1) - log(Q)
    """
    def __init__(self, hidden_dim: int = 256, exist_hid_dim: int = 128, num_queries: int = 10):
        super().__init__()
        self.num_queries = num_queries
        self.log_q = math.log(num_queries)
        # Input: h_j (hidden_dim) + base_margin (1) + refined_margin (1) + e_j (1) + g_j (1) = hidden_dim + 4
        in_dim = hidden_dim + 4
        self.mlp_exist_candidate = nn.Sequential(
            nn.Linear(in_dim, exist_hid_dim),
            nn.ReLU(),
            nn.Linear(exist_hid_dim, 1),
        )

    def forward(
        self,
        slot_repr: torch.Tensor,
        base_margin: torch.Tensor,
        pred_logits_refined: torch.Tensor,
        candidate_phrase_support: torch.Tensor,
        candidate_phrase_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slot_repr:                [B, Q, D] h_j
            base_margin:              [B, Q]
            pred_logits_refined:      [B, Q, 2]
            candidate_phrase_support: [B, Q] e_j
            candidate_phrase_gate:    [B, Q] g_j
        Returns:
            pred_exist_logits:        [B] scalar existence logit per sample
            q_j:                      [B, Q] candidate existence evidence
            refined_margin:           [B, Q] refined_margin_j
        """
        refined_margin = pred_logits_refined[..., 0] - pred_logits_refined[..., 1]  # [B, Q]

        z_j = torch.cat(
            [
                slot_repr,
                base_margin.unsqueeze(-1),
                refined_margin.unsqueeze(-1),
                candidate_phrase_support.unsqueeze(-1),
                candidate_phrase_gate.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, Q, D + 4]

        q_j = self.mlp_exist_candidate(z_j).squeeze(-1)  # [B, Q]

        # Smooth existential pooling: logsumexp(q_j, dim=1) - log(Q)
        exist_logit = torch.logsumexp(q_j, dim=1) - self.log_q  # [B]
        return exist_logit, q_j, refined_margin
