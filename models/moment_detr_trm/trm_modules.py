from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentivePooling(nn.Module):
    """
    AttentivePooling faithfully ported from official TRM (trm/modeling/trm/text_encoder.py).
    Computes phrase importance weights alpha_i over valid phrases via:
        alpha = softmax(to_alpha(tanh(feat2att(feats))))
    Masks padding phrases so they receive zero weight.
    """
    def __init__(self, feat_dim: int, att_hid_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.att_hid_dim = att_hid_dim
        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor, phrase_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            feats: [B, P, D] phrase representation vectors
            phrase_mask: [B, P] binary mask (1 for valid phrase, 0 for pad)
        Returns:
            attw: [B, P] normalized importance weights summing to 1 across valid phrases
        """
        attn_f = self.feat2att(feats)
        dot = torch.tanh(attn_f)
        alpha = self.to_alpha(dot)  # [B, P, 1]

        if phrase_mask is not None:
            # Mask out padding phrases before softmax
            alpha = alpha.masked_fill(phrase_mask.unsqueeze(-1) == 0, -1e9)

        attw = F.softmax(alpha, dim=1).squeeze(-1)  # [B, P]

        if phrase_mask is not None:
            # Safety check: if an entire row has phrase_mask sum == 0, distribute uniform weight
            mask_sum = phrase_mask.sum(dim=-1, keepdim=True)
            zero_mask = (mask_sum == 0)
            if zero_mask.any():
                uniform = torch.ones_like(attw) / max(1, attw.shape[1])
                attw = torch.where(zero_mask, uniform, attw)

        return attw


class PhraseSlotMatcher(nn.Module):
    """
    Phrase-to-Slot Cosine Matching module ported from TRM (trm/modeling/trm/trm.py).
    Computes cosine similarity between projected phrase representations and
    projected decoder slot proposal representations, scaled by 10 and passed
    through sigmoid (matching TRM official implementation).
    """
    def __init__(self, hidden_dim: int, proj_dim: int | None = None, scale: float = 10.0):
        super().__init__()
        proj_dim = proj_dim or hidden_dim
        self.phrase_proj = nn.Linear(hidden_dim, proj_dim)
        self.slot_proj = nn.Linear(hidden_dim, proj_dim)
        self.scale = scale

    def forward(
        self, phrase_repr: torch.Tensor, slot_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phrase_repr: [B, P, hidden_dim]
            slot_repr:   [B, Q, hidden_dim]
        Returns:
            phrase_scores: [B, P, Q] in (0, 1) sigmoid(scale * cosine)
            raw_cosine:    [B, P, Q] cosine similarity in [-1, 1]
            p_norm:        [B, P, proj_dim] normalized phrase projection
            s_norm:        [B, Q, proj_dim] normalized slot projection
        """
        p_proj = self.phrase_proj(phrase_repr)
        s_proj = self.slot_proj(slot_repr)

        p_norm = F.normalize(p_proj, p=2, dim=-1)  # [B, P, proj_dim]
        s_norm = F.normalize(s_proj, p=2, dim=-1)  # [B, Q, proj_dim]

        # Cosine dot product: [B, P, Q]
        raw_cosine = torch.bmm(p_norm, s_norm.transpose(1, 2))
        phrase_scores = torch.sigmoid(self.scale * raw_cosine)
        return phrase_scores, raw_cosine, p_norm, s_norm
