"""
Moment-DETR-TRM-GMR-Joint-v3:
End-to-End Joint Model integrating:
  1. Shared Transformer backbone (Moment-DETR)
  2. Phrase-Slot Cosine Matching (TRM)
  3. Candidate-Conditioned Phrase Attention (alpha_{ij} per slot j)
  4. Candidate-Wise Gated Phrase Refinement (learned gate g_j)
  5. Evidence-Aware Existence Head (smooth existential pooling over candidate evidence z_j)
Fully differentiable end-to-end without stop-gradient or modular decoupling.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moment_detr_gmr.moment_detr import MomentDETR, MLP, LinearLayer
from models.moment_detr_gmr.position_encoding import build_position_encoding
from models.moment_detr_gmr.moment_transformer import build_transformer
from models.moment_detr_trm_gmr_joint_v3.joint_modules import (
    PhraseSlotMatcher,
    CandidateConditionedPhraseAttention,
    CandidateWiseGatedRefinement,
    EvidenceAwareExistenceHead,
)


class MomentDETR_TRM_GMR_Joint(MomentDETR):
    """
    Joint Moment-DETR + TRM + Candidate-conditioned Attention + Gated Refinement + Evidence Existence.
    """
    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        txt_dim: int = 512,
        vid_dim: int = 2818,
        num_queries: int = 10,
        input_dropout: float = 0.5,
        aux_loss: bool = False,
        max_v_l: int = 200,
        span_loss_type: str = "l1",
        use_txt_pos: bool = False,
        n_input_proj: int = 2,
        aud_dim: int = 0,
        # TRM & Joint parameters:
        use_phrase: bool = True,
        max_phrases: int = 10,
        drop_phrase: bool = True,
        lambda_refine: float = 1.0,
        phrase_scale: float = 10.0,
        att_hid_dim: int = 128,
        phrase_proj_dim: int | None = None,
        scale: float = 10.0,
    ):
        # Initialize MomentDETR without vanilla exist_head (joint model uses EvidenceAwareExistenceHead)
        super().__init__(
            transformer=transformer,
            position_embed=position_embed,
            txt_position_embed=txt_position_embed,
            txt_dim=txt_dim,
            vid_dim=vid_dim,
            num_queries=num_queries,
            input_dropout=input_dropout,
            aux_loss=aux_loss,
            max_v_l=max_v_l,
            span_loss_type=span_loss_type,
            use_txt_pos=use_txt_pos,
            n_input_proj=n_input_proj,
            aud_dim=aud_dim,
            use_exist_head=False,
            exist_pool="max",
        )
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.use_phrase = bool(use_phrase)
        self.max_phrases = max_phrases
        self.drop_phrase = bool(drop_phrase)
        self.lambda_refine = float(lambda_refine)
        self.phrase_scale = float(phrase_scale)
        self.scale = float(scale)

        # 1. Phrase-Slot Cosine Matcher
        self.phrase_matcher = PhraseSlotMatcher(
            hidden_dim=hidden_dim, proj_dim=phrase_proj_dim, scale=scale
        )

        # 2. Candidate-Conditioned Phrase Attention (alpha_{ij} per slot j)
        self.candidate_attention = CandidateConditionedPhraseAttention(
            hidden_dim=hidden_dim, att_hid_dim=att_hid_dim
        )

        # 3. Candidate-Wise Gated Phrase Refinement
        self.gated_refinement = CandidateWiseGatedRefinement(
            hidden_dim=hidden_dim,
            gate_hid_dim=128,
            lambda_refine=self.lambda_refine,
            phrase_scale=self.phrase_scale,
        )

        # 4. Evidence-Aware Existence Head (Smooth Existential Pooling)
        self.evidence_exist_head = EvidenceAwareExistenceHead(
            hidden_dim=hidden_dim,
            exist_hid_dim=128,
            num_queries=num_queries,
        )

    def forward(
        self,
        src_txt: torch.Tensor,
        src_txt_mask: torch.Tensor,
        src_vid: torch.Tensor,
        src_vid_mask: torch.Tensor,
        src_aud: torch.Tensor | None = None,
        src_aud_mask: torch.Tensor | None = None,
        phrase_features: torch.Tensor | None = None,
        phrase_tokens_mask: torch.Tensor | None = None,
        phrase_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list]:
        """
        Forward pass with joint gradient flow.
        """
        bsz = src_vid.shape[0]
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        # 1. Primary Transformer Forward Path
        src_vid_proj = self.input_vid_proj(src_vid)
        src_txt_proj = self.input_txt_proj(src_txt)

        src = torch.cat([src_vid_proj, src_txt_proj], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid_proj, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt_proj) if self.use_txt_pos else torch.zeros_like(src_txt_proj)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)  # (#layers, bsz, Q, 2)
        outputs_coord = self.span_embed(hs)   # (#layers, bsz, Q, 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        pred_logits_base = outputs_class[-1]  # [B, Q, 2]
        pred_spans = outputs_coord[-1]        # [B, Q, 2]
        slot_repr = hs[-1]                    # [B, Q, d] (h_j)

        # 2. Phrase Representation & Masking
        assert phrase_features is not None and phrase_tokens_mask is not None, (
            "phrase_features and phrase_tokens_mask must be provided for joint model"
        )
        B, P, L_p, D_txt = phrase_features.shape

        # Flatten through shared input_txt_proj
        flat_phr = phrase_features.view(B * P, L_p, D_txt)
        flat_proj = self.input_txt_proj(flat_phr)  # [B * P, L_p, hidden_dim]

        # Masked mean pooling over valid tokens of each phrase
        flat_tok_mask = phrase_tokens_mask.view(B * P, L_p, 1)
        sum_tokens = (flat_proj * flat_tok_mask).sum(dim=1)
        valid_counts = flat_tok_mask.sum(dim=1).clamp(min=1.0)
        phrase_repr = (sum_tokens / valid_counts).view(B, P, self.hidden_dim)  # [B, P, d]

        # Phrase dropout during training (preserving at least 1 phrase per sample)
        if phrase_mask is not None:
            eff_phrase_mask = phrase_mask.clone()
        else:
            eff_phrase_mask = torch.ones((B, P), dtype=torch.float32, device=phrase_features.device)

        if self.training and self.drop_phrase:
            keep_p = torch.full_like(eff_phrase_mask, 0.9)
            drop_bern = torch.bernoulli(keep_p)
            dropped = eff_phrase_mask * drop_bern
            has_valid = (eff_phrase_mask.sum(dim=-1) > 0)
            all_dropped = (dropped.sum(dim=-1) == 0) & has_valid
            dropped[all_dropped] = eff_phrase_mask[all_dropped]
            eff_phrase_mask = dropped

        # 3. Phrase-Slot Cosine Matching: [B, P, Q]
        phrase_scores, raw_cosine, p_norm, s_norm = self.phrase_matcher(phrase_repr, slot_repr)

        # 4. Candidate-Conditioned Phrase Attention: e_j in [-1, 1], alpha_{ij} in [0, 1]
        candidate_phrase_support, candidate_phrase_attention = self.candidate_attention(
            phrase_repr, slot_repr, raw_cosine, eff_phrase_mask
        )

        # 5. Candidate-Wise Gated Phrase Refinement: refined_fg_j = base_fg_j + lambda * scale * g_j * e_j
        pred_logits, candidate_phrase_gate, base_margin = self.gated_refinement(
            slot_repr, pred_logits_base, candidate_phrase_support
        )

        # 6. Evidence-Aware Existence Head: smooth existential pooling over candidate evidence
        pred_exist_logits, candidate_exist_evidence, refined_margin = self.evidence_exist_head(
            slot_repr, base_margin, pred_logits, candidate_phrase_support, candidate_phrase_gate
        )

        # 7. Collect All Forward Outputs
        out = {
            "pred_logits": pred_logits,
            "pred_logits_base": pred_logits_base,
            "pred_spans": pred_spans,
            "pred_exist_logits": pred_exist_logits,
            "pred_exist_logits_semantic": pred_exist_logits,
            "pred_phrase_scores": phrase_scores,
            "raw_cosine": raw_cosine,
            "candidate_phrase_support": candidate_phrase_support,
            "candidate_phrase_attention": candidate_phrase_attention,
            "candidate_phrase_gate": candidate_phrase_gate,
            "candidate_exist_evidence": candidate_exist_evidence,
            "candidate_exist_evidence_semantic": candidate_exist_evidence,
            "base_margin": base_margin,
            "refined_margin": refined_margin,
            "phrase_mask": eff_phrase_mask,
            "phrase_repr": phrase_repr,
            "slot_repr": slot_repr,
            "p_proj": p_norm,
            "s_proj": s_norm,
        }

        # Saliency projection for compatibility
        vid_mem = memory[:, :src_vid_proj.shape[1]]
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out


def build_moment_detr_trm_gmr_joint(args):
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR_TRM_GMR_Joint(
        transformer=transformer,
        position_embed=position_embedding,
        txt_position_embed=txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        max_v_l=args.max_v_l,
        span_loss_type=args.span_loss_type,
        use_txt_pos=getattr(args, "use_txt_pos", False),
        n_input_proj=args.n_input_proj,
        aud_dim=args.aud_dim if hasattr(args, "aud_dim") else 0,
        use_phrase=getattr(args, "use_phrase", True),
        max_phrases=getattr(args, "max_phrases", 10),
        drop_phrase=getattr(args, "drop_phrase", True),
        lambda_refine=getattr(args, "lambda_refine", 1.0),
        phrase_scale=getattr(args, "phrase_scale", 10.0),
        att_hid_dim=getattr(args, "att_hid_dim", 128),
        phrase_proj_dim=getattr(args, "phrase_proj_dim", None),
        scale=getattr(args, "scale", 10.0),
    )
    return model
