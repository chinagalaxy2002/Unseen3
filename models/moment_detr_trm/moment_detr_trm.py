"""
Moment-DETR with Phrase-Level Temporal Relationship Mining (Moment-DETR-TRM).
Extends Moment-DETR with shared text projection, AttentivePooling,
phrase-slot cosine matching, and phrase support proposal refinement.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moment_detr_gmr.moment_detr import MomentDETR, MLP, LinearLayer
from models.moment_detr_gmr.position_encoding import build_position_encoding
from models.moment_detr_gmr.moment_transformer import build_transformer
from models.moment_detr_trm.trm_modules import AttentivePooling, PhraseSlotMatcher

class MomentDETR_TRM(MomentDETR):
    """
    Moment-DETR + TRM module for Phrase-Level Temporal Relationship Mining.
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
        max_v_l: int = 75,
        span_loss_type: str = "l1",
        use_txt_pos: bool = False,
        n_input_proj: int = 2,
        aud_dim: int = 0,
        use_exist_head: bool = False,
        exist_pool: str = "max",
        # TRM specific parameters:
        use_phrase: bool = True,
        max_phrases: int = 10,
        drop_phrase: bool = True,
        lambda_refine: float = 1.0,
        att_hid_dim: int = 128,
        phrase_proj_dim: int | None = None,
        scale: float = 10.0,
    ):
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
            use_exist_head=use_exist_head,
            exist_pool=exist_pool,
        )
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.use_phrase = bool(use_phrase)
        self.max_phrases = max_phrases
        self.drop_phrase = bool(drop_phrase)
        self.lambda_refine = float(lambda_refine)

        if self.use_phrase:
            self.attentive_pooling = AttentivePooling(feat_dim=hidden_dim, att_hid_dim=att_hid_dim)
            self.phrase_matcher = PhraseSlotMatcher(
                hidden_dim=hidden_dim, proj_dim=phrase_proj_dim, scale=scale
            )
        else:
            self.attentive_pooling = None
            self.phrase_matcher = None

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
        Args:
            src_txt: [B, L_txt, D_txt] full query token features
            src_txt_mask: [B, L_txt] padding mask (1 for valid token, 0 for pad)
            src_vid: [B, L_vid, D_vid] video features
            src_vid_mask: [B, L_vid] padding mask
            phrase_features: [B, P, L_p, D_txt] token features of each phrase
            phrase_tokens_mask: [B, P, L_p] valid token mask per phrase
            phrase_mask: [B, P] valid phrase mask
        Returns:
            out dict containing:
                pred_spans: [B, Q, 2] in [0, 1]
                pred_logits: [B, Q, 2] refined logits (matcher / inference target)
                pred_logits_base: [B, Q, 2] original logits before phrase refinement
                pred_phrase_scores: [B, P, Q] phrase-slot cosine matching matrix
                phrase_weights: [B, P] phrase attention importance weights
                phrase_mask: [B, P] effective phrase mask after dropout
                phrase_repr: [B, P, d]
                slot_repr: [B, Q, d]
        """
        bsz = src_vid.shape[0]
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        # 1. Moment-DETR primary forward path
        src_vid_proj = self.input_vid_proj(src_vid)
        src_txt_proj = self.input_txt_proj(src_txt)

        src = torch.cat([src_vid_proj, src_txt_proj], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid_proj, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt_proj) if self.use_txt_pos else torch.zeros_like(src_txt_proj)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)  # (#layers, bsz, Q, 2)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, Q, 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        pred_logits_base = outputs_class[-1]  # [B, Q, 2]
        pred_spans = outputs_coord[-1]        # [B, Q, 2]
        slot_repr = hs[-1]                    # [B, Q, d]

        # 2. Phrase branch forward path
        eff_phrase_mask = None
        phrase_repr = None
        phrase_scores = None
        phrase_weights = None

        if self.use_phrase and phrase_features is not None and phrase_tokens_mask is not None:
            B, P, L_p, D_txt = phrase_features.shape

            # Flatten to pass through shared input_txt_proj
            flat_phr = phrase_features.view(B * P, L_p, D_txt)
            flat_proj = self.input_txt_proj(flat_phr)  # [B * P, L_p, hidden_dim]

            # Masked mean pooling over valid tokens of each phrase
            flat_tok_mask = phrase_tokens_mask.view(B * P, L_p, 1)
            sum_tokens = (flat_proj * flat_tok_mask).sum(dim=1)
            valid_counts = flat_tok_mask.sum(dim=1).clamp(min=1.0)
            phrase_repr = (sum_tokens / valid_counts).view(B, P, self.hidden_dim)  # [B, P, d]

            # Phrase dropout during training
            if phrase_mask is not None:
                eff_phrase_mask = phrase_mask.clone()
            else:
                eff_phrase_mask = torch.ones((B, P), dtype=torch.float32, device=phrase_features.device)

            if self.training and self.drop_phrase:
                keep_p = torch.full_like(eff_phrase_mask, 0.9)
                drop_bern = torch.bernoulli(keep_p)
                dropped = eff_phrase_mask * drop_bern
                # Safety guarantee: ensure at least one valid phrase remains per sample
                has_valid = (eff_phrase_mask.sum(dim=-1) > 0)
                all_dropped = (dropped.sum(dim=-1) == 0) & has_valid
                dropped[all_dropped] = eff_phrase_mask[all_dropped]
                eff_phrase_mask = dropped

            # Phrase importance weights via Attentive Pooling: [B, P]
            phrase_weights = self.attentive_pooling(phrase_repr, eff_phrase_mask)

            # Phrase-slot cosine matching: [B, P, Q]
            phrase_scores, _, p_norm, s_norm = self.phrase_matcher(phrase_repr, slot_repr)

            # Aggregate phrase support per proposal slot: [B, Q]
            # alpha: [B, 1, P], phrase_scores: [B, P, Q] -> [B, 1, Q] -> [B, Q]
            phrase_support = torch.bmm(phrase_weights.unsqueeze(1), phrase_scores).squeeze(1)

            # Refine foreground logit (index 0), keep background logit (index 1) unchanged
            pred_logits = pred_logits_base.clone()
            pred_logits[:, :, 0] = pred_logits_base[:, :, 0] + self.lambda_refine * phrase_support
        else:
            pred_logits = pred_logits_base
            p_norm = None
            s_norm = None

        out = {
            "pred_logits": pred_logits,
            "pred_logits_base": pred_logits_base,
            "pred_spans": pred_spans,
            "pred_phrase_scores": phrase_scores,
            "phrase_weights": phrase_weights,
            "phrase_mask": eff_phrase_mask,
            "phrase_repr": phrase_repr,
            "slot_repr": slot_repr,
            "p_proj": p_norm,
            "s_proj": s_norm,
        }

        if self.exist_head is not None:
            out["pred_exist_logits"] = self.exist_head(hs[-1])

        vid_mem = memory[:, :src_vid_proj.shape[1]]
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out


def build_moment_detr_trm(args):
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR_TRM(
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
        use_exist_head=getattr(args, "use_exist_head", False),
        exist_pool=getattr(args, "exist_pool", "max"),
        use_phrase=getattr(args, "use_phrase", True),
        max_phrases=getattr(args, "max_phrases", 10),
        drop_phrase=getattr(args, "drop_phrase", True),
        lambda_refine=getattr(args, "lambda_refine", 1.0),
        att_hid_dim=getattr(args, "att_hid_dim", 128),
        phrase_proj_dim=getattr(args, "phrase_proj_dim", None),
        scale=getattr(args, "scale", 10.0),
    )
    return model
