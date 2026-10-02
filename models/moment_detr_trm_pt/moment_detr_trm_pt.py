"""
Moment-DETR-TRM-PT: Moment-DETR + TRM with Pre-trained Model Empowered Phrase Supervision.
[PAPER-DERIVED-REPRODUCTION] Derived from IJCV 2026 (Liu et al.).
Adds auxiliary phrase-level temporal span prediction supervised by pretrained multimodal pseudo labels on S+ training data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moment_detr_trm.moment_detr_trm import MomentDETR_TRM
from models.moment_detr_gmr.moment_detr import MLP
from models.moment_detr_gmr.position_encoding import build_position_encoding
from models.moment_detr_gmr.moment_transformer import build_transformer

class MomentDETR_TRM_PT(MomentDETR_TRM):
    """
    Moment-DETR-TRM-PT incorporates auxiliary phrase-level pseudo-temporal span prediction.
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
            use_phrase=use_phrase,
            max_phrases=max_phrases,
            drop_phrase=drop_phrase,
            lambda_refine=lambda_refine,
            att_hid_dim=att_hid_dim,
            phrase_proj_dim=phrase_proj_dim,
            scale=scale,
        )
        hidden_dim = transformer.d_model
        # Auxiliary phrase span head predicting [B, P, 2] in [0, 1]
        self.phrase_span_head = MLP(hidden_dim, hidden_dim, 2, 3)

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
        out = super().forward(
            src_txt=src_txt,
            src_txt_mask=src_txt_mask,
            src_vid=src_vid,
            src_vid_mask=src_vid_mask,
            src_aud=src_aud,
            src_aud_mask=src_aud_mask,
            phrase_features=phrase_features,
            phrase_tokens_mask=phrase_tokens_mask,
            phrase_mask=phrase_mask,
        )

        # Auxiliary phrase temporal span prediction
        phrase_repr = out.get("phrase_repr")
        if phrase_repr is not None:
            pred_phrase_spans = self.phrase_span_head(phrase_repr)
            if self.span_loss_type == "l1":
                pred_phrase_spans = pred_phrase_spans.sigmoid()
            out["pred_phrase_spans"] = pred_phrase_spans
        else:
            out["pred_phrase_spans"] = None

        return out

def build_moment_detr_trm_pt(args):
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR_TRM_PT(
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
