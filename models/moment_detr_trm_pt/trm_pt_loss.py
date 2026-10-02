"""
Loss computation for Moment-DETR-TRM-PT.
[PAPER-DERIVED-REPRODUCTION]
Adds auxiliary pseudo-temporal supervision on phrases for S+ training samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moment_detr_trm.trm_loss import SetCriterionTRM
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx, generalized_temporal_iou
from models.moment_detr_gmr.matcher import build_matcher

class SetCriterionTRM_PT(SetCriterionTRM):
    """
    SetCriterionTRM_PT extends SetCriterionTRM with auxiliary phrase pseudo-span losses.
    """
    def __init__(
        self,
        matcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
        losses: list[str] | None = None,
        span_loss_type: str = "l1",
        max_v_l: int = 200,
        saliency_margin: float = 0.2,
        iou_thresh: float = 0.1,
        use_focal: bool = True,
        eps: float = 1e-7,
        lambda_pt_span: float = 1.0,
        lambda_pt_giou: float = 0.5,
    ):
        if losses is None:
            losses = ["spans", "labels", "phrase_trm", "phrase_pt"]
        elif "phrase_pt" not in losses:
            losses.append("phrase_pt")

        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            span_loss_type=span_loss_type,
            max_v_l=max_v_l,
            saliency_margin=saliency_margin,
            iou_thresh=iou_thresh,
            use_focal=use_focal,
            eps=eps,
        )
        self.lambda_pt_span = lambda_pt_span
        self.lambda_pt_giou = lambda_pt_giou

    def loss_phrase_pt(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, list], indices: list
    ) -> dict[str, torch.Tensor]:
        device = outputs["pred_spans"].device
        zero = outputs["pred_spans"].sum() * 0.0

        pred_p_spans = outputs.get("pred_phrase_spans")  # [B, P, 2] in (cx, w)
        phrase_mask = outputs.get("phrase_mask")        # [B, P]
        pseudo_spans = targets.get("phrase_pseudo_spans")  # [B, P, 2] in (st, ed)
        pseudo_conf = targets.get("phrase_pseudo_conf")    # [B, P]

        if pred_p_spans is None or pseudo_spans is None or phrase_mask is None:
            return {
                "loss_phrase_pseudo_span": zero,
                "loss_phrase_pseudo_giou": zero,
            }

        # Convert pred_phrase_spans to [st, ed]
        pred_p_xx = span_cxw_to_xx(pred_p_spans)  # [B, P, 2]

        # Valid mask: valid phrase AND confidence > 0
        valid_mask = (phrase_mask > 0)
        if pseudo_conf is not None:
            valid_mask = valid_mask & (pseudo_conf > 0.0)

        valid_indices = torch.where(valid_mask)
        if len(valid_indices[0]) == 0:
            return {
                "loss_phrase_pseudo_span": zero,
                "loss_phrase_pseudo_giou": zero,
            }

        valid_pred_xx = pred_p_xx[valid_indices]        # [N_valid, 2]
        valid_pseudo_xx = pseudo_spans[valid_indices]    # [N_valid, 2]
        valid_conf = pseudo_conf[valid_indices] if pseudo_conf is not None else None

        # L1 regression loss on pseudo spans
        l1_diff = F.l1_loss(valid_pred_xx, valid_pseudo_xx, reduction="none").sum(dim=-1)  # [N_valid]
        if valid_conf is not None:
            l1_diff = l1_diff * valid_conf
        loss_span = l1_diff.mean()

        # GIoU loss using diagonal of cross-GIoU
        giou_diag = torch.diag(generalized_temporal_iou(valid_pred_xx, valid_pseudo_xx))  # [N_valid]
        loss_giou_elements = 1.0 - giou_diag
        if valid_conf is not None:
            loss_giou_elements = loss_giou_elements * valid_conf
        loss_giou = loss_giou_elements.mean()

        return {
            "loss_phrase_pseudo_span": loss_span,
            "loss_phrase_pseudo_giou": loss_giou,
        }

    def get_loss(self, loss: str, outputs: dict, targets: dict, indices: list, **kwargs) -> dict:
        if loss == "phrase_pt":
            return self.loss_phrase_pt(outputs, targets, indices, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, **kwargs)

def build_criterion_trm_pt(args):
    matcher = build_matcher(args)

    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_phrase_consistency": getattr(args, "lambda_con", 1.0),
        "loss_phrase_negative": getattr(args, "lambda_neg", 0.5),
        "loss_phrase_exclusiveness": getattr(args, "lambda_exc", 1.0),
        "loss_phrase_pseudo_span": getattr(args, "lambda_pt_span", 1.0),
        "loss_phrase_pseudo_giou": getattr(args, "lambda_pt_giou", 0.5),
    }

    if getattr(args, "lw_saliency", 0) > 0:
        weight_dict["loss_saliency"] = args.lw_saliency

    if getattr(args, "use_exist_head", False):
        weight_dict["loss_exist"] = args.exist_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items() if k in ["loss_span", "loss_giou", "loss_label"]})
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels", "phrase_trm", "phrase_pt"]
    if getattr(args, "lw_saliency", 0) > 0:
        losses.append("saliency")
    if getattr(args, "use_exist_head", False):
        losses.append("exist")

    criterion = SetCriterionTRM_PT(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=getattr(args, "saliency_margin", 0.2),
        iou_thresh=getattr(args, "iou_thresh", 0.1),
        use_focal=getattr(args, "use_focal_loss", True),
        lambda_pt_span=getattr(args, "lambda_pt_span", 1.0),
        lambda_pt_giou=getattr(args, "lambda_pt_giou", 0.5),
    )
    return criterion
