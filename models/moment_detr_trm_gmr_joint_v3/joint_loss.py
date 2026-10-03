"""
Loss computation for Moment-DETR-TRM-GMR-Joint-v3:
  L_total = L_span + L_giou + L_label
            + lambda_con * L_phrase_consistency
            + lambda_neg * L_phrase_negative
            + lambda_exc * L_phrase_exclusiveness
            + lambda_exist * L_exist
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moment_detr_gmr.moment_detr import SetCriterion
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx, temporal_iou
from models.moment_detr_gmr.matcher import build_matcher


from models.moment_detr_trm_gmr_joint_v3.robust_auc import SemanticRobustAUCLoss

class SetCriterionJoint(SetCriterion):
    """
    Joint SetCriterion for Moment-DETR-TRM-GMR-Joint-v3.
    Handles both S+ and S- samples end-to-end without detach.
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
        auc_margin: float = 1.0,
        robust_tau: float = 0.1,
        eps: float = 1e-7,
    ):
        if losses is None:
            losses = ["spans", "labels", "phrase_trm", "exist"]
        for required in ["phrase_trm", "exist"]:
            if required not in losses:
                losses.append(required)

        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            span_loss_type=span_loss_type,
            max_v_l=max_v_l,
            saliency_margin=saliency_margin,
        )
        self.iou_thresh = iou_thresh
        self.use_focal = use_focal
        self.eps = eps
        self.robust_auc = SemanticRobustAUCLoss(auc_margin, robust_tau)

    def loss_exist(self, outputs: dict, targets: dict, indices: list = None, log: bool = True) -> dict[str, torch.Tensor]:
        """
        Existence BCE loss:
        outputs["pred_exist_logits"]: [B]
        targets["exist_label"]: [B] in {0.0, 1.0}
        """
        assert "pred_exist_logits" in outputs, "pred_exist_logits missing in outputs"
        assert "exist_label" in targets, "exist_label missing in targets"

        logits = outputs["pred_exist_logits"].view(-1)
        labels = targets["exist_label"].float().view(-1)
        loss_exist = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        return {"loss_exist": loss_exist}

    def loss_robust_auc(self, outputs, targets, indices=None, log=True):
        return self.robust_auc(outputs["pred_exist_logits"], targets)

    def loss_phrase_trm(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, list], indices: list
    ) -> dict[str, torch.Tensor]:
        """
        Phrase-level temporal relationship losses:
          - Positive phrase consistency (MIL on A_pos)
          - Negative contrastive learning
          - Exclusiveness constraint (on A_neg)
        Correctly handles S- (empty GT) samples.
        """
        device = outputs["pred_spans"].device
        zero = outputs["pred_spans"].sum() * 0.0

        phrase_scores = outputs.get("pred_phrase_scores")  # [B, P, Q]
        phrase_mask = outputs.get("phrase_mask")          # [B, P]
        pred_spans = outputs["pred_spans"]                # [B, Q, 2]

        if phrase_scores is None or phrase_mask is None:
            return {
                "loss_phrase_consistency": zero,
                "loss_phrase_negative": zero,
                "loss_phrase_exclusiveness": zero,
            }

        B, P, Q = phrase_scores.shape
        target_spans_list = targets["span_labels"]  # len B list of dict(spans=[K, 2])

        con_losses = []
        exc_losses = []

        # Convert pred_spans to [st, ed] in [0, 1]
        pred_xx = span_cxw_to_xx(pred_spans)  # [B, Q, 2]

        # ----------------------------------------------------
        # 1. Positive Phrase Consistency & Exclusiveness per sample
        # ----------------------------------------------------
        for b in range(B):
            tgt_spans = target_spans_list[b]["spans"]  # [N_gt, 2] in (cx, w)
            sample_p_mask = phrase_mask[b]             # [P]
            valid_p_idx = torch.where(sample_p_mask > 0)[0]

            # If sample has no GT spans (S- negative sample) or no valid phrases, skip positive constraints
            if tgt_spans.numel() == 0 or len(valid_p_idx) == 0:
                continue

            # Compute IoU between all Q decoder slots and all GT spans for this sample
            tgt_xx = span_cxw_to_xx(tgt_spans)          # [N_gt, 2]
            ious, _ = temporal_iou(pred_xx[b], tgt_xx)  # [Q, N_gt]
            max_iou_per_slot, _ = ious.max(dim=1)       # [Q]

            pos_slots = torch.where(max_iou_per_slot >= self.iou_thresh)[0]

            # Ensure Hungarian matched positive slot is included in a_pos
            if indices is not None and b < len(indices):
                matched_src_idx, _ = indices[b]
                if matched_src_idx.numel() > 0:
                    matched_slots = matched_src_idx.to(device)
                    combined = torch.cat([pos_slots, matched_slots])
                    a_pos = torch.unique(combined)
                else:
                    a_pos = pos_slots
            else:
                a_pos = pos_slots

            if a_pos.numel() == 0:
                a_pos = max_iou_per_slot.argmax().unsqueeze(0)

            all_slots = torch.arange(Q, device=device)
            a_neg_mask = torch.ones(Q, dtype=torch.bool, device=device)
            a_neg_mask[a_pos] = False
            a_neg = all_slots[a_neg_mask]

            # (a) Positive Phrase Consistency Loss: max over a_pos
            scores_pos_slots = phrase_scores[b, valid_p_idx][:, a_pos]  # [N_valid_p, len(a_pos)]
            max_p_pos, _ = scores_pos_slots.max(dim=1)                 # [N_valid_p]
            max_p_pos = max_p_pos.clamp(min=self.eps, max=1.0 - self.eps)

            if self.use_focal:
                loss_con_sample = -(max_p_pos.log() * (1.0 - max_p_pos).pow(2)).mean()
            else:
                loss_con_sample = -(max_p_pos.log()).mean()
            con_losses.append(loss_con_sample)

            # (b) Phrase Exclusiveness Loss: min over valid phrases for each slot in a_neg
            if a_neg.numel() > 0 and len(valid_p_idx) > 0:
                scores_neg_slots = phrase_scores[b, valid_p_idx][:, a_neg]  # [N_valid_p, len(a_neg)]
                min_p_neg, _ = scores_neg_slots.min(dim=0)                  # [len(a_neg)]
                min_p_neg = min_p_neg.clamp(min=self.eps, max=1.0 - self.eps)

                if self.use_focal:
                    loss_exc_sample = -((1.0 - min_p_neg).log() * min_p_neg.pow(2)).mean()
                else:
                    loss_exc_sample = -((1.0 - min_p_neg).log()).mean()
                exc_losses.append(loss_exc_sample)

        loss_con = torch.stack(con_losses).mean() if con_losses else zero
        loss_exc = torch.stack(exc_losses).mean() if exc_losses else zero

        # ----------------------------------------------------
        # 2. Negative Contrastive Learning (Cross-batch)
        # ----------------------------------------------------
        p_proj = outputs.get("p_proj")  # [B, P, d]
        s_proj = outputs.get("s_proj")  # [B, Q, d]
        scale = getattr(self, "phrase_scale", 10.0)

        if B > 1 and p_proj is not None and s_proj is not None:
            # Negative video contrast: phrase b paired with slots b_roll
            b_roll = (torch.arange(B, device=device) + 1) % B
            neg_vid_raw = torch.bmm(p_proj, s_proj[b_roll].transpose(1, 2))  # [B, P, Q]
            neg_vid_scores = torch.sigmoid(scale * neg_vid_raw)

            # Negative phrase contrast: slots b paired with phrases b_roll
            neg_phr_raw = torch.bmm(p_proj[b_roll], s_proj.transpose(1, 2))  # [B, P, Q]
            neg_phr_scores = torch.sigmoid(scale * neg_phr_raw)
            mask_neg = phrase_mask[b_roll]

            max_neg_vid, _ = neg_vid_scores.max(dim=2)  # [B, P]
            max_neg_phr, _ = neg_phr_scores.max(dim=2)  # [B, P]

            max_neg_vid = max_neg_vid.clamp(min=self.eps, max=1.0 - self.eps)
            max_neg_phr = max_neg_phr.clamp(min=self.eps, max=1.0 - self.eps)

            if self.use_focal:
                loss_neg_vid = -(1.0 - max_neg_vid).log() * max_neg_vid.pow(2)
                loss_neg_phr = -(1.0 - max_neg_phr).log() * max_neg_phr.pow(2)
            else:
                loss_neg_vid = -(1.0 - max_neg_vid).log()
                loss_neg_phr = -(1.0 - max_neg_phr).log()

            sum_vid = (loss_neg_vid * phrase_mask).sum() / phrase_mask.sum().clamp(min=1.0)
            sum_phr = (loss_neg_phr * mask_neg).sum() / mask_neg.sum().clamp(min=1.0)
            loss_neg = (sum_vid + sum_phr) * 0.5
        else:
            loss_neg = zero

        return {
            "loss_phrase_consistency": loss_con,
            "loss_phrase_negative": loss_neg,
            "loss_phrase_exclusiveness": loss_exc,
        }

    def get_loss(self, loss: str, outputs: dict, targets: dict, indices: list, **kwargs) -> dict:
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
            "robust_auc": self.loss_robust_auc,
            "phrase_trm": self.loss_phrase_trm,
        }
        assert loss in loss_map, f"Unknown loss {loss}"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs: dict, targets: dict) -> dict[str, torch.Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in ["spans", "labels"]:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux)
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion_joint(args):
    matcher = build_matcher(args)

    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_phrase_consistency": getattr(args, "lambda_con", 1.0),
        "loss_phrase_negative": getattr(args, "lambda_neg", 0.5),
        "loss_phrase_exclusiveness": getattr(args, "lambda_exc", 1.0),
        "loss_exist": getattr(args, "exist_loss_coef", 1.0),
        "loss_robust_auc": getattr(args, "lambda_robust", 1.0),
        "loss_same_semantic": getattr(args, "lambda_matched", 1.0),
    }

    if getattr(args, "lw_saliency", 0) > 0:
        weight_dict["loss_saliency"] = args.lw_saliency

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                f"{k}_{i}": v for k, v in weight_dict.items() if k in ["loss_span", "loss_giou", "loss_label"]
            })
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels", "phrase_trm", "exist", "robust_auc"]
    if getattr(args, "lw_saliency", 0) > 0:
        losses.append("saliency")

    criterion = SetCriterionJoint(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=getattr(args, "saliency_margin", 0.2),
        iou_thresh=getattr(args, "iou_thresh", 0.1),
        use_focal=getattr(args, "use_focal_loss", True),
        auc_margin=getattr(args, "auc_margin", 1.0),
        robust_tau=getattr(args, "robust_tau", 0.1),
    )
    return criterion
