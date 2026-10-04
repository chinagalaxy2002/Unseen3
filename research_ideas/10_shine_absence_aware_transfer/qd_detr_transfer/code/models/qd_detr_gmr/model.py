"""
QD-DETR model and criterion for Generalized Moment Retrieval (GMR).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.qd_detr_gmr.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx
from models.qd_detr_gmr.matcher import build_matcher
from models.qd_detr_gmr.position_encoding import build_position_encoding
from models.qd_detr_gmr.misc import accuracy
from models.qd_detr_gmr.transformer import build_transformer, MLP, inverse_sigmoid
from models.qd_detr_gmr.gmr_adapter import GMRAdapter, compute_existence_loss


class LinearLayer(nn.Module):
    """Linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz: int, out_hsz: int, layer_norm: bool = True, dropout: float = 0.1, relu: bool = True):
        super().__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class QDDETR(nn.Module):
    """QD-DETR architecture with GMR adapter support."""

    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        txt_dim: int,
        vid_dim: int,
        num_queries: int,
        input_dropout: float,
        aux_loss: bool = True,
        max_v_l: int = 75,
        span_loss_type: str = "l1",
        use_txt_pos: bool = False,
        n_input_proj: int = 2,
        aud_dim: int = 0,
        use_exist_head: bool = False,
        exist_pool: str = "max",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l

        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: foreground, 1: background
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)

        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2]),
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2]),
        ][:n_input_proj])

        self.aux_loss = aux_loss
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.global_rep_token = nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = nn.Parameter(torch.randn(hidden_dim))

        self.use_exist_head = bool(use_exist_head)
        self.exist_pool = str(exist_pool)
        if self.use_exist_head:
            self.exist_head = GMRAdapter(hidden_dim, hidden_dim, pool=self.exist_pool)
        else:
            self.exist_head = None

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """
        Inputs:
            - src_txt: [batch_size, L_txt, D_txt]
            - src_txt_mask: [batch_size, L_txt] (1 for valid, 0 for pad)
            - src_vid: [batch_size, L_vid, D_vid]
            - src_vid_mask: [batch_size, L_vid] (1 for valid, 0 for pad)
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid + L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid + L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # Prepend global token and global position
        bsz = src.shape[0]
        mask_ = torch.ones((bsz, 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask_, mask], dim=1)

        src_ = self.global_rep_token.view(1, 1, self.hidden_dim).repeat(bsz, 1, 1)
        src = torch.cat([src_, src], dim=1)

        pos_ = self.global_rep_pos.view(1, 1, self.hidden_dim).repeat(bsz, 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        hs, reference, memory, memory_global = self.transformer(
            src, ~mask, self.query_embed.weight, pos, video_length=video_length
        )

        outputs_class = self.class_embed(hs)  # (#layers, bsz, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        out = {
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
        }

        if self.exist_head is not None:
            out["pred_exist_logits"] = self.exist_head(hs[-1])

        vid_mem = memory[:, :video_length]
        saliency_scores = (
            torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1)
            / math.sqrt(self.hidden_dim)
        )
        out["saliency_scores"] = saliency_scores
        out["video_mask"] = src_vid_mask

        # Negative pairs for contrastive saliency training
        if self.training and bsz > 1:
            src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
            src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
            src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
            mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()
            mask_neg = torch.cat([mask_, mask_neg], dim=1)
            src_neg = torch.cat([src_, src_neg], dim=1)
            pos_neg = pos.clone()

            _, _, memory_neg, memory_global_neg = self.transformer(
                src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length
            )
            vid_mem_neg = memory_neg[:, :video_length]
            out["saliency_scores_neg"] = (
                torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1)
                / math.sqrt(self.hidden_dim)
            )
        else:
            out["saliency_scores_neg"] = saliency_scores

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out


class SetCriterion(nn.Module):
    """Loss computation for QD-DETR with GMR support."""

    def __init__(
        self,
        matcher,
        weight_dict: dict,
        eos_coef: float,
        losses: list[str],
        span_loss_type: str = "l1",
        max_v_l: int = 75,
        saliency_margin: float = 0.2,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_spans(self, outputs, targets, indices):
        assert "pred_spans" in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            z = outputs["pred_spans"].sum() * 0.0
            return {"loss_span": z, "loss_giou": z}

        src_spans = outputs["pred_spans"][idx]
        tgt_spans = torch.cat([t["spans"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction="none")
            loss_giou = loss_span.new_zeros([1])

        losses = {
            "loss_span": loss_span.mean(),
            "loss_giou": loss_giou.mean(),
        }
        return losses

    def loss_labels(self, outputs, targets, indices, log: bool = True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], self.background_label, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {"loss_label": loss_ce.mean()}

        if log:
            if idx[0].numel() > 0:
                losses["class_error"] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
            else:
                losses["class_error"] = torch.tensor(0.0, device=src_logits.device)
        return losses

    def loss_saliency(self, outputs, targets, indices, log: bool = True):
        if "saliency_pos_labels" not in targets or "saliency_neg_labels" not in targets:
            z = outputs["pred_spans"].sum() * 0.0
            return {"loss_saliency": z}

        vid_token_mask = outputs["video_mask"]
        saliency_scores_neg = outputs.get("saliency_scores_neg")
        if saliency_scores_neg is None:
            saliency_scores_neg = outputs["saliency_scores"]

        loss_neg_pair = (
            (-torch.log(1.0 - torch.sigmoid(saliency_scores_neg).clamp(max=1.0 - 1e-6)) * vid_token_mask)
            .sum(dim=1)
            .mean()
        )

        saliency_scores = outputs["saliency_scores"]
        loss_rank_contrastive = 0.0
        if "saliency_all_labels" in targets:
            saliency_contrast_label = targets["saliency_all_labels"]
            saliency_scores_cat = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
            saliency_contrast_label_cat = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)
            mask_cat = vid_token_mask.repeat([1, 2])
            saliency_scores_cat = mask_cat * saliency_scores_cat + (1.0 - mask_cat) * -1e3

            tau = 0.5
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label_cat > 100)
                pos_mask = (saliency_contrast_label_cat >= rand_idx)

                if torch.sum(pos_mask) == 0:
                    continue
                batch_drop_mask = (torch.sum(pos_mask, dim=1) > 0).float()

                cur_scores = saliency_scores_cat * drop_mask / tau + (~drop_mask) * -1e3
                logits = cur_scores - torch.max(cur_scores, dim=1, keepdim=True)[0]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
                mean_log_prob_pos = (pos_mask * log_prob * mask_cat).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = -mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12.0

        pos_indices = targets["saliency_pos_labels"]
        neg_indices = targets["saliency_neg_labels"]
        num_pairs = pos_indices.shape[1]
        batch_indices = torch.arange(len(saliency_scores), device=saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1
        )
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1
        )
        loss_pair = (
            torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum()
            / (len(pos_scores) * num_pairs)
            * 2
        )

        loss_saliency = loss_pair + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def loss_exist(self, outputs, targets, indices=None, log: bool = True):
        return {"loss_exist": compute_existence_loss(outputs, targets)}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss: str, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
        }
        assert loss in loss_map, f"loss {loss} not in supported losses: {list(loss_map.keys())}"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "saliency" or loss == "exist":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_model(args):
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = QDDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=getattr(args, "a_feat_dim", 0),
        aux_loss=getattr(args, "aux_loss", True),
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_exist_head=bool(getattr(args, "use_exist_head", False)),
        exist_pool=str(getattr(args, "exist_pool", "max")),
    )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": getattr(args, "lw_saliency", 0.0),
    }

    if getattr(args, "aux_loss", True):
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                f"{k}_{i}": v for k, v in weight_dict.items() if k not in ["loss_saliency", "loss_exist"]
            })
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels"]
    if float(getattr(args, "lw_saliency", 0.0)) > 0:
        losses.append("saliency")

    if bool(getattr(args, "use_exist_head", False)):
        weight_dict["loss_exist"] = float(getattr(args, "exist_loss_coef", 1.0))
        losses.append("exist")

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=getattr(args, "saliency_margin", 0.2),
    )
    criterion.to(device)
    return model, criterion
