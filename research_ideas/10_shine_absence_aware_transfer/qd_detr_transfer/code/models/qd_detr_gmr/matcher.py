"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from models.qd_detr_gmr.utils.span_utils import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For momentum reasons, the process is done in two steps:
        1) we compute a cost matrix between all targets and predictions in the batch
        2) we use linear_sum_assignment from scipy to compute the optimal assignment
    """

    def __init__(self, cost_span: float = 1, cost_giou: float = 1, cost_class: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        super().__init__()
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.cost_class = cost_class
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_span != 0 or cost_giou != 0 or cost_class != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the ground-truth span coordinates
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        sizes = [len(v["spans"]) for v in targets]
        if sum(sizes) == 0:
            empty = torch.empty((0,), dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]

        # Collect all target spans in the batch.
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # Classification cost uses foreground probability.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # Flatten predictions before building cost matrices.
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Span L1 cost.
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Temporal GIoU cost.
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            cost_giou = 0

        # Solve the final matching cost.
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        indices = []
        empty = torch.empty((0,), dtype=torch.int64)
        splits = C.split(sizes, -1)
        for i, c in enumerate(splits):
            if sizes[i] == 0:
                indices.append((empty, empty))
                continue
            ii, jj = linear_sum_assignment(c[i])
            indices.append((torch.as_tensor(ii, dtype=torch.int64), torch.as_tensor(jj, dtype=torch.int64)))
        return indices


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )
