from __future__ import annotations

import math

import torch
from torch import nn


class VectorReadout(nn.Module):
    def __init__(self, input_dim=256, hidden=128, dropout=0.1, key="pooled"):
        super().__init__()
        self.key = key
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch):
        x = batch[self.key]
        if self.key == "query_tokens":
            mask = batch["query_mask"]
            x = (x * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        return self.net(x).squeeze(-1)


class QueryConditionedReadout(nn.Module):
    """R2/R3 have identical parameters; only pooled vs set evidence differs."""
    def __init__(self, hidden=128, dropout=0.1, use_slots=False):
        super().__init__()
        self.use_slots = use_slots
        self.evidence = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden))
        self.query = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden))
        self.cross_attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.output_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch):
        x = batch["slots"] if self.use_slots else batch["pooled"][:, None]
        x = self.evidence(x)
        query = self.query(batch["query_tokens"])
        mask = batch["query_mask"].bool()
        if not mask.any(dim=1).all():
            raise ValueError("Empty text query")
        context, _ = self.cross_attention(x, query, query, key_padding_mask=~mask, need_weights=False)
        fused = self.norm(x + context)
        fused = self.output_norm(fused + self.ffn(fused))
        candidate = self.head(fused).squeeze(-1)
        return torch.logsumexp(candidate, dim=1) - math.log(candidate.shape[1])


class ResidualDiagnostic(nn.Module):
    """V4 architecture, differing in only the explicitly named constraint."""
    def __init__(self, bound=2.0):
        super().__init__()
        self.bound = bound
        self.net = nn.Sequential(nn.LayerNorm(257), nn.Linear(257, 64), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, batch):
        base = batch["base_logit"]
        raw = self.net(torch.cat([batch["pooled"], base[:, None]], -1)).squeeze(-1)
        delta = raw if self.bound is None else self.bound * torch.tanh(raw / self.bound)
        return base + delta


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_pooled_hidden(target):
    return min(range(32, 769, 16), key=lambda width: abs(parameter_count(VectorReadout(hidden=width)) - target))


def make_readout(name, hidden=128):
    if name == "R1":
        return VectorReadout(hidden=hidden), ["pooled"]
    if name == "R1_matched":
        target = parameter_count(QueryConditionedReadout(hidden=hidden))
        width = matched_pooled_hidden(target)
        return VectorReadout(hidden=width), ["pooled"]
    if name in ("R2", "R3"):
        return QueryConditionedReadout(hidden=hidden, use_slots=name == "R3"), ["slots" if name == "R3" else "pooled", "query_tokens", "query_mask"]
    if name == "Cq":
        return VectorReadout(512, hidden, key="query_tokens"), ["query_tokens", "query_mask"]
    if name == "Cv":
        return VectorReadout(2816, hidden, key="video_mean"), ["video_mean"]
    if name in ("D_v4", "D_no_bound", "D_no_anchor"):
        return ResidualDiagnostic(bound=None if name == "D_no_bound" else 2.0), ["pooled", "base_logit"]
    raise ValueError(name)
