from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, representation_dim: int, hidden_dim: int = 64, dropout: float = 0.1, bound: float = 2.0):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.LayerNorm(representation_dim + 1),
            nn.Linear(representation_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_base: torch.Tensor, s0: torch.Tensor):
        x = torch.cat([z_base.detach(), s0.detach().reshape(-1, 1)], dim=-1)
        raw_delta = self.net(x).squeeze(-1)
        delta = self.bound * torch.tanh(raw_delta / self.bound)
        return s0.reshape(-1) + delta, delta
