from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

class QuantumResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor, sublayer_output: Tensor) -> Tensor:

        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1 - alpha) * self.norm(self.dropout(sublayer_output))

class QuantumSkipConnection(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer_output: Tensor) -> Tensor:

        return x + self.dropout(sublayer_output)

class QuantumDropout(nn.Module):

    def __init__(self, p: float = 0.1, mode: str = "amplitude"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        if self.mode == "amplitude":

            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)

        elif self.mode == "phase":

            noise = torch.randn_like(x) * self.p
            return x * torch.cos(noise)

        return x

class QuantumGatingUnit(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:

        combined = torch.cat([x, hidden], dim=-1)
        gate_value = self.gate(combined)
        return gate_value * x + (1 - gate_value) * hidden
