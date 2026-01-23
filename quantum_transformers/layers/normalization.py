from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

class QuantumLayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x_norm = (x - mean) / (std + self.eps)

        x_scaled = torch.tanh(x_norm)

        return self.gamma * x_scaled + self.beta

class QuantumRMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm

class AmplitudeNormalization(nn.Module):

    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:

        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / (norm + self.eps)

    def inverse(self, x: Tensor, original_norm: Tensor) -> Tensor:

        return x * original_norm
