from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

class VonNeumannEntropyRegularizer(nn.Module):

    def __init__(
        self,
        n_qubits: int,
        target_entropy: Optional[float] = None,
        min_entropy: float = 0.1,
        max_entropy: Optional[float] = None,
        weight: float = 0.01,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

        self.target_entropy = target_entropy or (0.5 * np.log(self.dim))
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy or (0.9 * np.log(self.dim))
        self.weight = weight

    def compute_entropy_from_probs(self, probs: Tensor) -> Tensor:

        eps = 1e-10
        probs = probs.clamp(min=eps)

        entropy = -torch.sum(probs * torch.log(probs), dim=-1)

        return entropy

    def compute_entropy_from_state(self, state: Tensor) -> Tensor:

        if state.dim() == 1:

            n_a = self.n_qubits // 2
            n_b = self.n_qubits - n_a
            dim_a = 2 ** n_a
            dim_b = 2 ** n_b

            state_matrix = state.view(dim_a, dim_b)

            rho_a = torch.mm(state_matrix, state_matrix.conj().t())

            eigenvalues = torch.linalg.eigvalsh(rho_a).real.clamp(min=1e-10)

            entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))

            return entropy

        entropies = []
        for s in state:
            entropies.append(self.compute_entropy_from_state(s))
        return torch.stack(entropies)

    def forward(
        self,
        quantum_outputs: Tensor,
        is_probability: bool = True,
    ) -> Tensor:

        if is_probability:
            entropy = self.compute_entropy_from_probs(quantum_outputs)
        else:
            entropy = self.compute_entropy_from_state(quantum_outputs)

        target_loss = (entropy - self.target_entropy) ** 2

        min_penalty = nn.functional.relu(self.min_entropy - entropy)
        max_penalty = nn.functional.relu(entropy - self.max_entropy)

        total_loss = target_loss + min_penalty + max_penalty

        return self.weight * total_loss.mean()

class EntanglementRegularizer(nn.Module):

    def __init__(
        self,
        n_qubits: int,
        target_entanglement: float = 0.5,
        weight: float = 0.01,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.target_entanglement = target_entanglement
        self.weight = weight

    def compute_linear_entropy(self, probs: Tensor) -> Tensor:

        return 1 - torch.sum(probs ** 2, dim=-1)

    def forward(self, quantum_outputs: Tensor) -> Tensor:

        linear_entropy = self.compute_linear_entropy(quantum_outputs)

        max_linear_entropy = 1 - 1 / (2 ** self.n_qubits)
        normalized = linear_entropy / max_linear_entropy

        loss = (normalized - self.target_entanglement) ** 2

        return self.weight * loss.mean()

class CircuitComplexityPenalty(nn.Module):

    def __init__(
        self,
        max_depth: int = 50,
        weight: float = 0.001,
        sparsity_target: float = 0.3,
    ):
        super().__init__()

        self.max_depth = max_depth
        self.weight = weight
        self.sparsity_target = sparsity_target

    def compute_effective_depth(self, model: nn.Module) -> int:

        depth = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                depth += param.numel()
        return depth

    def compute_parameter_sparsity(self, model: nn.Module) -> Tensor:

        total_params = 0
        near_zero = 0
        threshold = 0.01

        for param in model.parameters():
            total_params += param.numel()
            near_zero += (param.abs() < threshold).sum().item()

        return near_zero / max(1, total_params)

    def forward(self, model: nn.Module) -> Tensor:

        depth = self.compute_effective_depth(model)
        depth_penalty = nn.functional.relu(
            torch.tensor(depth - self.max_depth, dtype=torch.float32)
        ) / self.max_depth

        l1_reg = sum(p.abs().sum() for p in model.parameters())

        loss = depth_penalty + 0.001 * l1_reg

        return self.weight * loss

class QuantumTransformerLoss(nn.Module):

    def __init__(
        self,
        task_loss: nn.Module = None,
        n_qubits: int = 4,
        entropy_weight: float = 0.01,
        entanglement_weight: float = 0.01,
        complexity_weight: float = 0.001,
        target_entropy: float = None,
        target_entanglement: float = 0.5,
        max_depth: int = 50,
    ):
        super().__init__()

        self.task_loss = task_loss or nn.MSELoss()

        self.entropy_reg = VonNeumannEntropyRegularizer(
            n_qubits=n_qubits,
            target_entropy=target_entropy,
            weight=entropy_weight,
        )

        self.entanglement_reg = EntanglementRegularizer(
            n_qubits=n_qubits,
            target_entanglement=target_entanglement,
            weight=entanglement_weight,
        )

        self.complexity_penalty = CircuitComplexityPenalty(
            max_depth=max_depth,
            weight=complexity_weight,
        )

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        quantum_outputs: Optional[Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:

        losses = {}

        losses['task'] = self.task_loss(predictions, targets)

        if quantum_outputs is not None:
            losses['entropy'] = self.entropy_reg(quantum_outputs)
            losses['entanglement'] = self.entanglement_reg(quantum_outputs)
        else:
            losses['entropy'] = torch.tensor(0.0)
            losses['entanglement'] = torch.tensor(0.0)

        if model is not None:
            losses['complexity'] = self.complexity_penalty(model)
        else:
            losses['complexity'] = torch.tensor(0.0)

        losses['total'] = (
            losses['task'] +
            losses['entropy'] +
            losses['entanglement'] +
            losses['complexity']
        )

        return losses

class NoiseyRobustLoss(nn.Module):

    def __init__(
        self,
        base_loss: nn.Module = None,
        smoothing: float = 0.1,
        huber_delta: float = 1.0,
    ):
        super().__init__()

        self.base_loss = base_loss or nn.MSELoss()
        self.smoothing = smoothing
        self.huber_delta = huber_delta

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:

        diff = predictions - targets

        abs_diff = diff.abs()
        quadratic = 0.5 * diff ** 2
        linear = self.huber_delta * (abs_diff - 0.5 * self.huber_delta)

        loss = torch.where(abs_diff <= self.huber_delta, quadratic, linear)

        return loss.mean()
