from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

@dataclass
class BenchmarkConfig:

    n_samples: int = 1000
    n_random_circuits: int = 100
    dimension_samples: int = 500
    batch_size: int = 32

class EffectiveDimensionAnalyzer:

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples

    def compute_fisher_eigenvalues(
        self,
        model: nn.Module,
        data_loader,
        n_samples: int = 100,
    ) -> np.ndarray:

        gradients = []

        model.eval()
        count = 0

        for inputs, targets in data_loader:
            if count >= n_samples:
                break

            model.zero_grad()
            outputs = model(inputs)

            loss = ((outputs - targets) ** 2).sum()
            loss.backward()

            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))

            if grad_vec:
                grad_vec = torch.cat(grad_vec).detach().numpy()
                gradients.append(grad_vec)

            count += len(inputs)

        if not gradients:
            return np.array([1.0])

        G = np.array(gradients)

        n = G.shape[0]
        F = G.T @ G / n

        eigenvalues = np.linalg.eigvalsh(F)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        return np.sort(eigenvalues)[::-1]

    def compute_effective_dimension(
        self,
        eigenvalues: np.ndarray,
        n_data: int,
        gamma: float = 1.0,
    ) -> float:

        if len(eigenvalues) == 0:
            return 0

        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)

        term = np.sqrt(1 + gamma**2 * n_data * eigenvalues / (2 * np.pi))

        numerator = 2 * np.log(term.sum())
        denominator = np.log(n_data / gamma**2)

        if denominator <= 0:
            return len(eigenvalues)

        d_eff = numerator / denominator

        return min(d_eff, len(eigenvalues))

    def analyze(
        self,
        model: nn.Module,
        data_loader,
        n_data: int = None,
    ) -> Dict[str, float]:

        eigenvalues = self.compute_fisher_eigenvalues(model, data_loader)

        total_params = sum(p.numel() for p in model.parameters())

        n_data = n_data or self.n_samples
        d_eff = self.compute_effective_dimension(eigenvalues, n_data)

        spectral_decay = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        effective_rank = np.sum(eigenvalues > 1e-8)

        return {
            'effective_dimension': d_eff,
            'total_parameters': total_params,
            'dimension_ratio': d_eff / max(1, total_params),
            'spectral_decay': spectral_decay,
            'effective_rank': effective_rank,
            'top_eigenvalue': eigenvalues[0] if len(eigenvalues) > 0 else 0,
            'eigenvalue_sum': eigenvalues.sum(),
        }

class QuantumClassicalComparison:

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        quantum_params = 0
        classical_params = 0

        for name, param in model.named_parameters():
            if any(q in name.lower() for q in ['qubit', 'quantum', 'vqc', 'rot']):
                quantum_params += param.numel()
            else:
                classical_params += param.numel()

        return {
            'total': total,
            'trainable': trainable,
            'quantum': quantum_params,
            'classical': classical_params,
        }

    @staticmethod
    def theoretical_capacity(
        n_qubits: int,
        n_layers: int,
        n_classical_params: int,
    ) -> Dict[str, float]:

        hilbert_dim = 2 ** n_qubits

        quantum_params = n_layers * n_qubits * 3

        quantum_capacity = hilbert_dim * np.log2(hilbert_dim)

        classical_capacity = n_classical_params

        return {
            'hilbert_dimension': hilbert_dim,
            'quantum_parameters': quantum_params,
            'classical_parameters': n_classical_params,
            'quantum_capacity': quantum_capacity,
            'classical_capacity': classical_capacity,
            'quantum_advantage_ratio': quantum_capacity / max(1, classical_capacity),
        }

    @staticmethod
    def compare_expressibility(
        quantum_model: nn.Module,
        classical_model: nn.Module,
        n_samples: int = 1000,
    ) -> Dict[str, float]:

        quantum_outputs = []
        classical_outputs = []

        for _ in range(n_samples):

            x = torch.randn(1, 16)

            with torch.no_grad():
                for param in quantum_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
                q_out = quantum_model(x).detach().numpy()
                quantum_outputs.append(q_out.flatten())

                for param in classical_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
                c_out = classical_model(x).detach().numpy()
                classical_outputs.append(c_out.flatten())

        quantum_outputs = np.array(quantum_outputs)
        classical_outputs = np.array(classical_outputs)

        quantum_var = np.var(quantum_outputs)
        classical_var = np.var(classical_outputs)

        return {
            'quantum_expressibility': quantum_var,
            'classical_expressibility': classical_var,
            'ratio': quantum_var / max(1e-10, classical_var),
        }

class GeneralizationAnalyzer:

    @staticmethod
    def estimate_rademacher_complexity(
        model: nn.Module,
        data: Tensor,
        n_samples: int = 100,
    ) -> float:

        n = len(data)
        complexities = []

        for _ in range(n_samples):

            sigma = 2 * torch.randint(0, 2, (n,)) - 1
            sigma = sigma.float()

            with torch.no_grad():
                outputs = model(data)
                if outputs.dim() > 1:
                    outputs = outputs.mean(dim=-1)

            correlation = (sigma * outputs).mean().item()
            complexities.append(correlation)

        return np.mean(np.abs(complexities))

    @staticmethod
    def pac_bayes_bound(
        train_error: float,
        n_samples: int,
        n_params: int,
        delta: float = 0.05,
    ) -> float:

        kl_term = n_params * np.log(2)

        bound = train_error + np.sqrt(
            (kl_term + np.log(2 * n_samples / delta)) / (2 * n_samples)
        )

        return min(1.0, bound)

    @staticmethod
    def quantum_generalization_advantage(
        n_qubits: int,
        n_data: int,
        circuit_depth: int,
    ) -> Dict[str, float]:

        quantum_params = circuit_depth * n_qubits * 3

        classical_equivalent = 2 ** n_qubits

        quantum_bound = quantum_params / n_data
        classical_bound = classical_equivalent / n_data

        advantage = classical_bound / max(1e-10, quantum_bound)

        return {
            'quantum_bound': quantum_bound,
            'classical_bound': classical_bound,
            'advantage_ratio': advantage,
            'sample_efficiency': advantage,
        }

def full_benchmark(
    quantum_model: nn.Module,
    classical_model: nn.Module,
    train_loader,
    test_loader,
    n_qubits: int = 4,
) -> Dict:

    results = {}

    results['quantum_params'] = QuantumClassicalComparison.count_parameters(quantum_model)
    results['classical_params'] = QuantumClassicalComparison.count_parameters(classical_model)

    results['capacity'] = QuantumClassicalComparison.theoretical_capacity(
        n_qubits=n_qubits,
        n_layers=4,
        n_classical_params=results['classical_params']['total'],
    )

    analyzer = EffectiveDimensionAnalyzer()
    results['quantum_effective_dim'] = analyzer.analyze(quantum_model, train_loader)
    results['classical_effective_dim'] = analyzer.analyze(classical_model, train_loader)

    gen_analyzer = GeneralizationAnalyzer()
    results['quantum_generalization'] = gen_analyzer.quantum_generalization_advantage(
        n_qubits=n_qubits,
        n_data=len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 1000,
        circuit_depth=4,
    )

    return results
