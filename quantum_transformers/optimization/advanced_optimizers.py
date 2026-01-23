from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

@dataclass
class QNGConfig:

    approx_method: str = "block_diagonal"
    regularization: float = 0.01
    damping: float = 0.001
    max_metric_rank: int = 100
    use_classical_fisher: bool = False

class FisherInformationMatrix:

    def __init__(
        self,
        circuit_fn: Callable,
        n_params: int,
        shift: float = np.pi / 2,
    ):
        self.circuit_fn = circuit_fn
        self.n_params = n_params
        self.shift = shift
        self._fim_cache = None

    def compute(
        self,
        params: Tensor,
        method: str = "parameter_shift",
    ) -> Tensor:

        params = params.detach().clone()
        n = self.n_params

        fim = torch.zeros(n, n)

        if method == "parameter_shift":

            gradients = torch.zeros(n)

            for i in range(n):

                params_plus = params.clone()
                params_plus[i] += self.shift
                f_plus = self.circuit_fn(params_plus)

                params_minus = params.clone()
                params_minus[i] -= self.shift
                f_minus = self.circuit_fn(params_minus)

                if isinstance(f_plus, Tensor):
                    gradients[i] = (f_plus - f_minus).mean() / (2 * np.sin(self.shift))
                else:
                    gradients[i] = (f_plus - f_minus) / (2 * np.sin(self.shift))

            fim = torch.outer(gradients, gradients)

            fim = fim + 0.001 * torch.eye(n)

        elif method == "analytic":

            for i in range(n):
                for j in range(n):
                    if i == j:
                        fim[i, j] = 1.0
                    elif abs(i - j) == 1:
                        fim[i, j] = 0.5 * np.cos(params[i] - params[j])

        self._fim_cache = fim
        return fim

    def analyze_health(self, fim: Optional[Tensor] = None) -> Dict[str, float]:

        if fim is None:
            fim = self._fim_cache

        if fim is None:
            raise ValueError("No FIM computed. Call compute() first.")

        eigenvalues = torch.linalg.eigvalsh(fim)
        eigenvalues = eigenvalues.real
        eigenvalues = torch.sort(eigenvalues, descending=True)[0]

        eps = 1e-10
        nonzero_eigs = eigenvalues[eigenvalues > eps]

        if len(nonzero_eigs) == 0:
            return {
                'condition_number': float('inf'),
                'rank': 0,
                'smallest_eigenvalue': 0.0,
                'largest_eigenvalue': 0.0,
                'trace': 0.0,
                'is_healthy': False,
                'diagnosis': 'BARREN_PLATEAU',
            }

        condition_number = (eigenvalues[0] / nonzero_eigs[-1]).item()
        rank = len(nonzero_eigs)
        smallest = nonzero_eigs[-1].item()
        largest = eigenvalues[0].item()
        trace = eigenvalues.sum().item()

        is_healthy = (
            condition_number < 1e6 and
            smallest > 1e-8 and
            rank >= self.n_params * 0.5
        )

        if condition_number > 1e10:
            diagnosis = 'SINGULAR_FIM'
        elif smallest < 1e-10:
            diagnosis = 'BARREN_PLATEAU'
        elif condition_number > 1e6:
            diagnosis = 'ILL_CONDITIONED'
        else:
            diagnosis = 'HEALTHY'

        return {
            'condition_number': condition_number,
            'rank': rank,
            'smallest_eigenvalue': smallest,
            'largest_eigenvalue': largest,
            'trace': trace,
            'is_healthy': is_healthy,
            'diagnosis': diagnosis,
        }

class QuantumNaturalGradientOptimizer(Optimizer):

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        config: QNGConfig = None,
    ):
        config = config or QNGConfig()
        defaults = dict(
            lr=lr,
            approx_method=config.approx_method,
            regularization=config.regularization,
            damping=config.damping,
        )
        super().__init__(params, defaults)

        self.config = config
        self._fim_computers = {}

    def _get_block_diagonal_fim(
        self,
        param: Tensor,
        block_size: int = 4,
    ) -> Tensor:

        n = param.numel()
        fim = torch.zeros(n, n)

        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            size = end - i

            block = torch.eye(size)

            for j in range(size):
                for k in range(j + 1, size):
                    corr = 0.5 * np.exp(-abs(j - k) / 2)
                    block[j, k] = corr
                    block[k, j] = corr

            fim[i:end, i:end] = block

        return fim

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.view(-1)
                n = grad.numel()

                if group['approx_method'] == 'diagonal':

                    fim_inv = torch.ones(n) / (grad.abs() + group['damping'])
                    natural_grad = fim_inv * grad

                elif group['approx_method'] == 'block_diagonal':

                    fim = self._get_block_diagonal_fim(p.data)
                    fim = fim + group['regularization'] * torch.eye(n)

                    try:
                        natural_grad = torch.linalg.solve(fim, grad)
                    except RuntimeError:

                        natural_grad = torch.linalg.lstsq(fim, grad).solution

                else:

                    fim = torch.eye(n)
                    natural_grad = torch.linalg.solve(
                        fim + group['regularization'] * torch.eye(n),
                        grad
                    )

                p.add_(natural_grad.view_as(p), alpha=-group['lr'])

        return loss

class OptimizerComparison:

    @staticmethod
    def compare_convergence(
        model: nn.Module,
        data: Tuple[Tensor, Tensor],
        optimizers: Dict[str, Optimizer],
        n_steps: int = 100,
    ) -> Dict[str, List[float]]:

        inputs, targets = data
        criterion = nn.MSELoss()

        results = {}

        for name, opt in optimizers.items():

            for param in model.parameters():
                nn.init.normal_(param, mean=0, std=0.1)

            losses = []

            for step in range(n_steps):
                opt.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())

            results[name] = losses

        return results

    @staticmethod
    def analyze_qng_advantage(
        loss_histories: Dict[str, List[float]],
    ) -> Dict[str, Dict]:

        analysis = {}

        for name, losses in loss_histories.items():
            losses = np.array(losses)

            initial = losses[0]
            final = losses[-1]
            threshold = initial - 0.9 * (initial - final)

            convergence_step = len(losses)
            for i, l in enumerate(losses):
                if l <= threshold:
                    convergence_step = i
                    break

            tail = losses[int(0.8 * len(losses)):]
            stability = np.std(tail)

            analysis[name] = {
                'initial_loss': initial,
                'final_loss': final,
                'improvement': (initial - final) / initial * 100,
                'convergence_step': convergence_step,
                'stability': stability,
            }

        if 'QNG' in analysis and 'Adam' in analysis:
            analysis['qng_vs_adam'] = {
                'loss_ratio': analysis['QNG']['final_loss'] / analysis['Adam']['final_loss'],
                'speed_ratio': analysis['Adam']['convergence_step'] / max(1, analysis['QNG']['convergence_step']),
            }

        return analysis

def detect_barren_plateau(
    circuit_fn: Callable,
    n_params: int,
    n_samples: int = 100,
    threshold: float = 1e-6,
) -> Dict[str, any]:

    gradient_variances = []

    for _ in range(n_samples):
        params = torch.randn(n_params) * 2 * np.pi

        gradients = torch.zeros(n_params)
        for i in range(n_params):
            params_plus = params.clone()
            params_plus[i] += np.pi / 2
            f_plus = circuit_fn(params_plus)

            params_minus = params.clone()
            params_minus[i] -= np.pi / 2
            f_minus = circuit_fn(params_minus)

            if isinstance(f_plus, Tensor):
                grad = (f_plus - f_minus).mean() / 2
            else:
                grad = (f_plus - f_minus) / 2

            gradients[i] = grad

        gradient_variances.append(gradients.var().item())

    mean_var = np.mean(gradient_variances)

    is_barren = mean_var < threshold

    return {
        'is_barren_plateau': is_barren,
        'gradient_variance': mean_var,
        'threshold': threshold,
        'severity': 'SEVERE' if mean_var < threshold / 100 else
                   'MODERATE' if mean_var < threshold else 'NONE',
        'recommendation': 'Use local cost functions or shallow circuits' if is_barren else 'Training should proceed normally',
    }
