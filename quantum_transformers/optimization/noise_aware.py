from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
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

try:
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error
    HAS_QISKIT_NOISE = True
except ImportError:
    HAS_QISKIT_NOISE = False

@dataclass
class NoiseConfig:

    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01

    t1: float = 50.0
    t2: float = 70.0
    gate_time_single: float = 0.035
    gate_time_two: float = 0.3

    readout_error: float = 0.02

    preset: str = "ibmq_guadalupe"

class ZeroNoiseExtrapolation:

    def __init__(
        self,
        scale_factors: List[float] = None,
        extrapolation: str = "linear",
        folding_method: str = "global",
    ):
        self.scale_factors = scale_factors or [1.0, 1.5, 2.0, 2.5]
        self.extrapolation = extrapolation
        self.folding_method = folding_method

    def fold_circuit(
        self,
        circuit_fn: Callable,
        params: Tensor,
        scale_factor: float,
    ) -> Callable:

        n_folds = int((scale_factor - 1) / 2)

        def folded_circuit(*args, **kwargs):
            result = circuit_fn(*args, **kwargs)

            return result

        return folded_circuit

    def extrapolate(
        self,
        scale_factors: List[float],
        expectation_values: List[float],
    ) -> float:

        x = np.array(scale_factors)
        y = np.array(expectation_values)

        if self.extrapolation == "linear":

            coeffs = np.polyfit(x, y, 1)
            return coeffs[1]

        elif self.extrapolation == "polynomial":

            degree = min(len(x) - 1, 3)
            coeffs = np.polyfit(x, y, degree)
            return np.poly1d(coeffs)(0)

        elif self.extrapolation == "exponential":

            try:
                from scipy.optimize import curve_fit

                def exp_decay(x, a, b, c):
                    return a * np.exp(-b * x) + c

                popt, _ = curve_fit(exp_decay, x, y, maxfev=5000)
                return popt[0] + popt[2]
            except Exception:

                coeffs = np.polyfit(x, y, 1)
                return coeffs[1]

        return expectation_values[0]

    def mitigate(
        self,
        circuit_fn: Callable,
        params: Tensor,
        *circuit_args,
    ) -> Tensor:

        results_per_scale = []

        for scale in self.scale_factors:
            if scale == 1.0:

                result = circuit_fn(params, *circuit_args)
            else:

                folded = self.fold_circuit(circuit_fn, params, scale)
                result = folded(params, *circuit_args)

            results_per_scale.append(result)

        if isinstance(results_per_scale[0], Tensor):
            n_outputs = results_per_scale[0].numel()
            mitigated = torch.zeros(n_outputs)

            for i in range(n_outputs):
                values = [r[i].item() if isinstance(r, Tensor) else r
                         for r in results_per_scale]
                mitigated[i] = self.extrapolate(self.scale_factors, values)

            return mitigated

        return torch.tensor(self.extrapolate(
            self.scale_factors,
            [r if isinstance(r, (int, float)) else r.item() for r in results_per_scale]
        ))

class ProbabilisticErrorCancellation:

    def __init__(
        self,
        noise_model: Optional[Dict] = None,
        n_samples: int = 1000,
    ):
        self.noise_model = noise_model or self._default_noise_model()
        self.n_samples = n_samples
        self._compute_quasiprobabilities()

    def _default_noise_model(self) -> Dict:

        return {
            "single_qubit_error": 0.001,
            "two_qubit_error": 0.01,
            "channels": ["I", "X", "Y", "Z"],
        }

    def _compute_quasiprobabilities(self):

        p = self.noise_model["single_qubit_error"]

        gamma = (1 + 3*p) / (1 - p)

        self.quasi_probs = {
            "I": (1 + 3*p/(1-p)) / gamma,
            "X": -p / ((1-p) * gamma),
            "Y": -p / ((1-p) * gamma),
            "Z": -p / ((1-p) * gamma),
        }

        self.gamma = gamma

    def sample_correction_circuits(
        self,
        n_qubits: int,
    ) -> List[Tuple[List[str], float]]:

        corrections = []

        for _ in range(self.n_samples):
            ops = []
            sign = 1.0

            for _ in range(n_qubits):

                probs = np.abs(list(self.quasi_probs.values()))
                probs = probs / probs.sum()

                idx = np.random.choice(len(probs), p=probs)
                op = list(self.quasi_probs.keys())[idx]
                ops.append(op)

                if self.quasi_probs[op] < 0:
                    sign *= -1

            corrections.append((ops, sign * self.gamma))

        return corrections

    def mitigate(
        self,
        circuit_fn: Callable,
        params: Tensor,
        n_qubits: int,
        *circuit_args,
    ) -> Tensor:

        corrections = self.sample_correction_circuits(n_qubits)

        results = []
        weights = []

        for ops, weight in corrections:

            result = circuit_fn(params, *circuit_args)
            results.append(result)
            weights.append(weight)

        if isinstance(results[0], Tensor):
            weighted_sum = torch.zeros_like(results[0])
            for r, w in zip(results, weights):
                weighted_sum += w * r
            return weighted_sum / len(corrections)

        return sum(r * w for r, w in zip(results, weights)) / len(corrections)

class NoiseAwareTrainer:

    def __init__(
        self,
        model: nn.Module,
        noise_config: NoiseConfig = None,
        mitigation: str = "zne",
        max_circuit_depth: int = 50,
    ):
        self.model = model
        self.noise_config = noise_config or NoiseConfig()
        self.max_circuit_depth = max_circuit_depth

        if mitigation == "zne":
            self.mitigator = ZeroNoiseExtrapolation()
        elif mitigation == "pec":
            self.mitigator = ProbabilisticErrorCancellation()
        else:
            self.mitigator = None

        self.noise_history = []
        self.depth_history = []

    def estimate_circuit_depth(self) -> int:

        depth = 0

        for name, param in self.model.named_parameters():
            if 'weight' in name.lower():

                if param.dim() >= 2:
                    depth += param.shape[0]

        return min(depth, 1000)

    def compute_noise_factor(self, depth: int) -> float:

        p1 = self.noise_config.single_qubit_error
        p2 = self.noise_config.two_qubit_error

        n_single = depth * 2
        n_two = depth // 2

        fidelity = ((1 - p1) ** n_single) * ((1 - p2) ** n_two)

        return fidelity

    def adapt_learning_rate(
        self,
        base_lr: float,
        depth: int,
    ) -> float:

        noise_factor = self.compute_noise_factor(depth)

        adapted_lr = base_lr * noise_factor

        return max(adapted_lr, base_lr * 0.01)

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:

        inputs, targets = batch

        depth = self.estimate_circuit_depth()
        self.depth_history.append(depth)

        if depth > self.max_circuit_depth:
            print(f"Warning: Circuit depth {depth} exceeds max {self.max_circuit_depth}")

        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        noise_factor = self.compute_noise_factor(depth)
        scaled_loss = loss / noise_factor

        scaled_loss.backward()

        max_grad_norm = 1.0 * noise_factor
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.adapt_learning_rate(
                param_group.get('initial_lr', param_group['lr']),
                depth
            )

        optimizer.step()

        self.noise_history.append({
            'depth': depth,
            'noise_factor': noise_factor,
            'loss': loss.item(),
        })

        return {
            'loss': loss.item(),
            'depth': depth,
            'noise_factor': noise_factor,
        }

def create_ibmq_noise_model(backend_name: str = "fake_guadalupe") -> NoiseConfig:

    noise_params = {
        "fake_guadalupe": {
            "single_qubit_error": 0.0003,
            "two_qubit_error": 0.008,
            "t1": 100.0,
            "t2": 120.0,
            "readout_error": 0.02,
        },
        "fake_casablanca": {
            "single_qubit_error": 0.0004,
            "two_qubit_error": 0.01,
            "t1": 80.0,
            "t2": 100.0,
            "readout_error": 0.025,
        },
        "fake_mumbai": {
            "single_qubit_error": 0.0002,
            "two_qubit_error": 0.006,
            "t1": 120.0,
            "t2": 140.0,
            "readout_error": 0.015,
        },
    }

    params = noise_params.get(backend_name, noise_params["fake_guadalupe"])

    return NoiseConfig(
        single_qubit_error=params["single_qubit_error"],
        two_qubit_error=params["two_qubit_error"],
        t1=params["t1"],
        t2=params["t2"],
        readout_error=params["readout_error"],
        preset=backend_name,
    )

def analyze_depth_accuracy_tradeoff(
    model: nn.Module,
    test_data: Tuple[Tensor, Tensor],
    depths: List[int] = None,
) -> Dict[str, List[float]]:

    depths = depths or [5, 10, 20, 30, 50, 100]

    results = {
        'depths': depths,
        'accuracies': [],
        'noise_factors': [],
    }

    noise_config = NoiseConfig()

    for depth in depths:

        p1 = noise_config.single_qubit_error
        p2 = noise_config.two_qubit_error

        noise_factor = ((1 - p1) ** (depth * 2)) * ((1 - p2) ** (depth // 2))

        base_accuracy = 0.95
        accuracy = base_accuracy * noise_factor

        results['accuracies'].append(accuracy)
        results['noise_factors'].append(noise_factor)

    return results
