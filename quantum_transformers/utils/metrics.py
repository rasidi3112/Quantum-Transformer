from __future__ import annotations
from typing import Callable, List, Optional
import math
import numpy as np
import torch
from torch import Tensor

def quantum_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:

    state1 = state1 / np.linalg.norm(state1)
    state2 = state2 / np.linalg.norm(state2)
    return np.abs(np.vdot(state1, state2)) ** 2

def expressibility(
    circuit: Callable,
    n_params: int,
    n_samples: int = 1000,
    n_qubits: int = 4,
) -> float:

    fidelities = []

    for _ in range(n_samples):
        params1 = np.random.uniform(0, 2 * np.pi, n_params)
        params2 = np.random.uniform(0, 2 * np.pi, n_params)

        try:
            state1 = circuit(params1)
            state2 = circuit(params2)
            f = quantum_fidelity(state1, state2)
            fidelities.append(f)
        except Exception:
            continue

    if not fidelities:
        return float('inf')

    fidelities = np.array(fidelities)
    dim = 2 ** n_qubits

    hist, bins = np.histogram(fidelities, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    haar_pdf = (dim - 1) * np.maximum(1e-10, (1 - bin_centers) ** (dim - 2))
    haar_pdf = haar_pdf / np.sum(haar_pdf)

    hist = hist / np.sum(hist) + 1e-10
    kl_div = np.sum(hist * np.log(hist / (haar_pdf + 1e-10)))

    return kl_div

def entanglement_capability(
    circuit: Callable,
    n_qubits: int,
    n_params: int,
    n_samples: int = 100,
) -> float:

    entanglements = []

    for _ in range(n_samples):
        params = np.random.uniform(0, 2 * np.pi, n_params)

        try:
            state = circuit(params)
            mw = _meyer_wallach(state, n_qubits)
            entanglements.append(mw)
        except Exception:
            continue

    return np.mean(entanglements) if entanglements else 0.0

def _meyer_wallach(state: np.ndarray, n_qubits: int) -> float:

    Q = 0.0

    for k in range(n_qubits):
        rho_k = _partial_trace_single(state, k, n_qubits)
        Q += 2 * (1 - np.real(np.trace(rho_k @ rho_k)))

    return Q / n_qubits

def _partial_trace_single(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:

    dim = 2 ** n_qubits
    rho = np.outer(state, np.conj(state))
    rho_reduced = np.zeros((2, 2), dtype=complex)

    for i in range(2):
        for j in range(2):
            for k in range(dim // 2):
                idx1 = (k & ((1 << qubit) - 1)) | (i << qubit) | ((k >> qubit) << (qubit + 1))
                idx2 = (k & ((1 << qubit) - 1)) | (j << qubit) | ((k >> qubit) << (qubit + 1))
                rho_reduced[i, j] += rho[idx1, idx2]

    return rho_reduced

def attention_entropy(attention_weights: Tensor) -> Tensor:

    eps = 1e-10
    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + eps),
        dim=-1
    )
    return entropy.mean(dim=-1)

def circuit_depth_analysis(model) -> dict:

    stats = {
        "n_qubits": 0,
        "n_layers": 0,
        "total_gates": 0,
        "single_qubit_gates": 0,
        "two_qubit_gates": 0,
    }

    if hasattr(model, 'config'):
        config = model.config
        stats["n_qubits"] = getattr(config, 'n_qubits', 0)
        stats["n_layers"] = getattr(config, 'n_layers', 0)

        n_q = stats["n_qubits"]
        n_l = stats["n_layers"]

        stats["single_qubit_gates"] = n_l * n_q * 3
        stats["two_qubit_gates"] = n_l * (n_q - 1)
        stats["total_gates"] = stats["single_qubit_gates"] + stats["two_qubit_gates"]

    return stats

def gradient_variance(gradients: List[np.ndarray]) -> float:

    gradients = np.array(gradients)
    return np.mean(np.var(gradients, axis=0))

def parameter_efficiency(model, accuracy: float) -> float:

    n_params = sum(p.numel() for p in model.parameters())
    return accuracy / math.log(n_params + 1)
