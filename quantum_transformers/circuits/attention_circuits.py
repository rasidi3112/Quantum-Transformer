from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

class SwapTestCircuit(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits

        self.total_wires = 1 + 2 * n_qubits

        self.dev = qml.device('default.qubit', wires=self.total_wires)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def swap_test_circuit(psi_params, phi_params):
            ancilla = 0
            psi_wires = list(range(1, self.n_qubits + 1))
            phi_wires = list(range(self.n_qubits + 1, self.total_wires))

            qml.Hadamard(wires=ancilla)

            for i, w in enumerate(psi_wires):
                qml.RY(psi_params[i], wires=w)

            for i, w in enumerate(phi_wires):
                qml.RY(phi_params[i], wires=w)

            for i in range(self.n_qubits):
                qml.CSWAP(wires=[ancilla, psi_wires[i], phi_wires[i]])

            qml.Hadamard(wires=ancilla)

            return qml.expval(qml.PauliZ(ancilla))

        self.circuit = swap_test_circuit

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:

        psi = torch.tanh(psi[..., :self.n_qubits]) * np.pi
        phi = torch.tanh(phi[..., :self.n_qubits]) * np.pi

        if psi.dim() == 1:
            result = self.circuit(psi, phi)

            return (1 + result) / 2

        results = []
        for p, q in zip(psi, phi):
            r = self.circuit(p, q)
            results.append((1 + r) / 2)

        return torch.stack(results)

class InnerProductCircuit(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def inner_product_circuit(psi_params, phi_params):
            wires = list(range(self.n_qubits))

            for i in range(self.n_qubits):
                qml.RY(psi_params[i], wires=i)

            for i in range(self.n_qubits - 1, -1, -1):
                qml.RY(-phi_params[i], wires=i)

            return qml.probs(wires=wires)

        self.circuit = inner_product_circuit

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:

        psi = torch.tanh(psi[..., :self.n_qubits]) * np.pi
        phi = torch.tanh(phi[..., :self.n_qubits]) * np.pi

        probs = self.circuit(psi, phi)

        return probs[0]

class QuantumDotProductCircuit(nn.Module):

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.W_q = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.W_k = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        self.dev = qml.device('default.qubit', wires=2*n_qubits + 1)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def dot_product_circuit(q, k, W_q, W_k):
            ancilla = 0
            q_wires = list(range(1, self.n_qubits + 1))
            k_wires = list(range(self.n_qubits + 1, 2*self.n_qubits + 1))

            for i, w in enumerate(q_wires):
                qml.RY(q[i] * np.pi, wires=w)

            for layer in range(self.n_layers):
                for i, w in enumerate(q_wires):
                    qml.Rot(W_q[layer, i, 0], W_q[layer, i, 1], W_q[layer, i, 2], wires=w)
                for i in range(len(q_wires) - 1):
                    qml.CNOT(wires=[q_wires[i], q_wires[i+1]])

            for i, w in enumerate(k_wires):
                qml.RY(k[i] * np.pi, wires=w)

            for layer in range(self.n_layers):
                for i, w in enumerate(k_wires):
                    qml.Rot(W_k[layer, i, 0], W_k[layer, i, 1], W_k[layer, i, 2], wires=w)
                for i in range(len(k_wires) - 1):
                    qml.CNOT(wires=[k_wires[i], k_wires[i+1]])

            qml.Hadamard(wires=ancilla)
            for i in range(self.n_qubits):
                qml.CSWAP(wires=[ancilla, q_wires[i], k_wires[i]])
            qml.Hadamard(wires=ancilla)

            return qml.expval(qml.PauliZ(ancilla))

        self.circuit = dot_product_circuit

    def forward(self, query: Tensor, key: Tensor) -> Tensor:

        query = torch.tanh(query[..., :self.n_qubits])
        key = torch.tanh(key[..., :self.n_qubits])

        score = self.circuit(query, key, self.W_q, self.W_k)

        return (1 + score) / 2
