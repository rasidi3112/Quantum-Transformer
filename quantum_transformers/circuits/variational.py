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

class DataEncodingLayer(nn.Module):

    def __init__(self, n_qubits: int, encoding: str = "angle"):
        super().__init__()

        self.n_qubits = n_qubits
        self.encoding = encoding

    def apply(self, data: Tensor, wires: list):

        if self.encoding == "angle":
            for i, w in enumerate(wires):
                if i < len(data):
                    qml.RY(data[i] * np.pi, wires=w)
        elif self.encoding == "amplitude":
            qml.AmplitudeEmbedding(data, wires=wires, normalize=True)
        elif self.encoding == "iqp":
            for w in wires:
                qml.Hadamard(wires=w)
            for i, w in enumerate(wires):
                if i < len(data):
                    qml.RZ(data[i], wires=w)

class EntanglingLayer(nn.Module):

    def __init__(self, n_qubits: int, pattern: str = "circular"):
        super().__init__()

        self.n_qubits = n_qubits
        self.pattern = pattern

    def apply(self, wires: list):

        n = len(wires)

        if self.pattern == "circular":
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
        elif self.pattern == "linear":
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif self.pattern == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    qml.CZ(wires=[wires[i], wires[j]])
        elif self.pattern == "pairwise":
            for i in range(0, n - 1, 2):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

class StronglyEntanglingLayer(nn.Module):

    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

    def apply(self, wires: list):

        if HAS_PENNYLANE:
            qml.StronglyEntanglingLayers(
                self.weights.detach().numpy(),
                wires=wires
            )

class VariationalCircuit(nn.Module):

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        entanglement: str = "circular",
    ):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.encoder = DataEncodingLayer(n_qubits, encoding)
        self.entangler = EntanglingLayer(n_qubits, entanglement)

        self.dev = qml.device('default.qubit', wires=n_qubits)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def variational_circuit(inputs, weights):
            wires = list(range(self.n_qubits))

            for i in range(self.n_qubits):
                if i < len(inputs):
                    qml.RY(inputs[i] * np.pi, wires=i)

            for layer in range(self.n_layers):

                for i in range(self.n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i
                    )

                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = variational_circuit

    def forward(self, x: Tensor) -> Tensor:

        x = torch.tanh(x[..., :self.n_qubits])

        if x.dim() == 1:
            out = self.circuit(x, self.weights)
            return torch.stack(out)

        outputs = []
        for sample in x:
            out = self.circuit(sample, self.weights)
            outputs.append(torch.stack(out))

        return torch.stack(outputs).float()

    def draw(self, sample_input: Tensor = None) -> str:

        if sample_input is None:
            sample_input = torch.zeros(self.n_qubits)
        return qml.draw(self.circuit)(sample_input, self.weights)
