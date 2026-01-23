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

class VariationalLayer(nn.Module):

    def __init__(self, n_qubits: int = 4, entanglement: str = "circular"):
        super().__init__()

        self.n_qubits = n_qubits
        self.entanglement = entanglement

        self.rotations = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)

    def apply(self, wires: list):

        for i, w in enumerate(wires):
            qml.Rot(
                self.rotations[i, 0],
                self.rotations[i, 1],
                self.rotations[i, 2],
                wires=w
            )

        n = len(wires)
        if self.entanglement == "circular":
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
        elif self.entanglement == "linear":
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif self.entanglement == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    qml.CZ(wires=[wires[i], wires[j]])

class QuantumFeedForward(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        n_qubits: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        entanglement: str = "circular",
    ):
        super().__init__()

        self.d_model = d_model
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(d_model, n_qubits),
            nn.Tanh(),
        )

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, d_model),
            nn.Dropout(dropout),
        )

        self.entanglement = entanglement

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def ffn_circuit(inputs, weights):

            for i in range(self.n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i
                    )

                if self.entanglement == "circular":
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                else:
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = ffn_circuit

    def forward(self, x: Tensor) -> Tensor:

        original_shape = x.shape
        batch_size = x.size(0)

        if len(original_shape) == 3:
            seq_len = x.size(1)
            x = x.view(-1, self.d_model)
            batch_size = x.size(0)
        else:
            seq_len = None

        x_proj = self.input_proj(x)

        if HAS_PENNYLANE:
            outputs = []
            for b in range(batch_size):
                q_out = self.qnode(x_proj[b], self.weights)
                outputs.append(torch.stack(q_out))
            output = torch.stack(outputs).float()
        else:

            output = torch.tanh(x_proj)

        output = self.output_proj(output)

        if seq_len is not None:
            output = output.view(original_shape[0], seq_len, -1)

        return output

class QuantumMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        n_qubits: int = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(
                QuantumFeedForward(
                    d_model=dims[i],
                    n_qubits=n_qubits,
                    n_layers=2,
                )
            )
            if i < len(dims) - 2:
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
