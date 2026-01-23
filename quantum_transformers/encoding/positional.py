from __future__ import annotations
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

class QuantumPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class QuantumSinusoidalEncoding(QuantumPositionalEncoding):

    def __init__(self, d_model: int, max_len: int = 512, n_qubits: int = 4):
        super().__init__(d_model, max_len)

        self.n_qubits = n_qubits

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_encoding_circuit()

    def _create_encoding_circuit(self):
        @qml.qnode(self.dev, interface='torch')
        def encode_position(pos_embedding):

            for i in range(self.n_qubits):
                angle = pos_embedding[i % len(pos_embedding)] * np.pi
                qml.RY(angle, wires=i)
                qml.RZ(angle / 2, wires=i)

            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.encode_circuit = encode_position

    def forward(self, x: Tensor) -> Tensor:

        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

    def get_quantum_encoding(self, positions: Tensor) -> Tensor:

        if not HAS_PENNYLANE:
            return self.pe[positions]

        encodings = []
        for pos in positions:
            pe = self.pe[pos]
            q_enc = self.encode_circuit(pe[:self.n_qubits])
            encodings.append(torch.stack(q_enc))

        return torch.stack(encodings)

class QuantumRotationalEncoding(QuantumPositionalEncoding):

    def __init__(self, d_model: int, max_len: int = 512, n_qubits: int = 4):
        super().__init__(d_model, max_len)

        self.n_qubits = n_qubits

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

        self.theta = nn.Parameter(torch.randn(max_len, n_qubits) * 0.1)

        self.projection = nn.Linear(n_qubits, d_model)

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def rotational_encoding(theta):
            for i in range(self.n_qubits):
                qml.RY(theta[i], wires=i)

            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = rotational_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size, seq_len, _ = x.shape

        if HAS_PENNYLANE:
            pos_enc = []
            for pos in range(seq_len):
                q_out = self.qnode(self.theta[pos])
                pos_enc.append(torch.stack(q_out))
            pos_enc = torch.stack(pos_enc)
            pos_enc = self.projection(pos_enc.float())
        else:
            pos_enc = self.projection(torch.tanh(self.theta[:seq_len]))

        return x + pos_enc.unsqueeze(0)

class LearnableQuantumPositionalEncoding(QuantumPositionalEncoding):

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        n_qubits: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(d_model, max_len)

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.weights = nn.Parameter(
            torch.randn(max_len, n_layers, n_qubits, 3) * 0.1
        )

        self.projection = nn.Linear(n_qubits, d_model)

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def learnable_encoding(position_idx, weights):

            for i in range(self.n_qubits):
                qml.RY(position_idx * np.pi / self.max_len, wires=i)

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i
                    )
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = learnable_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size, seq_len, _ = x.shape

        if HAS_PENNYLANE:
            pos_enc = []
            for pos in range(seq_len):
                q_out = self.qnode(float(pos), self.weights[pos])
                pos_enc.append(torch.stack(q_out))
            pos_enc = torch.stack(pos_enc)
            pos_enc = self.projection(pos_enc.float())
        else:

            pos_enc = torch.zeros(seq_len, self.d_model)
            for pos in range(seq_len):
                pos_enc[pos] = self.projection(
                    torch.tanh(self.weights[pos].mean(dim=(0, 1)))
                )

        return x + pos_enc.unsqueeze(0)
