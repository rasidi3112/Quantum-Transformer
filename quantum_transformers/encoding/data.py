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

class QuantumAmplitudeEncoding(nn.Module):

    def __init__(self, n_qubits: int = 4, normalize: bool = True):
        super().__init__()

        self.n_qubits = n_qubits
        self.max_features = 2 ** n_qubits
        self.normalize = normalize

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch')
        def amplitude_encoding(features):
            qml.AmplitudeEmbedding(
                features,
                wires=range(self.n_qubits),
                normalize=self.normalize,
                pad_with=0.0,
            )
            return qml.state()

        self.encode = amplitude_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.size(0)

        if not HAS_PENNYLANE:

            return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        encoded = []
        for b in range(batch_size):

            features = x[b, :self.max_features]
            if len(features) < self.max_features:
                features = torch.nn.functional.pad(
                    features, (0, self.max_features - len(features))
                )

            state = self.encode(features)
            encoded.append(state)

        return torch.stack(encoded)

class QuantumAngleEncoding(nn.Module):

    def __init__(self, n_qubits: int = 4, rotation: str = "RY"):
        super().__init__()

        self.n_qubits = n_qubits
        self.rotation = rotation

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        gate = getattr(qml, self.rotation)

        @qml.qnode(self.dev, interface='torch')
        def angle_encoding(features):
            for i in range(self.n_qubits):
                if i < len(features):
                    gate(features[i] * np.pi, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.encode = angle_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.size(0)

        if not HAS_PENNYLANE:
            return torch.cos(x[..., :self.n_qubits] * np.pi)

        encoded = []
        for b in range(batch_size):
            features = torch.tanh(x[b, :self.n_qubits])
            out = self.encode(features)
            encoded.append(torch.stack(out))

        return torch.stack(encoded)

class QuantumBasisEncoding(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch')
        def basis_encoding(bits):
            for i in range(self.n_qubits):
                if bits[i] > 0.5:
                    qml.PauliX(wires=i)
            return qml.state()

        self.encode = basis_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.size(0)

        if not HAS_PENNYLANE:

            return torch.eye(2**self.n_qubits)[x.long() % (2**self.n_qubits)]

        encoded = []
        for b in range(batch_size):

            val = int(x[b].item()) % (2 ** self.n_qubits)
            bits = torch.tensor(
                [(val >> i) & 1 for i in range(self.n_qubits)],
                dtype=torch.float32
            )
            state = self.encode(bits)
            encoded.append(state)

        return torch.stack(encoded)

class QuantumIQPEncoding(nn.Module):

    def __init__(self, n_qubits: int = 4, n_reps: int = 2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_reps = n_reps

        if HAS_PENNYLANE:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def iqp_encoding(features):
            for _ in range(self.n_reps):

                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)

                for i in range(self.n_qubits):
                    if i < len(features):
                        qml.RZ(features[i], wires=i)

                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        if i < len(features) and j < len(features):
                            angle = features[i] * features[j]
                            qml.CNOT(wires=[i, j])
                            qml.RZ(angle, wires=j)
                            qml.CNOT(wires=[i, j])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.encode = iqp_encoding

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.size(0)

        if not HAS_PENNYLANE:

            return torch.cos(x[..., :self.n_qubits])

        encoded = []
        for b in range(batch_size):
            features = x[b, :self.n_qubits]
            out = self.encode(features)
            encoded.append(torch.stack(out))

        return torch.stack(encoded)
