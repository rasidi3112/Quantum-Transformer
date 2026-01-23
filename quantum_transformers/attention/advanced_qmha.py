from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

@dataclass
class AdvancedQMHAConfig:

    n_qubits: int = 4
    n_heads: int = 4
    n_layers: int = 2
    d_model: int = 64
    attention_type: str = "entanglement"
    data_reuploading: bool = True
    n_reuploading_layers: int = 3
    entanglement_pattern: str = "circular"
    use_approximation: bool = False
    locality_strength: float = 0.5

class DataReuploadingEncoder(nn.Module):

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        encoding_type: str = "angle",
        use_locality: bool = True,
    ):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.use_locality = use_locality

        if use_locality:

            self.weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.01
            )
        else:
            self.weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.5
            )

        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))

        self.dev = qml.device('default.qubit', wires=n_qubits)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def reuploading_circuit(x, weights, scaling):
            wires = list(range(self.n_qubits))

            for layer in range(self.n_layers):

                for i, w in enumerate(wires):
                    if i < len(x):
                        angle = x[i] * scaling[layer, i] * np.pi
                        qml.RY(angle, wires=w)
                        qml.RZ(angle / 2, wires=w)

                for i, w in enumerate(wires):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=w
                    )

                for i in range(len(wires)):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])

            return [qml.expval(qml.PauliZ(w)) for w in wires]

        self.circuit = reuploading_circuit

    def forward(self, x: Tensor) -> Tensor:

        x = torch.tanh(x[..., :self.n_qubits])

        if x.dim() == 1:
            out = self.circuit(x, self.weights, self.input_scaling)
            return torch.stack(out)

        outputs = []
        for sample in x:
            out = self.circuit(sample, self.weights, self.input_scaling)
            outputs.append(torch.stack(out))

        return torch.stack(outputs).float()

class EntanglementBasedAttention(nn.Module):

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        use_data_reuploading: bool = True,
    ):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if use_data_reuploading:
            self.query_encoder = DataReuploadingEncoder(n_qubits, n_layers)
            self.key_encoder = DataReuploadingEncoder(n_qubits, n_layers)

        self.ent_weights = nn.Parameter(
            torch.randn(n_qubits, 3) * 0.1
        )

        self.dev = qml.device('default.qubit', wires=2 * n_qubits)
        self._create_attention_circuit()

    def _create_attention_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def entanglement_attention(q_data, k_data, q_weights, k_weights, ent_weights):
            q_wires = list(range(self.n_qubits))
            k_wires = list(range(self.n_qubits, 2 * self.n_qubits))

            for layer in range(self.n_layers):
                for i, w in enumerate(q_wires):
                    if i < len(q_data):
                        qml.RY(q_data[i] * np.pi, wires=w)
                    qml.Rot(
                        q_weights[layer, i, 0],
                        q_weights[layer, i, 1],
                        q_weights[layer, i, 2],
                        wires=w
                    )
                for i in range(len(q_wires) - 1):
                    qml.CNOT(wires=[q_wires[i], q_wires[i + 1]])

            for layer in range(self.n_layers):
                for i, w in enumerate(k_wires):
                    if i < len(k_data):
                        qml.RY(k_data[i] * np.pi, wires=w)
                    qml.Rot(
                        k_weights[layer, i, 0],
                        k_weights[layer, i, 1],
                        k_weights[layer, i, 2],
                        wires=w
                    )
                for i in range(len(k_wires) - 1):
                    qml.CNOT(wires=[k_wires[i], k_wires[i + 1]])

            for i in range(self.n_qubits):
                qml.Rot(
                    ent_weights[i, 0],
                    ent_weights[i, 1],
                    ent_weights[i, 2],
                    wires=q_wires[i]
                )
                qml.CNOT(wires=[q_wires[i], k_wires[i]])

            return [
                qml.expval(qml.PauliZ(q_wires[i]) @ qml.PauliZ(k_wires[i]))
                for i in range(self.n_qubits)
            ]

        self.circuit = entanglement_attention

    def forward(self, query: Tensor, key: Tensor) -> Tensor:

        query = torch.tanh(query[..., :self.n_qubits])
        key = torch.tanh(key[..., :self.n_qubits])

        q_weights = self.query_encoder.weights if hasattr(self, 'query_encoder') else torch.zeros(self.n_layers, self.n_qubits, 3)
        k_weights = self.key_encoder.weights if hasattr(self, 'key_encoder') else torch.zeros(self.n_layers, self.n_qubits, 3)

        if query.dim() == 1:
            correlations = self.circuit(
                query, key, q_weights, k_weights, self.ent_weights
            )

            score = (torch.stack(correlations).mean() + 1) / 2
            return score

        batch_size, seq_len_q = query.shape[0], query.shape[1] if query.dim() > 2 else 1
        seq_len_k = key.shape[1] if key.dim() > 2 else 1

        scores = torch.zeros(batch_size, seq_len_q, seq_len_k)

        for b in range(batch_size):
            for i in range(seq_len_q):
                for j in range(seq_len_k):
                    q_i = query[b, i] if query.dim() > 2 else query[b]
                    k_j = key[b, j] if key.dim() > 2 else key[b]

                    correlations = self.circuit(
                        q_i, k_j, q_weights, k_weights, self.ent_weights
                    )
                    score = (torch.stack(correlations).mean() + 1) / 2
                    scores[b, i, j] = score

        return scores

class VariationalQuantumAttention(nn.Module):

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        use_crz: bool = True,
    ):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_crz = use_crz

        self.W_q_y = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.W_q_z = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        self.W_k_y = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.W_k_z = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        self.W_int = nn.Parameter(torch.randn(n_qubits) * 0.1)

        self.dev = qml.device('default.qubit', wires=2 * n_qubits)
        self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def vqa_circuit(q_data, k_data, w_q_y, w_q_z, w_k_y, w_k_z, w_int):
            q_wires = list(range(self.n_qubits))
            k_wires = list(range(self.n_qubits, 2 * self.n_qubits))

            for layer in range(self.n_layers):
                for i, w in enumerate(q_wires):
                    if i < len(q_data):

                        qml.RY(q_data[i] * np.pi, wires=w)

                    qml.RY(w_q_y[layer, i], wires=w)
                    qml.RZ(w_q_z[layer, i], wires=w)

                for i in range(len(q_wires) - 1):
                    qml.CNOT(wires=[q_wires[i], q_wires[i + 1]])

            for layer in range(self.n_layers):
                for i, w in enumerate(k_wires):
                    if i < len(k_data):
                        qml.RY(k_data[i] * np.pi, wires=w)
                    qml.RY(w_k_y[layer, i], wires=w)
                    qml.RZ(w_k_z[layer, i], wires=w)

                for i in range(len(k_wires) - 1):
                    qml.CNOT(wires=[k_wires[i], k_wires[i + 1]])

            for i in range(self.n_qubits):
                qml.CRY(w_int[i], wires=[q_wires[i], k_wires[i]])
                if self.use_crz:
                    qml.CRZ(w_int[i] / 2, wires=[k_wires[i], q_wires[i]])

            return qml.probs(wires=q_wires[:2])

        self.circuit = vqa_circuit

    def forward(self, query: Tensor, key: Tensor) -> Tensor:

        query = torch.tanh(query[..., :self.n_qubits])
        key = torch.tanh(key[..., :self.n_qubits])

        if query.dim() == 1:
            probs = self.circuit(
                query, key,
                self.W_q_y, self.W_q_z,
                self.W_k_y, self.W_k_z,
                self.W_int
            )

            return probs[0]

        scores = []
        for q, k in zip(query, key):
            probs = self.circuit(
                q, k,
                self.W_q_y, self.W_q_z,
                self.W_k_y, self.W_k_z,
                self.W_int
            )
            scores.append(probs[0])

        return torch.stack(scores)

class AdvancedQuantumMultiHeadAttention(nn.Module):

    def __init__(self, config: AdvancedQMHAConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        if config.attention_type == "entanglement":
            self.heads = nn.ModuleList([
                EntanglementBasedAttention(
                    config.n_qubits, config.n_layers,
                    config.data_reuploading
                )
                for _ in range(config.n_heads)
            ])
        elif config.attention_type == "variational":
            self.heads = nn.ModuleList([
                VariationalQuantumAttention(
                    config.n_qubits, config.n_layers
                )
                for _ in range(config.n_heads)
            ])
        else:

            self.heads = nn.ModuleList([
                EntanglementBasedAttention(
                    config.n_qubits, config.n_layers,
                    config.data_reuploading
                )
                for _ in range(config.n_heads)
            ])

        if config.data_reuploading:
            self.value_encoder = DataReuploadingEncoder(
                config.n_qubits, config.n_reuploading_layers
            )

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, _ = query.shape

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        head_outputs = []
        attention_weights = []

        for h in range(self.n_heads):

            scores = torch.zeros(batch_size, seq_len, seq_len)

            for b in range(batch_size):
                for i in range(seq_len):
                    for j in range(seq_len):
                        score = self.heads[h](Q[b, h, i], K[b, h, j])
                        if isinstance(score, Tensor):
                            scores[b, i, j] = score.mean()
                        else:
                            scores[b, i, j] = score

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            output = torch.bmm(weights, V[:, h])

            head_outputs.append(output)
            attention_weights.append(weights)

        output = torch.cat(head_outputs, dim=-1)
        output = self.W_o(output)

        return output, torch.stack(attention_weights, dim=1)
