from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
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

@dataclass
class QuantumAttentionConfig:

    n_qubits: int = 4
    n_heads: int = 4
    d_k: int = 16
    attention_type: str = "swap_test"
    use_ancilla: bool = True
    dropout: float = 0.1

class SwapTestAttention(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=2*n_qubits + 1)

        self.theta_q = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        self.theta_k = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)

        self.qnode = self._create_swap_test()

    def _create_swap_test(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def swap_test(q_state, k_state, theta_q, theta_k):
            ancilla = 0

            qml.Hadamard(wires=ancilla)

            for i in range(self.n_qubits):
                qml.RY(q_state[i] * np.pi, wires=i + 1)
                qml.Rot(theta_q[i, 0], theta_q[i, 1], theta_q[i, 2], wires=i + 1)

            for i in range(self.n_qubits):
                qml.RY(k_state[i] * np.pi, wires=self.n_qubits + 1 + i)
                qml.Rot(theta_k[i, 0], theta_k[i, 1], theta_k[i, 2], wires=self.n_qubits + 1 + i)

            for i in range(self.n_qubits):
                qml.CSWAP(wires=[ancilla, i + 1, self.n_qubits + 1 + i])

            qml.Hadamard(wires=ancilla)

            return qml.expval(qml.PauliZ(ancilla))

        return swap_test

    def forward(self, query: Tensor, key: Tensor) -> Tensor:

        batch_size, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        query = torch.tanh(query[..., :self.n_qubits])
        key = torch.tanh(key[..., :self.n_qubits])

        scores = torch.zeros(batch_size, seq_q, seq_k)

        for b in range(batch_size):
            for i in range(seq_q):
                for j in range(seq_k):

                    score = self.qnode(
                        query[b, i], key[b, j],
                        self.theta_q, self.theta_k
                    )

                    scores[b, i, j] = (score + 1) / 2

        return scores

class InnerProductAttention(nn.Module):

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.qnode = self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def inner_product_circuit(q_data, k_data, weights):

            for i in range(self.n_qubits):
                qml.RY((q_data[i] - k_data[i]) * np.pi, wires=i)

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

            return qml.probs(wires=range(self.n_qubits))

        return inner_product_circuit

    def forward(self, query: Tensor, key: Tensor) -> Tensor:

        batch_size, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        query = torch.tanh(query[..., :self.n_qubits])
        key = torch.tanh(key[..., :self.n_qubits])

        scores = torch.zeros(batch_size, seq_q, seq_k)

        for b in range(batch_size):
            for i in range(seq_q):
                for j in range(seq_k):
                    probs = self.qnode(query[b, i], key[b, j], self.weights)

                    scores[b, i, j] = probs[0]

        return scores

class QuantumSelfAttention(nn.Module):

    def __init__(self, config: QuantumAttentionConfig):
        super().__init__()

        self.config = config

        if config.attention_type == "swap_test":
            self.attention = SwapTestAttention(config.n_qubits)
        else:
            self.attention = InnerProductAttention(config.n_qubits)

        self.W_v = nn.Linear(config.d_k, config.d_k)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        scores = self.attention(x, x)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        values = self.W_v(x)
        output = torch.bmm(attn_weights, values)

        return output, attn_weights

class QuantumCrossAttention(nn.Module):

    def __init__(self, config: QuantumAttentionConfig):
        super().__init__()

        self.config = config
        self.attention = SwapTestAttention(config.n_qubits)
        self.W_v = nn.Linear(config.d_k, config.d_k)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        scores = self.attention(query, key_value)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        values = self.W_v(key_value)
        output = torch.bmm(attn_weights, values)

        return output, attn_weights

class QuantumMultiHeadAttention(nn.Module):

    def __init__(self, config: QuantumAttentionConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.d_k = config.d_k

        self.heads = nn.ModuleList([
            QuantumSelfAttention(config)
            for _ in range(config.n_heads)
        ])

        self.output_proj = nn.Linear(config.n_heads * config.d_k, config.d_k * config.n_heads)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, d_model = query.shape

        q_heads = query.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = key.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = value.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        head_outputs = []
        attention_weights = []

        for h in range(self.n_heads):
            q_h = q_heads[:, h]
            out, weights = self.heads[h](q_h, mask)
            head_outputs.append(out)
            attention_weights.append(weights)

        output = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(output)

        return output, torch.stack(attention_weights, dim=1)
