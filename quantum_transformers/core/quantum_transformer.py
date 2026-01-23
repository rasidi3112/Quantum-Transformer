from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

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
class QuantumTransformerConfig:

    n_qubits: int = 4
    n_heads: int = 4
    n_layers: int = 6
    d_model: int = 64
    d_ff: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    attention_type: str = "swap_test"
    positional_encoding: str = "quantum_sinusoidal"
    measurement: str = "expval"
    device: str = "default.qubit"

    def __post_init__(self):

        max_amplitude_dim = 2 ** self.n_qubits
        if self.d_model > max_amplitude_dim * self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) too large for {self.n_qubits} qubits "
                f"with {self.n_heads} heads. Max: {max_amplitude_dim * self.n_heads}"
            )

class QuantumMultiHeadAttention(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required for QuantumMultiHeadAttention")

        self.config = config
        self.n_qubits = config.n_qubits
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.q_devices = [
            qml.device(config.device, wires=config.n_qubits * 3 + 1)
            for _ in range(config.n_heads)
        ]

        self.W_q = nn.Parameter(torch.randn(config.n_heads, config.n_qubits, 3) * 0.1)
        self.W_k = nn.Parameter(torch.randn(config.n_heads, config.n_qubits, 3) * 0.1)
        self.W_v = nn.Parameter(torch.randn(config.n_heads, config.n_qubits, 3) * 0.1)

        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.attention_circuits = self._create_attention_circuits()

    def _create_attention_circuits(self):

        circuits = []

        for h, dev in enumerate(self.q_devices):
            @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
            def attention_circuit(q_data, k_data, v_data, w_q, w_k, w_v):
                n_q = self.n_qubits

                for i in range(n_q):
                    qml.RY(q_data[i % len(q_data)] * np.pi, wires=i)
                    qml.Rot(w_q[i, 0], w_q[i, 1], w_q[i, 2], wires=i)

                for i in range(n_q):
                    qml.RY(k_data[i % len(k_data)] * np.pi, wires=n_q + i)
                    qml.Rot(w_k[i, 0], w_k[i, 1], w_k[i, 2], wires=n_q + i)

                for i in range(n_q):
                    qml.RY(v_data[i % len(v_data)] * np.pi, wires=2*n_q + i)
                    qml.Rot(w_v[i, 0], w_v[i, 1], w_v[i, 2], wires=2*n_q + i)

                ancilla = 3 * n_q
                qml.Hadamard(wires=ancilla)
                for i in range(n_q):
                    qml.CSWAP(wires=[ancilla, i, n_q + i])
                qml.Hadamard(wires=ancilla)

                for i in range(n_q):
                    qml.CRY(np.pi/4, wires=[ancilla, 2*n_q + i])

                return [qml.expval(qml.PauliZ(2*n_q + i)) for i in range(n_q)]

            circuits.append(attention_circuit)

        return circuits

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size, seq_len, _ = query.shape

        q_heads = query.view(batch_size, seq_len, self.n_heads, self.d_k)
        k_heads = key.view(batch_size, seq_len, self.n_heads, self.d_k)
        v_heads = value.view(batch_size, seq_len, self.n_heads, self.d_k)

        q_heads = torch.tanh(q_heads)
        k_heads = torch.tanh(k_heads)
        v_heads = torch.tanh(v_heads)

        head_outputs = []
        attention_scores = []

        for h in range(self.n_heads):
            head_out = []
            scores = []

            for b in range(batch_size):
                seq_out = []
                seq_scores = []

                for i in range(seq_len):

                    q_i = q_heads[b, i, h, :self.n_qubits]

                    attended = torch.zeros(self.n_qubits)
                    pos_scores = []

                    for j in range(seq_len):
                        k_j = k_heads[b, j, h, :self.n_qubits]
                        v_j = v_heads[b, j, h, :self.n_qubits]

                        out = self.attention_circuits[h](
                            q_i, k_j, v_j,
                            self.W_q[h], self.W_k[h], self.W_v[h]
                        )
                        attended += torch.stack(out)
                        pos_scores.append(out[0].item())

                    seq_out.append(attended / seq_len)
                    seq_scores.append(pos_scores)

                head_out.append(torch.stack(seq_out))
                scores.append(seq_scores)

            head_outputs.append(torch.stack(head_out))
            attention_scores.append(scores)

        output = torch.cat(head_outputs, dim=-1)

        if output.shape[-1] < self.config.d_model:
            output = torch.nn.functional.pad(
                output, (0, self.config.d_model - output.shape[-1])
            )

        output = self.output_proj(output)

        attn_weights = torch.tensor(attention_scores[0])

        return output, attn_weights

class QuantumFeedForward(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required")

        self.config = config
        self.n_qubits = config.n_qubits
        n_layers = 3

        self.dev = qml.device(config.device, wires=config.n_qubits)

        self.weights = nn.Parameter(
            torch.randn(n_layers, config.n_qubits, 3) * 0.1
        )

        self.input_proj = nn.Linear(config.d_model, config.n_qubits)
        self.output_proj = nn.Linear(config.n_qubits, config.d_model)

        self.qnode = self._create_ffn_circuit()

    def _create_ffn_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def ffn_circuit(inputs, weights):
            n_layers = weights.shape[0]

            for i in range(self.n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)

            for layer in range(n_layers):
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

        return ffn_circuit

    def forward(self, x: Tensor) -> Tensor:

        batch_size, seq_len, _ = x.shape

        x_proj = torch.tanh(self.input_proj(x))

        outputs = []
        for b in range(batch_size):
            seq_out = []
            for s in range(seq_len):
                q_out = self.qnode(x_proj[b, s], self.weights)
                seq_out.append(torch.stack(q_out))
            outputs.append(torch.stack(seq_out))

        output = torch.stack(outputs).float()

        return self.output_proj(output)

class QuantumPositionalEncoding(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        self.d_model = config.d_model
        self.max_len = config.max_seq_len

        pe = torch.zeros(self.max_len, config.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() *
            (-math.log(10000.0) / config.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:config.d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

class QuantumTransformerBlock(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        self.attention = QuantumMultiHeadAttention(config)
        self.ffn = QuantumFeedForward(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class QuantumTransformerEncoder(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x

class QuantumTransformerDecoder(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config

    def forward(self, x: Tensor, encoder_output: Tensor) -> Tensor:
        return x

class QuantumTransformer(nn.Module):

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()

        self.config = config

        self.pos_encoding = QuantumPositionalEncoding(config)

        self.encoder = QuantumTransformerEncoder(config)

        self.output_layer = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.pos_encoding(x)
        x = self.dropout(x)

        encoded = self.encoder(x, mask)

        return self.output_layer(encoded)

    def count_parameters(self) -> dict:

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
