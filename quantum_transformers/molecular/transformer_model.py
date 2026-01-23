from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from quantum_transformers.core.quantum_transformer import (
    QuantumTransformer,
    QuantumTransformerConfig,
)

@dataclass
class MolecularModelConfig:

    vocab_size: int = 100
    max_seq_len: int = 128
    n_qubits: int = 4
    n_heads: int = 4
    n_layers: int = 4
    d_model: int = 64
    dropout: float = 0.1
    task: str = "energy"

class QuantumTransformerForMolecules(nn.Module):

    def __init__(self, config: MolecularModelConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        transformer_config = QuantumTransformerConfig(
            n_qubits=config.n_qubits,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.transformer = QuantumTransformer(transformer_config)

        if config.task == "energy":
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, 1),
            )
        elif config.task == "properties":
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 12),
            )
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.embedding(input_ids)

        hidden = self.transformer(x, attention_mask)

        if self.config.task in ["energy", "properties"]:
            pooled = hidden.mean(dim=1)
            output = self.head(pooled)
        else:
            output = self.head(hidden)

        return output

    def predict_energy(self, input_ids: Tensor) -> Tensor:

        return self.forward(input_ids)

    def predict_properties(self, input_ids: Tensor) -> Dict[str, Tensor]:

        output = self.forward(input_ids)

        return {
            "homo": output[:, 0],
            "lumo": output[:, 1],
            "gap": output[:, 2],
            "dipole": output[:, 3:6],
            "polarizability": output[:, 6],
            "energy": output[:, 7],
            "enthalpy": output[:, 8],
            "free_energy": output[:, 9],
            "heat_capacity": output[:, 10],
            "zpve": output[:, 11],
        }

class MolecularEnergyPredictor(QuantumTransformerForMolecules):

    def __init__(
        self,
        vocab_size: int = 100,
        n_qubits: int = 4,
        n_layers: int = 4,
    ):
        config = MolecularModelConfig(
            vocab_size=vocab_size,
            n_qubits=n_qubits,
            n_layers=n_layers,
            task="energy",
        )
        super().__init__(config)

        self.energy_shift = nn.Parameter(torch.tensor(0.0))
        self.energy_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        raw_energy = super().forward(input_ids, **kwargs)
        return raw_energy * self.energy_scale + self.energy_shift

class MolecularPropertyPredictor(QuantumTransformerForMolecules):

    def __init__(
        self,
        vocab_size: int = 100,
        n_qubits: int = 4,
        n_layers: int = 4,
        n_properties: int = 12,
    ):
        config = MolecularModelConfig(
            vocab_size=vocab_size,
            n_qubits=n_qubits,
            n_layers=n_layers,
            task="properties",
        )
        super().__init__(config)

        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, n_properties),
        )

class MolecularGenerator(nn.Module):

    def __init__(self, config: MolecularModelConfig):
        super().__init__()

        config.task = "generation"
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        transformer_config = QuantumTransformerConfig(
            n_qubits=config.n_qubits,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_model=config.d_model,
        )

        self.transformer = QuantumTransformer(transformer_config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        hidden = self.transformer(x)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tensor:

        generated = start_tokens.clone()

        for _ in range(max_length):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == 2).all():
                break

        return generated
