from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

class AtomEmbedding(nn.Module):

    ATOM_FEATURES = {
        'H': [1, 1, 0.31, 2.20],
        'C': [6, 4, 0.77, 2.55],
        'N': [7, 3, 0.71, 3.04],
        'O': [8, 2, 0.66, 3.44],
        'F': [9, 1, 0.57, 3.98],
        'S': [16, 2, 1.05, 2.58],
        'P': [15, 3, 1.07, 2.19],
        'Cl': [17, 1, 0.99, 3.16],
        'Br': [35, 1, 1.20, 2.96],
        'I': [53, 1, 1.39, 2.66],
    }

    def __init__(
        self,
        num_atoms: int = 100,
        d_model: int = 64,
        use_features: bool = True,
    ):
        super().__init__()

        self.num_atoms = num_atoms
        self.d_model = d_model
        self.use_features = use_features

        self.embedding = nn.Embedding(num_atoms, d_model)

        if use_features:
            self.feature_proj = nn.Linear(4, d_model)
            self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, atom_ids: Tensor, atom_features: Optional[Tensor] = None) -> Tensor:

        embedded = self.embedding(atom_ids)

        if self.use_features and atom_features is not None:
            features = self.feature_proj(atom_features)
            embedded = self.combine(torch.cat([embedded, features], dim=-1))

        return embedded

class BondEmbedding(nn.Module):

    BOND_TYPES = {
        'SINGLE': 0,
        'DOUBLE': 1,
        'TRIPLE': 2,
        'AROMATIC': 3,
        'NONE': 4,
    }

    def __init__(self, d_model: int = 64):
        super().__init__()

        self.embedding = nn.Embedding(len(self.BOND_TYPES), d_model)

    def forward(self, bond_types: Tensor) -> Tensor:

        return self.embedding(bond_types)

class MolecularGraphEmbedding(nn.Module):

    def __init__(
        self,
        num_atoms: int = 100,
        d_model: int = 64,
        max_atoms: int = 128,
    ):
        super().__init__()

        self.atom_embed = AtomEmbedding(num_atoms, d_model)
        self.bond_embed = BondEmbedding(d_model)

        self.pos_embed = nn.Embedding(max_atoms, d_model)

        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        atom_ids: Tensor,
        bond_matrix: Optional[Tensor] = None,
        atom_features: Optional[Tensor] = None,
    ) -> Tensor:

        batch_size, num_atoms = atom_ids.shape

        atom_emb = self.atom_embed(atom_ids, atom_features)

        positions = torch.arange(num_atoms, device=atom_ids.device)
        pos_emb = self.pos_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)

        combined = self.combine(torch.cat([atom_emb, pos_emb], dim=-1))

        return combined

class ConformerEmbedding(nn.Module):

    def __init__(self, d_model: int = 64):
        super().__init__()

        self.coord_proj = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.dist_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(
        self,
        coordinates: Tensor,
        center_of_mass: bool = True,
    ) -> Tensor:

        if center_of_mass:
            com = coordinates.mean(dim=1, keepdim=True)
            coordinates = coordinates - com

        coord_emb = self.coord_proj(coordinates)

        distances = torch.norm(coordinates, dim=-1, keepdim=True)
        dist_emb = self.dist_embed(distances)

        return coord_emb + dist_emb
