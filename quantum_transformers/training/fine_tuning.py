from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

@dataclass
class FineTuningConfig:

    base_lr: float = 0.001
    quantum_lr: float = 0.0001
    classical_lr: float = 0.01

    freeze_quantum: bool = False
    freeze_classical: bool = False

    weight_decay: float = 0.01
    dropout: float = 0.1

    warmup_epochs: int = 5
    total_epochs: int = 100

    task: str = "classification"

class LayerwiseFineTuner:

    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig = None,
    ):
        self.model = model
        self.config = config or FineTuningConfig()

        self.quantum_layers = []
        self.classical_layers = []

        for name, module in model.named_modules():
            if any(q in name.lower() for q in ['quantum', 'qubit', 'attention']):
                self.quantum_layers.append((name, module))
            elif isinstance(module, (nn.Linear, nn.LayerNorm)):
                self.classical_layers.append((name, module))

    def freeze_all_except_head(self):

        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'output_layer'):
            for param in self.model.output_layer.parameters():
                param.requires_grad = True

    def unfreeze_layer(self, layer_idx: int):

        all_layers = self.quantum_layers + self.classical_layers

        if 0 <= layer_idx < len(all_layers):
            name, module = all_layers[layer_idx]
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfrozen layer: {name}")

    def get_optimizer_groups(self) -> List[Dict]:

        groups = []

        quantum_params = []
        for name, param in self.model.named_parameters():
            if any(q in name.lower() for q in ['quantum', 'qubit', 'rot', 'entangle']):
                if param.requires_grad:
                    quantum_params.append(param)

        if quantum_params:
            groups.append({
                'params': quantum_params,
                'lr': self.config.quantum_lr,
                'weight_decay': self.config.weight_decay * 0.1,
            })

        classical_params = []
        for name, param in self.model.named_parameters():
            if not any(q in name.lower() for q in ['quantum', 'qubit', 'rot', 'entangle']):
                if param.requires_grad:
                    classical_params.append(param)

        if classical_params:
            groups.append({
                'params': classical_params,
                'lr': self.config.classical_lr,
                'weight_decay': self.config.weight_decay,
            })

        return groups

    def progressive_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_stages: int = 3,
        epochs_per_stage: int = 10,
    ) -> Dict[str, List[float]]:

        history = {'train_loss': [], 'val_loss': [], 'stage': []}
        criterion = nn.MSELoss()

        for stage in range(n_stages):
            print(f"\n=== Stage {stage + 1}/{n_stages} ===")

            if stage == 0:
                self.freeze_all_except_head()
            elif stage == 1:

                n_blocks = len(self.quantum_layers) + len(self.classical_layers)
                for i in range(max(0, n_blocks - 3), n_blocks):
                    self.unfreeze_layer(i)
            else:

                for param in self.model.parameters():
                    param.requires_grad = True

            groups = self.get_optimizer_groups()
            optimizer = torch.optim.AdamW(groups)

            for epoch in range(epochs_per_stage):
                self.model.train()
                epoch_loss = 0

                for batch in train_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    else:
                        inputs = batch
                        targets = batch

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                history['train_loss'].append(epoch_loss / len(train_loader))
                history['stage'].append(stage)

                if val_loader:
                    val_loss = self._validate(val_loader, criterion)
                    history['val_loss'].append(val_loss)

        return history

    def _validate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = batch

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(loader)

class FinancialTimeSeriesAdapter:

    def __init__(
        self,
        model: nn.Module,
        sequence_length: int = 32,
        n_features: int = 5,
    ):
        self.model = model
        self.sequence_length = sequence_length
        self.n_features = n_features

        self.classification_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def preprocess(
        self,
        prices: np.ndarray,
        normalize: bool = True,
    ) -> Tensor:

        returns = np.diff(prices, axis=0) / prices[:-1]

        if normalize:

            window = min(20, len(returns) // 4)
            vol = np.std(returns[-window:])
            returns = returns / (vol + 1e-8)

        return torch.tensor(returns, dtype=torch.float32)

    def create_sequences(
        self,
        data: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        sequences = []
        seq_labels = []

        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length]

            sequences.append(seq)
            seq_labels.append(label)

        return torch.stack(sequences), torch.stack(seq_labels)

    def forward(self, x: Tensor) -> Tensor:

        features = self.model(x)

        if features.dim() == 3:
            features = features.mean(dim=1)

        logits = self.classification_head(features)

        return logits

class MolecularStructureAdapter:

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int = 100,
        max_atoms: int = 128,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.max_atoms = max_atoms

        self.energy_head = nn.Linear(64, 1)
        self.property_head = nn.Linear(64, 12)
        self.conformer_head = nn.Linear(64, max_atoms * 3)

    def predict_energy(self, tokens: Tensor) -> Tensor:

        features = self.model(tokens)

        if features.dim() == 3:
            features = features.mean(dim=1)

        return self.energy_head(features)

    def predict_properties(self, tokens: Tensor) -> Dict[str, Tensor]:

        features = self.model(tokens)

        if features.dim() == 3:
            features = features.mean(dim=1)

        props = self.property_head(features)

        return {
            'mu': props[:, 0],
            'alpha': props[:, 1],
            'homo': props[:, 2],
            'lumo': props[:, 3],
            'gap': props[:, 4],
            'r2': props[:, 5],
            'zpve': props[:, 6],
            'u0': props[:, 7],
            'u': props[:, 8],
            'h': props[:, 9],
            'g': props[:, 10],
            'cv': props[:, 11],
        }

    def predict_conformer(self, tokens: Tensor) -> Tensor:

        features = self.model(tokens)

        if features.dim() == 3:
            features = features.mean(dim=1)

        coords_flat = self.conformer_head(features)

        return coords_flat.view(-1, self.max_atoms, 3)

class TransferLearningManager:

    def __init__(
        self,
        source_model: nn.Module,
        target_task: str,
    ):
        self.source_model = source_model
        self.target_task = target_task

    def create_target_model(
        self,
        output_dim: int,
        freeze_backbone: bool = True,
    ) -> nn.Module:

        target_model = type(self.source_model)(
            getattr(self.source_model, 'config', None)
        )

        target_model.load_state_dict(
            self.source_model.state_dict(),
            strict=False
        )

        if freeze_backbone:
            for param in target_model.parameters():
                param.requires_grad = False

        if hasattr(target_model, 'head'):
            in_features = target_model.head[-1].in_features if isinstance(
                target_model.head, nn.Sequential
            ) else 64
        else:
            in_features = 64

        target_model.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, output_dim),
        )

        return target_model

    @staticmethod
    def discriminative_lr(
        model: nn.Module,
        base_lr: float = 0.001,
        decay_factor: float = 0.9,
    ) -> List[Dict]:

        params = []
        layers = list(model.named_parameters())
        n_layers = len(layers)

        for i, (name, param) in enumerate(layers):
            lr = base_lr * (decay_factor ** (n_layers - i - 1))
            params.append({
                'params': [param],
                'lr': lr,
                'name': name,
            })

        return params

def fine_tune_for_task(
    model: nn.Module,
    task: str,
    train_data,
    val_data,
    config: FineTuningConfig = None,
) -> Dict:

    config = config or FineTuningConfig()

    fine_tuner = LayerwiseFineTuner(model, config)

    if isinstance(train_data, tuple):
        train_dataset = torch.utils.data.TensorDataset(*train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    else:
        train_loader = train_data

    if isinstance(val_data, tuple):
        val_dataset = torch.utils.data.TensorDataset(*val_data)
        val_loader = DataLoader(val_dataset, batch_size=32)
    else:
        val_loader = val_data

    history = fine_tuner.progressive_training(
        train_loader,
        val_loader,
        n_stages=3,
        epochs_per_stage=config.total_epochs // 3,
    )

    return {
        'history': history,
        'model': model,
        'config': config,
    }
