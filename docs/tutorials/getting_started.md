# Getting Started with Quantum Transformer

This tutorial walks you through building your first Quantum Transformer for molecular property prediction.

## Prerequisites

```bash
pip install quantum_transformers
```

## Step 1: Import Libraries

```python
import torch
from quantum_transformers import QuantumTransformer, QuantumTransformerConfig
from quantum_transformers.molecular import (
    QuantumTransformerForMolecules,
    SMILESTokenizer,
    MolecularModelConfig,
)
```

## Step 2: Create a Quantum Transformer

```python
# Configure the model
config = QuantumTransformerConfig(
    n_qubits=4,           # 4 qubits per attention head
    n_heads=4,            # 4 attention heads
    n_layers=4,           # 4 transformer blocks
    d_model=64,           # Model dimension
    attention_type="swap_test",  # SWAP test for attention
)

# Create model
model = QuantumTransformer(config)
print(f"Parameters: {model.count_parameters()}")
```

## Step 3: Molecular Property Prediction

```python
# Create molecular model
mol_config = MolecularModelConfig(
    vocab_size=100,
    n_qubits=4,
    n_layers=4,
)

mol_model = QuantumTransformerForMolecules(mol_config)

# Tokenize molecules
tokenizer = SMILESTokenizer()
tokens = tokenizer(["CCO", "c1ccccc1"], max_length=32, return_tensors="pt")

# Predict energies
energies = mol_model(tokens)
print(f"Predicted energies: {energies}")
```

## Step 4: Training

```python
from quantum_transformers.optimization import QuantumAdamOptimizer

# Training data (example)
train_tokens = tokenizer(["C", "CC", "CCC"], max_length=16, return_tensors="pt")
train_energies = torch.tensor([[-1.0], [-2.0], [-3.0]])

# Optimizer
optimizer = QuantumAdamOptimizer(mol_model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    pred = mol_model(train_tokens)
    loss = criterion(pred, train_energies)
    loss.backward()  # Quantum gradients via parameter-shift
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Understanding Quantum Attention

The Quantum Transformer uses **SWAP test** for attention:

```
|0⟩ ──H──●──H── Measure
         │
|ψ_q⟩ ───X───── (Query)
         │
|ψ_k⟩ ───X───── (Key)

P(0) = (1 + |⟨ψ_q|ψ_k⟩|²) / 2
```

This computes the similarity between query and key states using quantum interference.

## Next Steps

- [API Reference](../api/reference.md)
- [Quantum Attention Theory](../theory/foundations.md)
- [Tutorial Notebooks](../../notebooks/)

## Troubleshooting

### Slow Training
- Reduce `n_qubits` (e.g., 2 instead of 4)
- Reduce `n_layers` (e.g., 2 instead of 4)

### Memory Issues
- Reduce batch size
- Use smaller `d_model`
