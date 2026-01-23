# Quantum Transformer API Reference

## Core Module

### QuantumTransformer

```python
class QuantumTransformer(nn.Module):
    """Pure Quantum Transformer with quantum attention and FFN."""
    
    def __init__(self, config: QuantumTransformerConfig)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor
    
    def count_parameters(self) -> dict
```

### QuantumTransformerConfig

```python
@dataclass
class QuantumTransformerConfig:
    n_qubits: int = 4           # Qubits per attention head
    n_heads: int = 4            # Number of attention heads
    n_layers: int = 6           # Transformer blocks
    d_model: int = 64           # Model dimension
    d_ff: int = 256             # Feed-forward dimension
    max_seq_len: int = 512      # Maximum sequence length
    dropout: float = 0.1        # Dropout rate
    attention_type: str = "swap_test"  # Attention mechanism
```

---

## Attention Module

### QuantumMultiHeadAttention

```python
class QuantumMultiHeadAttention(nn.Module):
    """Multi-head attention using SWAP test circuits."""
    
    def __init__(self, config: QuantumAttentionConfig)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]
```

### SwapTestAttention

```python
class SwapTestAttention(nn.Module):
    """Single-head SWAP test attention."""
    
    def __init__(self, n_qubits: int = 4)
    
    def forward(self, query: Tensor, key: Tensor) -> Tensor
```

---

## Circuits Module

### SwapTestCircuit

```python
class SwapTestCircuit(nn.Module):
    """SWAP test for computing |⟨ψ|φ⟩|²."""
    
    def __init__(self, n_qubits: int = 4)
    
    def forward(self, psi: Tensor, phi: Tensor) -> Tensor
```

### VariationalCircuit

```python
class VariationalCircuit(nn.Module):
    """Parameterized quantum circuit."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        entanglement: str = "circular",
    )
    
    def forward(self, x: Tensor) -> Tensor
    
    def draw(self) -> str
```

---

## Molecular Module

### QuantumTransformerForMolecules

```python
class QuantumTransformerForMolecules(nn.Module):
    """Quantum Transformer for molecular property prediction."""
    
    def __init__(self, config: MolecularModelConfig)
    
    def forward(self, input_ids: Tensor) -> Tensor
    
    def predict_energy(self, input_ids: Tensor) -> Tensor
```

### SMILESTokenizer

```python
class SMILESTokenizer:
    """Tokenizer for SMILES molecular notation."""
    
    def tokenize(self, smiles: str) -> List[str]
    
    def encode(self, smiles: str, max_length: int) -> List[int]
    
    def decode(self, ids: List[int]) -> str
```

---

## Optimization Module

### QuantumAdamOptimizer

```python
class QuantumAdamOptimizer(Optimizer):
    """Adam with quantum-aware learning rates."""
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        quantum_lr: float = 0.001,
        betas: Tuple = (0.9, 0.999),
    )
```

### ParameterShiftGradient

```python
class ParameterShiftGradient:
    """Exact quantum gradients via parameter shift."""
    
    def compute(self, circuit: Callable, params: Tensor) -> Tensor
```
