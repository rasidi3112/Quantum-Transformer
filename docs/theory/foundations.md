# Quantum Transformer Theory

## 1. Classical vs Quantum Attention

### Classical Attention
Standard attention computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Quantum Attention (SWAP Test)
Quantum attention uses quantum interference to compute similarity:
$$P_{\text{similar}} = |\langle\psi_q|\psi_k\rangle|^2$$

The SWAP test circuit:
```
|0⟩ ──H──●──H── Measure
         │
|ψ_q⟩ ───X───── (Query state)
         │
|ψ_k⟩ ───X───── (Key state)
```

Probability of measuring |0⟩:
$$P(0) = \frac{1 + |\langle\psi_q|\psi_k\rangle|^2}{2}$$

## 2. Quantum Positional Encoding

Position is encoded into quantum amplitudes:
$$|pos_i\rangle = U(\theta_i)|0\rangle^{\otimes n}$$

where $\theta_i$ is derived from sinusoidal functions:
$$\theta_i^{(k)} = \sin\left(\frac{i}{10000^{2k/d}}\right)$$

## 3. Quantum Feed-Forward Network

Replaces classical FFN with variational quantum circuit:

$$\text{QFF}(x) = \langle 0|U^\dagger(\theta)O U(\theta)|0\rangle$$

The variational circuit structure:
```
Layer 1: RY(θ₁) ─ RZ(θ₂) ─ CNOT chain
Layer 2: RY(θ₃) ─ RZ(θ₄) ─ CNOT chain
...
```

## 4. Quantum Gradient Computation

Gradients are computed via the **parameter-shift rule**:

$$\frac{\partial f}{\partial \theta_i} = \frac{f(\theta_i + \pi/2) - f(\theta_i - \pi/2)}{2}$$

This enables exact gradients through quantum circuits.

## 5. Advantages of Quantum Transformer

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Attention Complexity | O(n²d) | O(n² log d) |
| Expressibility | Limited | Exponential Hilbert space |
| Entanglement | None | Natural quantum correlations |
| Parameter Efficiency | Lower | Higher (exponential state space) |

## References

1. "Quantum Transformers" - Cherrat et al. (2023)
2. "Attention via SWAP Test" - Quantum ML (2024)
3. "Variational Quantum Circuits" - Schuld & Petruccione (2021)
