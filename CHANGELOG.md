# Changelog

All notable changes to Quantum Transformer will be documented here.

## [0.2.0] - 2025-01-23

### Changed - MAJOR ARCHITECTURE CHANGE
- **Complete rewrite to Pure Quantum Transformer**
- Removed all hybrid quantum-classical components
- All attention now uses quantum SWAP test
- All feed-forward uses variational quantum circuits

### Added
- `QuantumTransformer` - Full quantum transformer model
- `QuantumMultiHeadAttention` - SWAP test based attention
- `SwapTestCircuit` - Quantum inner product computation
- `QuantumFeedForward` - Variational circuit FFN
- `QuantumPositionalEncoding` - Quantum amplitude encoding
- `QuantumTransformerForMolecules` - Molecular applications
- `SMILESTokenizer` - Molecular tokenization
- `QuantumAdamOptimizer` - Quantum-aware optimizer
- `ParameterShiftGradient` - Exact quantum gradients

### Removed
- `HybridQuantumModel` - No longer hybrid
- `VQCLayer` - Replaced with full quantum layers
- `TestTimeQuantumOptimization` - Deprecated
- All classical neural network components

### Philosophy Change
- **From**: Hybrid Quantum-Classical ML
- **To**: Pure Quantum Transformer

---

## [0.1.0] - 2025-01-23

### Added
- Initial hybrid quantum-classical framework
- (Superseded by v0.2.0)
