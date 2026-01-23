__version__ = "0.2.0"
__author__ = "Quantum AI Research Team"
__license__ = "Apache-2.0"

from quantum_transformers.core.quantum_transformer import (
    QuantumTransformer,
    QuantumTransformerConfig,
    QuantumTransformerEncoder,
    QuantumTransformerBlock,
    QuantumMultiHeadAttention as CoreQuantumAttention,
    QuantumFeedForward as CoreQuantumFFN,
    QuantumPositionalEncoding as CorePositionalEncoding,
)

from quantum_transformers.attention.quantum_attention import (
    QuantumMultiHeadAttention,
    QuantumSelfAttention,
    QuantumCrossAttention,
    QuantumAttentionConfig,
    SwapTestAttention,
    InnerProductAttention,
)

from quantum_transformers.encoding.positional import (
    QuantumPositionalEncoding,
    QuantumSinusoidalEncoding,
    QuantumRotationalEncoding,
    LearnableQuantumPositionalEncoding,
)

from quantum_transformers.encoding.data import (
    QuantumAmplitudeEncoding,
    QuantumAngleEncoding,
    QuantumIQPEncoding,
)

from quantum_transformers.layers.feedforward import (
    QuantumFeedForward,
    QuantumMLP,
    VariationalLayer,
)

from quantum_transformers.layers.normalization import (
    QuantumLayerNorm,
    QuantumRMSNorm,
    AmplitudeNormalization,
)

from quantum_transformers.layers.connections import (
    QuantumResidualConnection,
    QuantumDropout,
)

from quantum_transformers.circuits.attention_circuits import (
    SwapTestCircuit,
    InnerProductCircuit,
    QuantumDotProductCircuit,
)

from quantum_transformers.circuits.variational import (
    VariationalCircuit,
    EntanglingLayer,
    DataEncodingLayer,
)

from quantum_transformers.circuits.measurements import (
    ExpvalMeasurement,
    ProbsMeasurement,
)

from quantum_transformers.molecular.transformer_model import (
    QuantumTransformerForMolecules,
    MolecularEnergyPredictor,
    MolecularPropertyPredictor,
    MolecularModelConfig,
)

from quantum_transformers.molecular.tokenizer import (
    MolecularTokenizer,
    SMILESTokenizer,
    AtomTokenizer,
)

from quantum_transformers.molecular.embeddings import (
    AtomEmbedding,
    MolecularGraphEmbedding,
)

from quantum_transformers.optimization.optimizers import (
    QuantumAdamOptimizer,
    QuantumSGDOptimizer,
    QuantumNaturalGradient,
)

from quantum_transformers.optimization.gradient import (
    ParameterShiftGradient,
    SPSAGradient,
)

from quantum_transformers.optimization.schedulers import (
    WarmupScheduler,
    CosineAnnealingScheduler,
)

__all__ = [

    "__version__",

    "QuantumTransformer",
    "QuantumTransformerConfig",
    "QuantumTransformerEncoder",
    "QuantumTransformerBlock",

    "QuantumMultiHeadAttention",
    "QuantumSelfAttention",
    "QuantumCrossAttention",
    "QuantumAttentionConfig",
    "SwapTestAttention",
    "InnerProductAttention",

    "QuantumPositionalEncoding",
    "QuantumSinusoidalEncoding",
    "QuantumRotationalEncoding",
    "QuantumAmplitudeEncoding",
    "QuantumAngleEncoding",
    "QuantumIQPEncoding",

    "QuantumFeedForward",
    "QuantumMLP",
    "QuantumLayerNorm",
    "QuantumDropout",

    "SwapTestCircuit",
    "VariationalCircuit",
    "EntanglingLayer",

    "QuantumTransformerForMolecules",
    "MolecularEnergyPredictor",
    "MolecularTokenizer",
    "SMILESTokenizer",

    "QuantumAdamOptimizer",
    "QuantumNaturalGradient",
    "ParameterShiftGradient",
]

def get_info() -> dict:

    try:
        import pennylane as qml
        pl_version = qml.__version__
    except ImportError:
        pl_version = "Not installed"

    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"

    return {
        "package": "Quantum Transformer",
        "version": __version__,
        "architecture": "Pure Quantum Transformer",
        "pennylane": pl_version,
        "pytorch": torch_version,
        "components": [
            "Quantum Multi-Head Attention (SWAP Test)",
            "Quantum Feed-Forward (Variational Circuits)",
            "Quantum Positional Encoding",
            "Molecular Energy Prediction",
        ],
    }
