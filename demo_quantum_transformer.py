import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from quantum_transformers.molecular.transformer_model import QuantumTransformerForMolecules, MolecularModelConfig
from quantum_transformers.molecular.tokenizer import SMILESTokenizer
from quantum_transformers.attention.advanced_qmha import AdvancedQMHAConfig, AdvancedQuantumMultiHeadAttention
from quantum_transformers.optimization.noise_aware import NoiseAwareTrainer, NoiseConfig
from quantum_transformers.optimization.advanced_optimizers import QuantumNaturalGradientOptimizer

def demo_quantum_transformer():
    print("\n" + "="*60)
    print("QUANTUM TRANSFORMER: ADVANCED QUANTUM ARCHITECTURE DEMO")
    print("="*60 + "\n")

    print("1. Molecular Tokenization")
    print("-" * 30)
    tokenizer = SMILESTokenizer()
    smiles = "CCO"

    tokens = tokenizer(smiles, max_length=16, return_tensors="pt")
    decoded = tokenizer.decode(tokens[0].tolist())

    print(f"   Input SMILES : {smiles}")
    print(f"   Tokens Shape : {tokens.shape}")
    print(f"   Token IDs    : {tokens[0].tolist()}")
    print(f"   Decoded      : {decoded}")
    print("   Tokenizer works\n")

    print("2. Initializing Quantum Transformer Model")
    print("-" * 30)
    config = MolecularModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_qubits=4,
        n_heads=2,
        n_layers=2,
        d_model=16,
        task="energy"
    )

    model = QuantumTransformerForMolecules(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture : Pure Quantum Transformer")
    print(f"   Task         : Molecular Energy Prediction")
    print(f"   Total Params : {params}")
    print("   Model initialized\n")

    print("3. Running Quantum Inference")
    print("-" * 30)
    print("   Executing quantum circuits for prediction...")

    with torch.no_grad():
        energy = model(tokens)

    print(f"   Input Molecule   : Ethanol ({smiles})")
    print(f"   Predicted Energy : {energy.item():.6f} Hartree")
    print("   Forward pass successful\n")

    print("4. Demonstrating Advanced Entanglement Attention")
    print("-" * 30)

    adv_config = AdvancedQMHAConfig(
        n_qubits=4,
        n_heads=2,
        d_model=16,
        attention_type="entanglement",
        data_reuploading=True
    )

    print(f"   Attention Type   : {adv_config.attention_type.upper()}")
    print(f"   Data Re-uploading: {adv_config.data_reuploading}")

    adv_attention = AdvancedQuantumMultiHeadAttention(adv_config)

    x = torch.randn(1, 5, 16)

    print("   Computing attention scores...")
    attn_out, attn_weights = adv_attention(x, x, x)

    print(f"   Attention Output : {attn_out.shape}")
    print(f"   Attention Weights: {attn_weights.shape}")
    print("   Advanced attention circuit executed\n")

    print("5. Configuring NISQ Noise-Aware Training")
    print("-" * 30)

    noise_config = NoiseConfig(preset="fake_guadalupe")
    trainer = NoiseAwareTrainer(
        model=model,
        noise_config=noise_config,
        mitigation="zne"
    )

    depth = trainer.estimate_circuit_depth()
    noise_factor = trainer.compute_noise_factor(depth)

    print(f"   Target Backend   : {noise_config.preset}")
    print(f"   Error Mitigation : Zero-Noise Extrapolation (ZNE)")
    print(f"   Est. Circuit Depth: {depth}")
    print(f"   Exp. Fidelity    : {noise_factor:.4f}")
    print("   Noise-aware trainer ready\n")

    print("="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("The Quantum Transformer ecosystem is fully operational.")
    print("="*60 + "\n")

if __name__ == "__main__":
    demo_quantum_transformer()
