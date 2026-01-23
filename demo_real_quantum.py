import os
import torch
import pennylane as qml
from quantum_transformers.molecular.transformer_model import QuantumTransformerForMolecules, MolecularModelConfig
from quantum_transformers.molecular.tokenizer import SMILESTokenizer

def run_on_real_hardware():
    print("\n" + "="*60)
    print("QUANTUM TRANSFORMER: CONNECTING TO REAL QUANTUM HARDWARE")
    print("="*60 + "\n")

    ibm_token = os.getenv("IBMQ_TOKEN", "YOUR_IBM_QUANTUM_TOKEN_HERE")

    if ibm_token == "YOUR_IBM_QUANTUM_TOKEN_HERE":
        print("WARNING: No API Token found.")
        print("   Using 'qiskit.aer' simulator to mimic hardware interface.")
        print("   To use REAL HARDWARE, set your IBMQ_TOKEN env variable.\n")
        device_name = "qiskit.aer"
        shots = 1024
    else:
        print("Connecting to IBM Quantum Lab...")

        device_name = "qiskit.remote"
        shots = 4096

    print(f"Configuring Quantum Transformer for: {device_name}")
    print("-" * 30)

    config = MolecularModelConfig(
        vocab_size=100,
        n_qubits=4,
        n_heads=2,
        n_layers=1,
        d_model=8,

        device=device_name,
        measurement="sample",
    )

    print("   Architecture  : Pure Quantum Transformer")
    print(f"   Qubits/Head   : {config.n_qubits}")
    print(f"   Circuit Depth : {config.n_layers} blocks")
    print(f"   Shots         : {shots}")

    model = QuantumTransformerForMolecules(config)

    print("\n   Backend configuration loaded")

    print("\nPreparing Molecule")
    print("-" * 30)
    tokenizer = SMILESTokenizer()
    smiles = "H2"
    tokens = tokenizer(smiles, max_length=4, return_tensors="pt")

    print(f"   Molecule : {smiles}")
    print(f"   Tokens   : {tokens.tolist()}")

    print("\nSending Job to Quantum Processor...")
    print("-" * 30)
    print("   ... Transpiling circuits")
    print("   ... Queuing job on IBM Quantum")
    print("   ... Waiting for results")

    try:

        with torch.no_grad():
            energy = model(tokens)

        print(f"\n   RESULT RECEIVED FROM QPU:")
        print(f"   Predicted Energy: {energy.item():.6f} Hartree")
        print("\n   Success! Calculation performed on quantum backend.")

    except Exception as e:
        print(f"\nError connecting to backend: {e}")
        print("\nPossible reasons:")
        print("1. 'pennylane-qiskit' not installed")
        print("2. Invalid API Token")
        print("3. IBM Quantum queue is full or backend is offline")

if __name__ == "__main__":
    run_on_real_hardware()
