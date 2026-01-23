import pytest
import numpy as np
import torch

pennylane = pytest.importorskip("pennylane")

class TestQuantumTransformerConfig:

    def test_default_config(self):
        from quantum_transformers.core.quantum_transformer import QuantumTransformerConfig

        config = QuantumTransformerConfig()

        assert config.n_qubits == 4
        assert config.n_heads == 4
        assert config.n_layers == 6
        assert config.d_model == 64

    def test_custom_config(self):
        from quantum_transformers.core.quantum_transformer import QuantumTransformerConfig

        config = QuantumTransformerConfig(
            n_qubits=8,
            n_heads=8,
            n_layers=12,
            d_model=128,
        )

        assert config.n_qubits == 8
        assert config.n_heads == 8

class TestQuantumPositionalEncoding:

    def test_sinusoidal_encoding(self):
        from quantum_transformers.encoding.positional import QuantumSinusoidalEncoding

        encoding = QuantumSinusoidalEncoding(d_model=64, max_len=100)

        x = torch.randn(2, 10, 64)
        output = encoding(x)

        assert output.shape == x.shape

        assert not torch.allclose(output, x)

    def test_rotational_encoding(self):
        from quantum_transformers.encoding.positional import QuantumRotationalEncoding

        encoding = QuantumRotationalEncoding(d_model=64, n_qubits=4)

        x = torch.randn(2, 10, 64)
        output = encoding(x)

        assert output.shape == x.shape

class TestQuantumFeedForward:

    def test_forward_shape(self):
        from quantum_transformers.layers.feedforward import QuantumFeedForward

        ffn = QuantumFeedForward(d_model=64, n_qubits=4, n_layers=2)

        x = torch.randn(2, 10, 64)
        output = ffn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self):
        from quantum_transformers.layers.feedforward import QuantumFeedForward

        ffn = QuantumFeedForward(d_model=32, n_qubits=2, n_layers=1)

        x = torch.randn(1, 4, 32, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert ffn.weights.grad is not None

class TestSwapTestCircuit:

    def test_initialization(self):
        from quantum_transformers.circuits.attention_circuits import SwapTestCircuit

        circuit = SwapTestCircuit(n_qubits=2)

        assert circuit.n_qubits == 2
        assert circuit.total_wires == 5

    def test_identical_states(self):
        from quantum_transformers.circuits.attention_circuits import SwapTestCircuit

        circuit = SwapTestCircuit(n_qubits=2)

        psi = torch.tensor([0.5, 0.5])
        phi = torch.tensor([0.5, 0.5])

        overlap = circuit(psi, phi)

        assert overlap > 0.5

class TestVariationalCircuit:

    def test_forward(self):
        from quantum_transformers.circuits.variational import VariationalCircuit

        circuit = VariationalCircuit(n_qubits=4, n_layers=2)

        x = torch.randn(4)
        output = circuit(x)

        assert output.shape == (4,)

        assert (output >= -1).all() and (output <= 1).all()

    def test_batch_forward(self):
        from quantum_transformers.circuits.variational import VariationalCircuit

        circuit = VariationalCircuit(n_qubits=2, n_layers=1)

        x = torch.randn(8, 2)
        output = circuit(x)

        assert output.shape == (8, 2)

class TestMolecularTokenizer:

    def test_smiles_tokenizer(self):
        from quantum_transformers.molecular.tokenizer import SMILESTokenizer

        tokenizer = SMILESTokenizer()

        smiles = "CCO"
        tokens = tokenizer.tokenize(smiles)

        assert len(tokens) == 3
        assert tokens == ["C", "C", "O"]

    def test_encode_decode(self):
        from quantum_transformers.molecular.tokenizer import SMILESTokenizer

        tokenizer = SMILESTokenizer()

        smiles = "CCO"
        encoded = tokenizer.encode(smiles, max_length=10)

        assert len(encoded) == 10

        decoded = tokenizer.decode(encoded)
        assert "CCO" in decoded

class TestQuantumTransformerForMolecules:

    def test_initialization(self):
        from quantum_transformers.molecular.transformer_model import (
            QuantumTransformerForMolecules,
            MolecularModelConfig,
        )

        config = MolecularModelConfig(
            vocab_size=50,
            n_qubits=2,
            n_layers=1,
        )

        model = QuantumTransformerForMolecules(config)

        assert model.config.vocab_size == 50

    def test_forward(self):
        from quantum_transformers.molecular.transformer_model import (
            QuantumTransformerForMolecules,
            MolecularModelConfig,
        )

        config = MolecularModelConfig(
            vocab_size=50,
            n_qubits=2,
            n_heads=2,
            n_layers=1,
            d_model=16,
        )

        model = QuantumTransformerForMolecules(config)

        tokens = torch.randint(0, 50, (2, 8))
        output = model(tokens)

        assert output.shape == (2, 1)

@pytest.mark.slow
class TestQuantumTransformerTraining:

    def test_training_loop(self):
        from quantum_transformers.molecular.transformer_model import (
            MolecularEnergyPredictor,
        )
        from quantum_transformers.optimization.optimizers import QuantumAdamOptimizer

        model = MolecularEnergyPredictor(vocab_size=50, n_qubits=2, n_layers=1)
        optimizer = QuantumAdamOptimizer(model.parameters(), lr=0.01)

        tokens = torch.randint(0, 50, (4, 8))
        target = torch.randn(4, 1)

        model.train()
        optimizer.zero_grad()
        output = model(tokens)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
