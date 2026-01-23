import pytest
import numpy as np
import torch

@pytest.fixture
def sample_tokens():

    return torch.randint(0, 50, (4, 16))

@pytest.fixture
def sample_features():

    return torch.randn(4, 16, 64)

@pytest.fixture
def quantum_config():

    from quantum_transformers.core.quantum_transformer import QuantumTransformerConfig

    return QuantumTransformerConfig(
        n_qubits=2,
        n_heads=2,
        n_layers=1,
        d_model=16,
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "quantum_hardware: marks tests requiring real quantum hardware")
