from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

class ExpvalMeasurement(nn.Module):

    def __init__(
        self,
        n_qubits: int,
        observables: Optional[List[str]] = None,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.observables = observables or ["Z"] * n_qubits

    def get_measurements(self, wires: List[int]):

        measurements = []

        for i, (wire, obs) in enumerate(zip(wires, self.observables)):
            if obs == "X":
                measurements.append(qml.expval(qml.PauliX(wire)))
            elif obs == "Y":
                measurements.append(qml.expval(qml.PauliY(wire)))
            elif obs == "Z":
                measurements.append(qml.expval(qml.PauliZ(wire)))
            elif obs == "H":
                measurements.append(qml.expval(qml.Hadamard(wire)))

        return measurements

class ProbsMeasurement(nn.Module):

    def __init__(self, n_qubits: int, subset: Optional[List[int]] = None):
        super().__init__()

        self.n_qubits = n_qubits
        self.subset = subset

    def get_measurement(self, wires: List[int]):

        if self.subset:
            return qml.probs(wires=[wires[i] for i in self.subset])
        return qml.probs(wires=wires)

class StateMeasurement(nn.Module):

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

    def get_measurement(self, wires: List[int]):

        return qml.state()

class SampleMeasurement(nn.Module):

    def __init__(self, n_qubits: int, shots: int = 1000):
        super().__init__()

        self.n_qubits = n_qubits
        self.shots = shots

    def get_measurement(self, wires: List[int]):

        return [qml.sample(qml.PauliZ(w)) for w in wires]
