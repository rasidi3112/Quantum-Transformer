from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import torch
from torch import Tensor

class GradientEstimator(ABC):

    @abstractmethod
    def compute(self, circuit: Callable, params: Tensor) -> Tensor:

        pass

class ParameterShiftGradient(GradientEstimator):

    def __init__(self, shift: float = np.pi / 2):
        self.shift = shift

    def compute(self, circuit: Callable, params: Tensor) -> Tensor:

        with torch.no_grad():
            params = params.clone()
            gradient = torch.zeros_like(params)

            flat_params = params.view(-1)
            flat_grad = gradient.view(-1)

            for i in range(len(flat_params)):

                flat_params[i] += self.shift
                f_plus = circuit(params.view_as(params))

                flat_params[i] -= 2 * self.shift
                f_minus = circuit(params.view_as(params))

                flat_params[i] += self.shift

                flat_grad[i] = (f_plus - f_minus) / (2 * np.sin(self.shift))

            return gradient

class SPSAGradient(GradientEstimator):

    def __init__(
        self,
        c: float = 0.1,
        gamma: float = 0.101,
    ):
        self.c = c
        self.gamma = gamma
        self.iteration = 0

    def compute(self, circuit: Callable, params: Tensor) -> Tensor:

        self.iteration += 1

        with torch.no_grad():

            ck = self.c / (self.iteration ** self.gamma)

            delta = torch.sign(torch.rand_like(params) - 0.5)

            f_plus = circuit(params + ck * delta)
            f_minus = circuit(params - ck * delta)

            gradient = (f_plus - f_minus) / (2 * ck * delta)

            return gradient

class FiniteDifferenceGradient(GradientEstimator):

    def __init__(self, epsilon: float = 0.01, method: str = "central"):
        self.epsilon = epsilon
        self.method = method

    def compute(self, circuit: Callable, params: Tensor) -> Tensor:

        with torch.no_grad():
            gradient = torch.zeros_like(params)
            flat_params = params.view(-1)
            flat_grad = gradient.view(-1)

            f0 = circuit(params) if self.method != "central" else None

            for i in range(len(flat_params)):
                params_plus = flat_params.clone()
                params_plus[i] += self.epsilon
                f_plus = circuit(params_plus.view_as(params))

                if self.method == "forward":
                    flat_grad[i] = (f_plus - f0) / self.epsilon
                elif self.method == "backward":
                    params_minus = flat_params.clone()
                    params_minus[i] -= self.epsilon
                    f_minus = circuit(params_minus.view_as(params))
                    flat_grad[i] = (f0 - f_minus) / self.epsilon
                else:
                    params_minus = flat_params.clone()
                    params_minus[i] -= self.epsilon
                    f_minus = circuit(params_minus.view_as(params))
                    flat_grad[i] = (f_plus - f_minus) / (2 * self.epsilon)

            return gradient
