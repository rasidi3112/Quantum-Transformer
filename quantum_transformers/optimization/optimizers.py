from __future__ import annotations
from typing import Dict, Iterator, List, Optional
import math
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

class QuantumAdamOptimizer(Optimizer):

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        quantum_lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad: Optional[float] = 1.0,
    ):
        defaults = dict(
            lr=lr,
            quantum_lr=quantum_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_grad=clip_grad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if group['clip_grad'] is not None:
                    torch.nn.utils.clip_grad_norm_([p], group['clip_grad'])

                is_quantum = p.numel() < 100
                lr = group['quantum_lr'] if is_quantum else group['lr']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class QuantumSGDOptimizer(Optimizer):

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if momentum != 0:
                    state = self.state[p]

                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)

                        if nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss

class QuantumNaturalGradient(Optimizer):

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        regularization: float = 0.01,
        approx: str = "block_diagonal",
    ):
        defaults = dict(lr=lr, regularization=regularization, approx=approx)
        super().__init__(params, defaults)

    def _compute_metric(self, param: Tensor) -> Tensor:

        n = param.numel()

        metric = torch.eye(n, device=param.device)

        reg = self.defaults['regularization']
        metric = metric + reg * torch.eye(n, device=param.device)

        return metric

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.view(-1)

                metric = self._compute_metric(p)

                try:
                    nat_grad = torch.linalg.solve(metric, grad)
                except:
                    nat_grad = grad

                nat_grad = nat_grad.view_as(p)
                p.add_(nat_grad, alpha=-group['lr'])

        return loss
