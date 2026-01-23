from __future__ import annotations
import math
from typing import Optional

class QuantumLRScheduler:

    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class WarmupScheduler(QuantumLRScheduler):

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        return self.base_lrs

class CosineAnnealingScheduler(QuantumLRScheduler):

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:

            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]

class StepScheduler(QuantumLRScheduler):

    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]

class QuantumAwareScheduler(QuantumLRScheduler):

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        quantum_lr_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.quantum_lr_factor = quantum_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / self.warmup_steps
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]

            if 'is_quantum' in group and group['is_quantum']:
                lrs.append(base_lr * scale * self.quantum_lr_factor)
            else:
                lrs.append(base_lr * scale)

        return lrs
