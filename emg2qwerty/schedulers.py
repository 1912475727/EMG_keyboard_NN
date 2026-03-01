# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingStepsLR(LRScheduler):
    """Step-based linear warmup then cosine annealing to eta_min.
    Use with Lightning interval='step' to warm up over the first warmup_steps steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = (step + 1) / self.warmup_steps
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * scale
                for base_lr in self.base_lrs
            ]
        if step >= self.total_steps:
            return [self.eta_min for _ in self.base_lrs]
        # Cosine decay from base_lr to eta_min over (total_steps - warmup_steps) steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
