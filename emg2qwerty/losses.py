# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loss modules for sequence recognition.

CRCTCLoss implements Consistency-Regularized CTC (CR-CTC) from:
  "CR-CTC: Consistency regularization on CTC for improved speech recognition"
  (Yao et al., ICLR 2025; https://arxiv.org/abs/2410.05101).
It combines standard CTC with optional:
  - Consistency loss between two augmented views (self-distillation).
  - Entropy regularization to suppress peaky distributions.
"""

from __future__ import annotations

import torch
from torch import nn


def _masked_mean(
    x: torch.Tensor, lengths: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """Mean over dim, considering only positions t < lengths[n] for each batch n."""
    # x: (T, N, ...), lengths: (N,) long
    T, N = x.shape[0], x.shape[1]
    device = x.device
    mask = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1) < lengths.unsqueeze(0)
    # mask (T, N): True where valid
    x_flat = x.reshape(T, N, -1)
    mask_expand = mask.unsqueeze(-1).expand_as(x_flat)
    sum_val = (x_flat * mask_expand.float()).sum()
    count = mask_expand.float().sum().clamp(min=1e-8)
    return sum_val / count


class CRCTCLoss(nn.Module):
    """Consistency-Regularized CTC (CR-CTC) loss.

    Combines:
    1. Standard CTC loss on the primary view.
    2. Optional consistency term: MSE or KL between primary and augmented-view
       log-probs (when log_probs_aug is provided).
    3. Optional entropy regularization: encourages less peaky posteriors.

    When log_probs_aug is None and entropy_weight=0, this is equivalent to
    nn.CTCLoss. Used with consistency_weight > 0 and a second forward pass
    (augmented input) for full CR-CTC training.

    Args:
        blank: Blank token index (same as nn.CTCLoss).
        consistency_weight: Weight for consistency loss between two views (default 0).
        entropy_weight: Weight for entropy regularization; positive encourages
            higher entropy / less peaky distributions (default 0).
        consistency_type: 'mse' (mean squared error between log-probs) or
            'kl' (KL from primary to augmented) (default 'mse').
    """

    def __init__(
        self,
        blank: int,
        consistency_weight: float = 0.0,
        entropy_weight: float = 0.0,
        consistency_type: str = "mse",
    ) -> None:
        super().__init__()
        self.blank = blank
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        if consistency_type not in ("mse", "kl"):
            raise ValueError("consistency_type must be 'mse' or 'kl'")
        self.consistency_type = consistency_type
        self._ctc = nn.CTCLoss(blank=blank)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        log_probs_aug: torch.Tensor | None = None,
        input_lengths_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            log_probs: (T, N, C) log probabilities (primary view).
            targets: (N, S) target indices (same as nn.CTCLoss).
            input_lengths: (N,) length of each sequence in log_probs (emission lengths).
            target_lengths: (N,) length of each target.
            log_probs_aug: Optional (T', N, C) log probs from augmented view.
                If provided, consistency_loss is added when consistency_weight > 0.
                Typically T' == T and input_lengths_aug == input_lengths.
            input_lengths_aug: Optional (N,) lengths for augmented view.
                Required if log_probs_aug is not None.

        Returns:
            Scalar loss: ctc_loss + consistency_weight * consistency_loss
                         - entropy_weight * mean_entropy.
        """
        # CTC loss (primary view)
        loss = self._ctc(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        T, N, C = log_probs.shape

        # Entropy regularization: -sum_c p_c log p_c over valid frames
        if self.entropy_weight > 0:
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)  # (T, N)
            mean_entropy = _masked_mean(entropy, input_lengths, dim=0)
            loss = loss - self.entropy_weight * mean_entropy

        # Consistency between primary and augmented view
        if (
            self.consistency_weight > 0
            and log_probs_aug is not None
            and input_lengths_aug is not None
        ):
            T_aug = log_probs_aug.shape[0]
            if T_aug != T:
                # Align by trimming to min length per batch (or could interpolate)
                T_min = min(T, T_aug)
                log_probs = log_probs[:T_min]
                log_probs_aug = log_probs_aug[:T_min]
                input_lengths = input_lengths.clamp(max=T_min)
                input_lengths_aug = input_lengths_aug.clamp(max=T_min)
            lengths_both = torch.minimum(input_lengths, input_lengths_aug)
            if self.consistency_type == "mse":
                diff = (log_probs - log_probs_aug) ** 2
                cons = _masked_mean(diff, lengths_both)
            else:
                # KL(primary || aug) = sum_c p_primary * (log p_primary - log p_aug)
                probs = log_probs.exp()
                kl = (probs * (log_probs - log_probs_aug)).sum(dim=-1)  # (T, N)
                cons = _masked_mean(kl, lengths_both)
            loss = loss + self.consistency_weight * cons

        return loss
