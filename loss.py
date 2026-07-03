from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_pos_weight(labels: np.ndarray | list[int]) -> float:
    labels = np.asarray(labels).astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return float(neg / max(pos, 1))


def build_loss(loss_name: str, pos_weight: float, device: torch.device, focal_gamma: float = 2.0) -> nn.Module:
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    name = loss_name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    if name == "focal":
        return BinaryFocalLoss(gamma=focal_gamma, pos_weight=weight_tensor, reduction="mean")
    raise ValueError(f"Unknown loss_name={loss_name}. Choose one of: bce, focal")

