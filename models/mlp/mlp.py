"""
mlp.py -- Standard MLP baseline for the diffusion PINN comparison.

Uses tanh activations. This is not a stylistic choice: the physics loss
requires a second spatial derivative of the network output taken via
autograd, and ReLU-family activations have zero second derivative almost
everywhere, which starves the PDE residual of gradient information. tanh
is the standard choice in the physics-informed neural network literature
for exactly this reason (Raissi, Perdikaris & Karniadakis, 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
