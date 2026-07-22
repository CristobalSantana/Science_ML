"""
kan.py -- A minimal, self-contained Kolmogorov-Arnold Network (KAN).

Why a from-scratch implementation instead of pykan / efficient-kan
--------------------------------------------------------------------
This project needs a KAN whose parameter count we can compute exactly by
hand (to match it against an MLP baseline for a fair comparison) and
whose internals are simple enough to explain line-by-line in a technical
article. The reference implementation (pykan, Liu et al. 2024) is feature
-heavy (symbolic regression, pruning, adaptive grids) and its API has
changed substantially across versions; efficient-kan is lighter but still
a third-party moving target. A ~100-line module with zero KAN-specific
dependencies removes both risks while staying mathematically faithful to
the original formulation.

The math
---------
A KAN layer replaces "linear map + fixed pointwise nonlinearity" (what an
MLP layer does) with a learnable 1D function on every input-output edge.
Each edge's function uses the residual-activation trick from the paper
(eq. 2.10 in Liu et al. 2024), which is what makes it trainable by plain
backprop from a generic initialization:

    phi(x) = w_base * SiLU(x)  +  spline(x)

spline(x) is a linear combination of B-spline basis functions evaluated
on a FIXED, uniform knot grid (no adaptive grid updates -- kept fixed for
simplicity and full training determinism). A KANLinear layer with
`in_features` inputs, `out_features` outputs, grid size G and spline
order k has exactly

    in_features * out_features * (1 + G + k)

parameters: one w_base per edge, plus (G + k) spline coefficients per
edge. No separate activation function is needed between KAN layers --
unlike an MLP, the nonlinearity already lives on every edge.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_grid(in_features: int, grid_size: int, spline_order: int,
                 grid_range: tuple[float, float]) -> torch.Tensor:
    """Uniform, clamped knot vector, padded by `spline_order` extra knots on
    each side (standard construction for open/clamped B-splines), shared by
    every input feature (same range for all of them).

    Returns shape (in_features, grid_size + 2*spline_order + 1).
    """
    step = (grid_range[1] - grid_range[0]) / grid_size
    knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
    knots = knots * step + grid_range[0]
    return knots.unsqueeze(0).expand(in_features, -1).contiguous()


def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Cox-de Boor recursion, vectorized over batch and input features.

    x:    (batch, in_features)
    grid: (in_features, grid_size + 2*spline_order + 1)
    returns: (batch, in_features, grid_size + spline_order) basis values

    The grid is uniform and fixed, so every recursion-step denominator is a
    constant nonzero knot spacing -- no epsilon guards needed.
    """
    x = x.unsqueeze(-1)                                              # (batch, in, 1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)    # order 0
    for p in range(1, spline_order + 1):
        left = (x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)])
        right = (grid[:, p + 1:] - x) / (grid[:, p + 1:] - grid[:, 1:-p])
        bases = left * bases[..., :-1] + right * bases[..., 1:]
    return bases


class KANLinear(nn.Module):
    """One Kolmogorov-Arnold layer: in_features x out_features learnable
    1D edge functions, see module docstring for the math."""

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 6, spline_order: int = 3,
                 grid_range: tuple[float, float] = (-1.5, 1.5)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        self.register_buffer("grid", _build_grid(in_features, grid_size, spline_order, grid_range))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Small init so every edge starts close to a flat function: the
        # base (SiLU) branch carries the initial signal, and the spline
        # branch's contribution grows from ~0 as training shapes it.
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        # Clamp only for the spline lookup (not the base branch): keeps
        # every edge's spline component "alive" (non-zero support) even if
        # an activation drifts outside the fixed grid range during
        # training. The tiny upper-bound epsilon avoids landing exactly on
        # the last knot, where the half-open basis interval would be zero.
        x_clamped = x.clamp(self.grid_range[0], self.grid_range[1] - 1e-4)
        bases = b_spline_basis(x_clamped, self.grid, self.spline_order)
        spline = torch.einsum("bik,oik->bo", bases, self.spline_weight)
        return base + spline

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class KAN(nn.Module):
    """A stack of KANLinear layers. No inter-layer activation -- unlike an
    MLP, the nonlinearity already lives on every edge."""

    def __init__(self, layer_sizes: list[int], grid_size: int = 6,
                 spline_order: int = 3, grid_range: tuple[float, float] = (-1.5, 1.5)):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(layer_sizes[i], layer_sizes[i + 1], grid_size, spline_order, grid_range)
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
