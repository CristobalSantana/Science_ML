"""
pinn.py -- Physics-informed loss for the 1D diffusion PDE, in dimensionless
form, shared by both the MLP and the KAN.

We train entirely in non-dimensional coordinates so both networks see O(1)
inputs and outputs (raw physical values -- D ~ 1e-14, c ~ 1e4 -- would give
a neural net terrible, vanishing gradients):

    x_hat = x / L                  in [0, 1]
    t_hat = t / t_end               in [0, 1]
    c_hat = (c - c0) / c_scale      starts at 0, grows to O(1)

c_scale is an *analytic* estimate of the surface concentration rise over
the run, from the semi-infinite-domain diffusion approximation -- it does
not depend on (or peek at) the finite-difference reference solution from
generators/diffusion_1d/generate.py:

    c_scale = 2 * J * sqrt(t_end / (pi * D))

Substituting c = c0 + c_scale * c_hat, x = L * x_hat, t = t_end * t_hat
into dc/dt = D * d^2c/dx^2 and simplifying gives

    dc_hat/dt_hat = Fo * d^2c_hat/dx_hat^2,      Fo = D * t_end / L^2

so the *Fourier number* -- the same dimensionless group generator.py
already reports in Phase 1 -- is the only physical constant the PDE
residual needs. The boundary/initial conditions become:

    x_hat = 0 (flux BC):  dc_hat/dx_hat = -J*L / (D*c_scale)  =: bc0_target
    x_hat = 1 (no-flux):  dc_hat/dx_hat = 0
    t_hat = 0 (IC):       c_hat = 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DimensionlessProblem:
    """Dimensionless constants for the PINN loss, derived from the physical
    DiffusionParams used by the Phase 1 generator."""
    fourier: float
    bc0_target: float
    c_scale: float

    @classmethod
    def from_params(cls, params) -> "DimensionlessProblem":
        D, L, J, t_end = params.D, params.L, params.surface_flux, params.t_end
        c_scale = 2.0 * J * math.sqrt(t_end / (math.pi * D))
        fourier = D * t_end / L**2
        bc0_target = -J * L / (D * c_scale)
        return cls(fourier=fourier, bc0_target=bc0_target, c_scale=c_scale)


def sample_collocation_points(n_pde: int, n_bc: int, n_ic: int, device: str = "cpu"):
    """Sample collocation points in the dimensionless domain [0,1] x [0,1].

    Call once, right after torch.manual_seed(seed), and reuse the SAME
    points for every model trained in a run. That is what makes the
    MLP-vs-KAN comparison apples-to-apples (identical training locations)
    while each model still gets its own deterministic, seeded
    initialization.
    """
    xt_pde = torch.rand(n_pde, 2, device=device, requires_grad=True)

    t_bc = torch.rand(n_bc, 1, device=device)
    xt_bc0 = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1).requires_grad_(True)
    xt_bcL = torch.cat([torch.ones_like(t_bc), t_bc], dim=1).requires_grad_(True)

    x_ic = torch.rand(n_ic, 1, device=device)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)

    return xt_pde, xt_bc0, xt_bcL, xt_ic


def pde_residual(model, xt: torch.Tensor, fourier: float) -> torch.Tensor:
    """r = dc_hat/dt_hat - Fo * d^2c_hat/dx_hat^2, via nested autograd."""
    c = model(xt)
    grad_c = torch.autograd.grad(c, xt, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_x, c_t = grad_c[:, 0:1], grad_c[:, 1:2]
    c_xx = torch.autograd.grad(c_x, xt, grad_outputs=torch.ones_like(c_x), create_graph=True)[0][:, 0:1]
    return c_t - fourier * c_xx


def bc_flux_residual(model, xt: torch.Tensor, target_slope: float) -> torch.Tensor:
    """Residual for a Neumann BC: dc_hat/dx_hat - target_slope."""
    c = model(xt)
    c_x = torch.autograd.grad(c, xt, grad_outputs=torch.ones_like(c), create_graph=True)[0][:, 0:1]
    return c_x - target_slope


def ic_residual(model, xt: torch.Tensor, target_value: float = 0.0) -> torch.Tensor:
    return model(xt) - target_value


def physics_informed_loss(model, points, problem: DimensionlessProblem, weights: dict | None = None):
    """Weighted sum of squared residuals: PDE + both BCs + IC.

    Returns (total_loss_tensor, dict of the individual (float) components)
    for logging.
    """
    xt_pde, xt_bc0, xt_bcL, xt_ic = points
    weights = weights or dict(pde=1.0, bc=1.0, ic=1.0)

    r_pde = pde_residual(model, xt_pde, problem.fourier)
    r_bc0 = bc_flux_residual(model, xt_bc0, problem.bc0_target)
    r_bcL = bc_flux_residual(model, xt_bcL, 0.0)
    r_ic = ic_residual(model, xt_ic, 0.0)

    loss_pde = (r_pde ** 2).mean()
    loss_bc = (r_bc0 ** 2).mean() + (r_bcL ** 2).mean()
    loss_ic = (r_ic ** 2).mean()

    total = weights["pde"] * loss_pde + weights["bc"] * loss_bc + weights["ic"] * loss_ic
    return total, dict(pde=loss_pde.item(), bc=loss_bc.item(), ic=loss_ic.item())
