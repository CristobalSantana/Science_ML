"""
generator.py -- Reference (ground-truth) solution of the 1D diffusion equation.

Physics
-------
This module solves the 1D linear diffusion equation

    dc/dt = D * d^2c/dx^2,          x in [0, L],   t in [0, T]

which is mathematically identical to the heat equation (c <-> temperature).
Here it is framed as lithium-ion concentration diffusing inside a battery
electrode particle during charging:

    c(x, t)   lithium concentration in the solid phase        [mol/m^3]
    x         position across the particle; x=0 is the particle
              surface (in contact with the electrolyte), x=L is the
              particle interior / current-collector side       [m]
    D         solid-state Li+ diffusion coefficient of the
              electrode material                                [m^2/s]

Boundary conditions
--------------------
x = 0 (surface): a constant applied flux models the lithium-insertion
current during charging (Neumann BC, Fick's first law):

    -D * dc/dx |_{x=0} = J_surface        [mol/(m^2 s)]

x = L (particle interior): zero-flux / symmetry boundary -- lithium
cannot leave through this side:

    dc/dx |_{x=L} = 0

Initial condition
------------------
    c(x, 0) = c0   (uniform initial lithiation)

Default parameters are order-of-magnitude realistic for a graphite anode
particle (D ~ 1e-14-1e-13 m^2/s, particle size ~ few micrometers, see e.g.
Chen et al. 2020 parameter sets), simplified to 1D planar geometry -- the
same simplification step from sphere to slab that is standard in
introductory single-particle battery models.

Numerical method
-----------------
Crank-Nicolson finite differences (2nd order accurate in space and time,
unconditionally stable), with the Neumann boundary conditions folded into
the discrete Laplacian via ghost nodes. The flux boundary condition
"switches on" abruptly at t=0, which is a classic trigger for Crank-
Nicolson's mild start-up oscillations; we damp this with a handful of
backward-Euler warm-up steps before switching to Crank-Nicolson
(Rannacher time-stepping, Rannacher 1984) -- a standard, cheap fix.

This module produces ONLY the reference solution via a classical
numerical method. No neural networks are involved here. The whole
pipeline is deterministic: given the same DiffusionParams, solve_diffusion_1d
always returns bit-identical output (no randomness is used anywhere).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "outputs"


@dataclass(frozen=True)
class DiffusionParams:
    """Physical and numerical parameters of the 1D diffusion problem.

    All defaults are order-of-magnitude realistic for lithium diffusion in
    a graphite anode particle, simplified to 1D planar geometry.
    """
    D: float = 4.0e-14            # diffusion coefficient [m^2/s]
    L: float = 6.0e-6             # domain length [m]
    c0: float = 15_000.0          # initial uniform concentration [mol/m^3]
    surface_flux: float = 2.5e-5  # applied flux at x=0 [mol/(m^2 s)]
    t_end: float = 1200.0         # total simulated time [s]
    nx: int = 161                 # number of spatial grid points
    nt: int = 800                 # number of time steps

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionSolution:
    """Container for the reference solution grid."""
    x: np.ndarray   # shape (nx,)        [m]
    t: np.ndarray   # shape (nt+1,)      [s]
    C: np.ndarray   # shape (nt+1, nx)   [mol/m^3], C[n, i] = c(x_i, t_n)
    params: DiffusionParams


def _build_laplacian(nx: int, dx: float) -> sp.csr_matrix:
    """Discrete second-derivative operator (1/dx^2 * tridiagonal[-2,1,1])
    with the homogeneous parts of both Neumann boundary conditions folded
    in via ghost nodes:

      x=0: c_{-1} = c_1 + 2*dx*J/D  -> homogeneous half doubles the
           coupling to the first interior neighbour (the J/D term is a
           constant and is added separately as a source vector, see
           solve_diffusion_1d).
      x=L: c_{nx} = c_{nx-2}        -> zero-flux mirror, same trick.
    """
    main = -2.0 * np.ones(nx)
    off = np.ones(nx - 1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    A[0, 1] = 2.0
    A[-1, -2] = 2.0
    return A.tocsr() / dx**2


def solve_diffusion_1d(params: DiffusionParams = DiffusionParams()) -> DiffusionSolution:
    """Solve the 1D diffusion equation with Crank-Nicolson finite differences
    (plus a short backward-Euler warm-up to damp start-up oscillations).

    Returns
    -------
    DiffusionSolution with x (nx,), t (nt+1,), C (nt+1, nx).
    """
    nx, nt = params.nx, params.nt
    dx = params.L / (nx - 1)
    dt = params.t_end / nt
    D = params.D

    x = np.linspace(0.0, params.L, nx)
    t = np.linspace(0.0, params.t_end, nt + 1)

    A = _build_laplacian(nx, dx)   # discrete Laplacian, homogeneous Neumann folded in
    I = sp.identity(nx, format="csr")

    # Constant source term from the x=0 flux BC. Derivation: substituting
    # the ghost node c_{-1}=c_1+2*dx*J/D into the central-difference
    # Laplacian at i=0 splits it into a homogeneous part (already in A)
    # plus a constant 2*J/dx (the D cancels), i.e. dc/dt = D*A@c + source.
    source = np.zeros(nx)
    source[0] = 2.0 * params.surface_flux / dx

    C = np.empty((nt + 1, nx))
    C[0] = params.c0
    c = C[0].copy()

    # --- Rannacher start-up: a few fully-implicit (backward-Euler) steps ---
    n_warmup = min(4, nt)
    if n_warmup:
        r_be = D * dt
        lhs_be = (I - r_be * A).tocsc()
        lu_be = spla.splu(lhs_be)
        for n in range(n_warmup):
            b = c + dt * source
            c = lu_be.solve(b)
            C[n + 1] = c

    # --- Crank-Nicolson for the remaining steps ---
    n_remaining = nt - n_warmup
    if n_remaining:
        r = D * dt / 2.0
        lhs = (I - r * A).tocsc()
        rhs_op = (I + r * A).tocsr()
        lu = spla.splu(lhs)
        source_term = dt * source   # CN average of a time-constant source is itself
        for k in range(n_remaining):
            n = n_warmup + k
            b = rhs_op @ c + source_term
            c = lu.solve(b)
            C[n + 1] = c

    return DiffusionSolution(x=x, t=t, C=C, params=params)


def check_mass_balance(sol: DiffusionSolution) -> float:
    """Validate the solution against an exact physical invariant.

    Integrating dc/dt = D*d^2c/dx^2 over x in [0,L] and using both
    boundary conditions gives an exact mass balance:

        d/dt [ integral_0^L c dx ] = J_surface   (constant)

    so the total lithium accumulated per unit surface area must equal
    the total flux delivered over the run:

        integral_0^L [c(x,T) - c(x,0)] dx  ==  J_surface * T

    Returns the relative error between the two sides (should be small,
    set by the finite-difference truncation error, not by chance).
    """
    p = sol.params
    inserted_numeric = np.trapezoid(sol.C[-1] - sol.C[0], sol.x)
    inserted_expected = p.surface_flux * p.t_end
    return abs(inserted_numeric - inserted_expected) / inserted_expected


def save_solution(sol: DiffusionSolution, outdir: Path) -> tuple[Path, Path]:
    """Save the solution grid (.npz, for reuse by later experiments) and
    the parameters (.json, for human-readable reproducibility)."""
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "diffusion_1d_solution.npz"
    json_path = outdir / "diffusion_1d_params.json"

    np.savez_compressed(npz_path, x=sol.x, t=sol.t, C=sol.C, **sol.params.as_dict())
    json_path.write_text(json.dumps(sol.params.as_dict(), indent=2))

    return npz_path, json_path


def plot_solution(sol: DiffusionSolution, save_path: Path) -> None:
    """Save a two-panel figure: a 3D surface c(x,t) and a 2D heatmap of the
    same data, in readable engineering units (micrometers, minutes)."""
    x_um = sol.x * 1e6
    t_min = sol.t / 60.0
    X, T = np.meshgrid(x_um, t_min)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(X, T, sol.C, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_xlabel("x  [µm]")
    ax1.set_ylabel("t  [min]")
    ax1.set_zlabel("c  [mol/m³]")
    ax1.set_title("Reference solution  c(x, t)")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label="c [mol/m³]")

    ax2 = fig.add_subplot(1, 2, 2)
    mesh = ax2.pcolormesh(X, T, sol.C, cmap="viridis", shading="gouraud")
    ax2.set_xlabel("x  [µm]  (0 = particle surface)")
    ax2.set_ylabel("t  [min]")
    ax2.set_title("Same data, top-down view")
    fig.colorbar(mesh, ax=ax2, label="c [mol/m³]")

    fig.suptitle("1D diffusion equation -- Li-ion concentration in an electrode particle")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in fields(DiffusionParams):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR,
                        help="Directory to save the .npz/.json/.png outputs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    param_fields = {f.name for f in fields(DiffusionParams)}
    params = DiffusionParams(**{k: v for k, v in vars(args).items() if k in param_fields})

    sol = solve_diffusion_1d(params)

    tau = params.L**2 / params.D
    fourier = params.D * params.t_end / params.L**2
    rel_err = check_mass_balance(sol)

    print("1D diffusion generator -- deterministic, no randomness used")
    print(f"  diffusion time scale  tau = L^2/D        = {tau:.1f} s  ({tau/60:.2f} min)")
    print(f"  Fourier number        Fo  = D*t_end/L^2   = {fourier:.3f}")
    print(f"  c range over the run       = [{sol.C.min():.1f}, {sol.C.max():.1f}] mol/m^3")
    print(f"  mass-balance check (numeric integral vs. J*T): relative error = {rel_err:.2e}")
    if rel_err > 1e-2:
        print("  WARNING: mass-balance error is larger than expected for this grid; "
              "consider increasing nx/nt.")

    npz_path, json_path = save_solution(sol, args.outdir)
    png_path = args.outdir / "diffusion_1d_solution.png"
    plot_solution(sol, png_path)

    print(f"  saved solution grid -> {npz_path}")
    print(f"  saved parameters    -> {json_path}")
    print(f"  saved surface plot  -> {png_path}")


if __name__ == "__main__":
    main()
