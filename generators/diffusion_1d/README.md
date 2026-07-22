# diffusion_1d

Deterministic finite-difference reference solution for the 1D diffusion
equation:

```
dc/dt = D * d^2c/dx^2,      x in [0, L],   t in [0, T]
```

mathematically identical to the heat equation. Framed here as **lithium-ion
concentration diffusing inside a battery electrode particle during
charging**:

| Symbol | Meaning | Units |
|---|---|---|
| `c(x, t)` | lithium concentration in the solid phase | mol/m³ |
| `x` | position across the particle (0 = surface, L = interior) | m |
| `D` | solid-state Li⁺ diffusion coefficient | m²/s |

**Boundary conditions**
- `x = 0` (surface): a constant applied flux models the lithium-insertion
  current during charging - `-D * dc/dx|_{x=0} = J_surface`
- `x = L` (particle interior): zero-flux / symmetry - lithium cannot leave
  through this side - `dc/dx|_{x=L} = 0`

**Initial condition**: `c(x, 0) = c0` (uniform initial lithiation).

Default parameters are order-of-magnitude realistic for a graphite anode
particle, simplified to 1D planar geometry (the same simplification step
from sphere to slab used in introductory single-particle battery models).

## Method

Crank-Nicolson finite differences (2nd order accurate, unconditionally
stable), Neumann boundary conditions folded into the discrete Laplacian via
ghost nodes, with a short backward-Euler warm-up (Rannacher time-stepping)
to damp the start-up oscillation caused by the flux boundary condition
switching on abruptly at `t=0`.

Fully deterministic: no randomness is used anywhere, so the same
`DiffusionParams` always produce a bit-identical solution.

## Usage

```bash
python generate.py                              # solve with the defaults below
python generate.py --D 2e-14 --t_end 600         # override any parameter
```

Every field of `DiffusionParams` is a CLI flag. Outputs (`outputs/`):
- `diffusion_1d_solution.npz` - the solution grid (`x`, `t`, `C`) plus the
  physical parameters used to produce it
- `diffusion_1d_params.json` - the same parameters, human-readable
- `diffusion_1d_solution.png` - a 3D surface + top-down heatmap of `c(x,t)`

A built-in sanity check integrates the PDE's exact mass-balance invariant
(total lithium inserted must equal `J * T`) and reports the relative error
- with the default grid this comes out at machine precision (~1e-12).

## Reuse in an experiment

Experiments don't call `python generate.py` directly - they import
`DiffusionParams` and `solve_diffusion_1d` and pass in whatever physical
parameters their own `config.yaml` specifies (see
`experiments/kan_vs_mlp_battery_diffusion/`), so this generator can serve
multiple experiments with different physics without editing this file.
