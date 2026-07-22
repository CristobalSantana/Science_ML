# Science_ML

Science_ML is a collection of lesser-known neural architectures paired with
synthetic data generators built from differential equations, for honest,
fully reproducible scientific experimentation. Every experiment states its
physical setup, its exact hyperparameters, and its finding - including
where the "interesting" architecture loses, not just where it wins.

## Structure

```
generators/     data generators, one per equation, each reusable across experiments
models/         architecture implementations, one per folder
experiments/    each experiment combines a generator + one or more models
```

- **`generators/<name>/`** - `generate.py`, a `README.md` explaining the
  equation and its physical meaning, and an `outputs/` folder with the
  reference solution.
- **`models/<name>/`** - the architecture's implementation and a short
  `README.md`.
- **`experiments/<name>/`** - `run.py` (single entry point, runnable end to
  end), `config.yaml` (every reproducible parameter - physical constants,
  seeds, architecture sizes), a `README.md` with the finding and how to
  reproduce it, and `outputs/`, `checkpoints/`, `media/` for the results.

## Experiments

| Experiment | Question | Finding |
|---|---|---|
| [`kan_vs_mlp_battery_diffusion`](experiments/kan_vs_mlp_battery_diffusion/) | Can a KAN solve a PDE as a physics-informed network as well as an MLP with the same parameter budget? | KAN is ~1.6x more accurate, consistently across 5 seeds - but ~16x slower to train. |

## Generators

| Generator | Equation |
|---|---|
| [`diffusion_1d`](generators/diffusion_1d/) | 1D diffusion / heat equation, framed as lithium-ion battery charging |

## Models

| Model | What it is |
|---|---|
| [`kan`](models/kan/) | Kolmogorov-Arnold Network - learnable spline functions on every edge instead of fixed activations |
| [`mlp`](models/mlp/) | Standard feedforward baseline |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Then `cd` into an experiment folder and run its `run.py` - see that
experiment's own `README.md` for expected runtime and what it produces.
