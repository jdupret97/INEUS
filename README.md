# INEUS: PDE and PIDE experiments

Research implementations of INEUS for high-dimensional partial differential
equations (PDEs) and partial integro-differential equations (PIDEs).
The quadratic-linear and nonlinear HJB experiments also include PINN and
deep-BSDE comparisons.

The code separates equation definitions, neural architectures, sampling,
training and evaluation. Experiments are configured through YAML files.

## Available experiments

| Experiment | Default dimension | Runs | Methods |
|---|---|---|---|
| Quadratic-terminal linear PIDE | 10 | 10 | INEUS, PINN, deep BSDE |
| Nonlinear HJB | 10 | 10 | INEUS, PINN, deep BSDE |
| Linear Black–Scholes basket call | 10 | 10 | INEUS |
| Nonlinear Black–Scholes with default risk | 100 | 5 | INEUS |

These are the defaults in the main configurations; dimensions, coefficients
and repetition counts can be changed. Both Black–Scholes cases support
INEUS only, not the three-method comparison.

| Configuration | Experiment guide |
|---|---|
| [configs/linear.yaml](configs/linear.yaml) | [Quadratic-terminal linear PIDE](docs/linear.md) |
| [configs/nonlinear.yaml](configs/nonlinear.yaml) | [Nonlinear HJB](docs/nonlinear_hjb.md) |
| [configs/linear_black_scholes.yaml](configs/linear_black_scholes.yaml) | [Linear Black–Scholes](docs/linear_black_scholes.md) |
| [configs/black_scholes.yaml](configs/black_scholes.yaml) | [Nonlinear Black–Scholes](docs/nonlinear_black_scholes.md) |

## Installation

Use Python 3.10. From the repository root, create the supplied environment:

```bash
conda env create -f environment_hpc.yml
conda activate ineus
```

Alternatively, install the package and its dependencies into a compatible
Python environment:

```bash
python -m pip install -e .
```

Dependencies include TensorFlow, NumPy, SciPy, pandas, Matplotlib, PyYAML and
Jinja2. Package version constraints are listed in [pyproject.toml](pyproject.toml).
On HPC, retain a working cluster-compatible TensorFlow/CUDA environment and
adapt the supplied Slurm templates to its activation procedure.

## Run INEUS

Run commands from the repository root. The following examples use one
repetition with the full training settings of the selected configuration:

```bash
python -u -m ineus --config configs/linear.yaml --mode ineus --runs 1
python -u -m ineus --config configs/nonlinear.yaml --mode ineus --runs 1
python -u -m ineus --config configs/linear_black_scholes.yaml --runs 1
python -u -m ineus --config configs/black_scholes.yaml --runs 1
```

For quadratic linear and HJB, `--mode ineus` skips PINN and deep-BSDE training.
Omit `--runs 1` to use the repetition count in the YAML, or set `--runs N`
for a different count. Repeated runs use consecutive training seeds.

## Compare INEUS, PINN and deep BSDE

Comparison mode is available for the quadratic-linear and HJB experiments:

```bash
python -u -m ineus --config configs/linear.yaml --mode comparison --runs 1
python -u -m ineus --config configs/nonlinear.yaml --mode comparison --runs 1
```

Each repetition trains the three methods sequentially. INEUS and PINN start
from the same value-network weights and collocation points. The deep-BSDE
baseline trains local models, then fits a global solution surrogate to their
predictions. All methods are evaluated on the same evaluation points.

INEUS uses iterative frozen-target regression and periodic distillation;
the PINN directly minimizes a strong-equation residual. Their training
losses and per-epoch computational costs are therefore different.

## Configuration

The main YAML files list the equation, network, training and evaluation
settings. Top-level fields select the experiment and how it is executed:

```yaml
experiment: linear
mode: ineus
seed: 29
runs: 10
output_dir: results/linear_ineus
parameters: {}
```

This minimal example uses Python defaults for omitted parameters. For explicit
settings, start from one of the full configurations linked above. A partial YAML
does not inherit settings from another YAML file.

Command-line options override YAML values; omitted YAML fields use the selected
experiment's Python defaults. Common overrides are:

| Option | Purpose |
|---|---|
| `--mode ineus` / `--mode comparison` | Select the training methods. |
| `--runs N` | Set the number of repetitions per dimension. |
| `--seed N` | Set the base training seed. |
| `--output-dir PATH` | Choose where results are written. |

For example, save a single INEUS run in a separate directory:

```bash
python -u -m ineus --config configs/linear.yaml --mode ineus --runs 1 --output-dir results/linear_ineus
```

See [Configuration](docs/configuration.md) for all shared options, override
behavior and output-path conventions. Experiment-specific parameters are
explained in the individual guides.

### Dimensions and no-jump equations

Quadratic linear and HJB accept `--dimension`, `--dimensions` and
`--jump-intensity`. For example:

```bash
python -u -m ineus --config configs/linear.yaml --mode ineus --dimension 100 --jump-intensity 0 --runs 1
python -u -m ineus --config configs/nonlinear.yaml --mode ineus --dimensions 5 10 25 50 100
```

The first command selects a 100D no-jump equation; the second runs an HJB
dimension sweep with the configured repetitions at each dimension.
Ready-made sweep profiles are [linear_sweep.yaml](configs/linear_sweep.yaml)
and [nonlinear_sweep.yaml](configs/nonlinear_sweep.yaml).

For Black–Scholes, dimensions and jump settings are selected through YAML.
Methods, dimensions and repetitions run sequentially.

## Results

Each launch writes its resolved settings to `config_resolved.yaml` under
`output_dir`. Relative paths are interpreted from the current working directory.
Choose a fresh output directory for each experiment to avoid overwriting results.

- Quadratic linear and HJB save training histories, MAE summaries, runtimes and
  comparison figures, with aggregated statistics for repeated runs.
- Linear Black–Scholes saves Monte Carlo reference curves, predictions and
  per-run/per-dimension error summaries.
- Nonlinear Black–Scholes saves reference-point predictions, histories and
  model checkpoints, with separate plotting utilities.

Reference construction, diagnostic losses and model-selection behavior differ
between experiments. The guides describe their metric definitions, output
filenames and checkpoint availability. Monte Carlo references have sampling
error and can require substantial memory and computation.

## Running on HPC

The [scripts/](scripts/) directory provides a Slurm template for each experiment.
Before submission, adapt the partition/account, requested resources, wall time
and environment activation to your cluster. Submit from the repository root:

```bash
sbatch --export=ALL,MODE=ineus,RUNS=1 scripts/run_linear.slurm
sbatch --export=ALL,MODE=comparison,RUNS=1 scripts/run_nonlinear.slurm
```

The templates accept `CONFIG`, `SEED`, `RUNS` and `OUTPUT_DIR` environment
overrides; the quadratic-linear and HJB scripts also accept `MODE`.
Omitted mode, seed and run-count overrides retain the YAML settings. By default,
the scripts choose job-specific output directories under `results/`.
Full comparisons and Monte Carlo reference calculations may be expensive,
especially over several dimensions and repetitions. Resource requirements
depend on the selected experiment; the templates are starting points.

## Repository layout

```text
configs/                     YAML experiment configurations
docs/                        Shared configuration guide and four experiment guides
ineus/
  linear/                    Quadratic-terminal linear PIDE
  nonlinear/                 Nonlinear HJB
  linear_black_scholes/       Basket-call Black–Scholes
  black_scholes/              Nonlinear default-risk Black–Scholes
  cli.py                     Shared command-line interface
  configuration.py           YAML loading and configuration validation
scripts/                     Slurm templates and repository utilities
tests/                       Automated regression checks
environment_hpc.yml          Conda environment
pyproject.toml               Package metadata and dependencies
```

Within each experiment, `architectures/` defines the neural networks and
`training/` contains the method-specific optimization routines. Equation,
sampling and experiment orchestration modules are kept alongside them.
