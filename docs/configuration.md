# Configuration

Each experiment has a main YAML file and a guide covering its equation,
architecture, training settings and outputs:

| Experiment | YAML | Guide |
|---|---|---|
| Quadratic-terminal linear PIDE | [linear.yaml](../configs/linear.yaml) | [Linear PIDE](linear.md) |
| Nonlinear HJB | [nonlinear.yaml](../configs/nonlinear.yaml) | [HJB](nonlinear_hjb.md) |
| Linear Black–Scholes | [linear_black_scholes.yaml](../configs/linear_black_scholes.yaml) | [Linear Black–Scholes](linear_black_scholes.md) |
| Nonlinear Black–Scholes | [black_scholes.yaml](../configs/black_scholes.yaml) | [Nonlinear Black–Scholes](nonlinear_black_scholes.md) |

## YAML structure

| Field | Purpose |
|---|---|
| `experiment` | `linear`, `nonlinear` (HJB), `linear_black_scholes` or `black_scholes`. |
| `mode` | `ineus` or `comparison`; both Black–Scholes cases accept only `ineus`. |
| `seed` | Base training seed. |
| `runs` | Number of seeded repetitions per selected dimension. |
| `output_dir` | Directory for configuration snapshots and results. |
| `parameters` | Experiment-specific problem, model, training and evaluation settings. |

For example, a minimal quadratic-linear configuration is:

```yaml
experiment: linear
mode: ineus
seed: 29
runs: 1
output_dir: results/linear_ineus
parameters: {}
```

An empty or partial `parameters` section uses the selected experiment's Python
defaults for omitted fields. It does not inherit values from another YAML file.
Use the main YAML files above when you want all settings explicitly listed.

## Command-line overrides

Run from the repository root:

```bash
python -u -m ineus --config configs/linear.yaml --mode ineus --runs 1 --output-dir results/linear_ineus
```

The precedence is **command-line overrides → YAML values → Python defaults**.

| Option | Effect |
|---|---|
| `--config PATH` | Select a YAML file. |
| `--experiment NAME` | Select an experiment without YAML; must agree with the YAML if both are supplied. |
| `--mode ineus` / `--mode comparison` | Select the training methods. |
| `--seed N` | Set the base training seed. |
| `--runs N` | Set the number of repetitions. |
| `--epochs N` | Override training-loop lengths as described below. |
| `--output-dir PATH` | Set the result directory. |

For quadratic linear and HJB, `--epochs` changes INEUS/PINN epochs, each local
BSDE's iteration count and the global surrogate's iteration count. These are
different training units, not equal computational budgets. For Black–Scholes,
it changes the INEUS epoch count. It does not reduce Monte Carlo reference paths
or distillation steps.

Quadratic linear and HJB also accept `--dimension`, `--dimensions` and
`--jump-intensity`. Their guides explain dimension sweeps and no-jump settings.
For Black–Scholes, select dimensions and equation coefficients through YAML.

The experiment-specific entry points also accept configuration options:
`python -m ineus.linear`, `python -m ineus.nonlinear`,
`python -m ineus.linear_black_scholes` and `python -m ineus.black_scholes`.
Use `python -m ineus --help` for the complete option list.

## Repetitions and outputs

Repetitions use consecutive training seeds starting at `seed`; reference sampling
may use separate fixed seeds, as described in each experiment guide. In a dimension
sweep, `runs` applies to each dimension. Methods, repetitions and dimensions are
executed sequentially.

Relative paths are resolved from the working directory, not from the YAML file's
location. Each launch saves its resolved settings as `config_resolved.yaml`.
Choose a fresh output directory to avoid overwriting results. Output filenames
and aggregation formats are documented in the experiment guides.

The loader rejects unknown or duplicate keys, incompatible types and invalid
settings. Equation-specific restrictions are described alongside each experiment.
