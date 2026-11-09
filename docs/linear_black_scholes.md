# Linear Black–Scholes basket-call PIDE

This experiment trains INEUS for a linear Black–Scholes/Merton jump-diffusion
PIDE with an equally weighted basket-call terminal payoff. It is separate from
the quadratic-terminal `linear` experiment and the nonlinear default-risk
`black_scholes` experiment. Its configuration key is `linear_black_scholes`.
Only INEUS is implemented here; there are no PINN or neural deep-BSDE baselines
for this case.

## Equation and default coefficients

For asset prices $S=(S_1,\ldots,S_d)$, the equation is

$$
\partial_t u + \sum_i(r-\lambda_i\kappa_i)S_i\partial_{S_i}u + \frac{1}{2}\sum_{i,j}\sigma_i\sigma_j\Sigma_{ij}S_iS_j \partial_{S_iS_j}u - ru + \sum_i\lambda_i\mathbb E[u(t,S^{(i,Z_i)})-u(t,S)] = 0,
$$

with $u(T,S)=(\sum_i w_iS_i-K)_+$, $w_i=1/d$,
$Z_i\sim\mathcal N(\mu_i,s_i^2)$, and
$\kappa_i=\exp(\mu_i+s_i^2/2)-1$.
Here $S^{(i,Z_i)}$ multiplies only coordinate $i$ by $e^{Z_i}$.
Each coordinate has its own independent Poisson jump process; Brownian
coordinates are correlated through $\Sigma$.

The default configuration runs dimension 10 with ten independent seeds.
The `problem.dimensions` list can also specify a dimension sweep, for example
`[5, 10, 25, 50, 100]`, retaining the configured run count for each dimension.
The equation uses $T=1$, $r=0.05$, $K=30$, $\sigma_i=0.15$,
$\lambda_i=0.5$, $\mu_i=0.20$, $s_i=0.30$, and Brownian correlation 0.10
off the diagonal. The strike is configured through
`problem.K_multiplier * problem.max_S` (0.3 times 100 by default).

## Architecture and training

The width-50 DGM has three configured layers: one initial state and two gated
updates. Each update has independent gate parameters, and `tanh` is used
for the initial state and all gates, including H. As in the quadratic-terminal
linear model, parameters are not shared across stages; unlike that model,
H retains its `tanh` activation. Input projections have biases, while state
projections do not.

The value network enforces the terminal payoff exactly:

$$
u_\theta(t,S)=\left(\sum_iw_iS_i-K\right)_++(T-t)N_\theta(t,S).
$$

The default terminal-loss weight is therefore zero. Input normalization uses
asset bounds `[0, 100]` and time bounds `[0, 1]`; path samples can exceed those
asset bounds and are not clipped.

The training profile uses:

- 1,000 Adam outer epochs with constant learning rate 0.0005.
- `n_inner: 20`, 32 minibatches per inner step, and batch size 1,024, giving
  `(20 - 1) * 32 = 608` main optimizer updates per outer epoch.
- `h_scale: 0.5`, `relaxation: 0.4`: the inner increment is `0.5 / 19`,
  and the block-averaging weight is `0.4 / 0.5 = 0.8`.
- Exact jump-diffusion marginal sampling from the initial state
  `(K, ..., K)`, with 2,048 initial points and a replay capacity of 4,096.
- Residual-adaptive sampling every five epochs, from 16,384 path candidates,
  with 25 jump draws per candidate and replacement of points through the
  replay buffer.
- Block distillation every ten epochs, with 200 Adam steps at constant
  learning rate 0.001. The student replaces the current and target models.

An optional second phase uses Adagrad at constant learning rate 0.001 for the
same number of outer epochs; it is disabled by default. Enable it with
`train.run_phase2: true`. Each phase initializes its own optimizer and replay
data. The second phase retains the trained model weights.

INEUS trains against its iterative frozen-network target. The diffusion/drift
operator uses nested automatic differentiation with respect to an auxiliary
scalar, not a full explicitly formed spatial Hessian. Strong-residual
diagnostics and residual-adaptive sampling are also retained.
`train.nc_samples` is recorded configuration metadata and is not used by
the training calculation. `problem.sample_b` bounds the optional uniform
sampler; the configured training workflow uses path sampling.

## Configuration and execution

The configuration is
[linear_black_scholes.yaml](../configs/linear_black_scholes.yaml), using
experiment key `linear_black_scholes`. From the repository root:

```bash
python -u -m ineus --config configs/linear_black_scholes.yaml
```

Use `--runs 1` for one seed. Set `parameters.problem.dimensions` in YAML to
select a different dimension or a sweep. For example, a partial YAML can contain:

```yaml
experiment: linear_black_scholes
mode: ineus
seed: 23
runs: 10
output_dir: results/linear_black_scholes/sweep
parameters:
  problem:
    dimensions: [5, 10, 25, 50, 100]
```

Omitted fields keep their defaults. Set `parameters.problem.jump_intensity: 0.0`
for the diffusion-only PDE. The module entry point
`python -m ineus.linear_black_scholes` also selects this experiment directly.
See the [configuration guide](configuration.md) for shared YAML and CLI options.

For HPC, adapt the resource directives and environment activation in
[run_linear_black_scholes.slurm](../scripts/run_linear_black_scholes.slurm),
then submit from the repository root:

```bash
sbatch scripts/run_linear_black_scholes.slurm
```

Dimensions and repetitions run sequentially, not across multiple GPUs or nodes.
A full sweep is expensive; choose resources and wall time for both training
and reference construction. The supplied environment includes Jinja2, required
for pandas' LaTeX-table export.

## Evaluation and outputs

For each dimension, one fixed set of 2,000 test points is drawn uniformly from
`[0.001, max_S]^d` and `[0, T]`. Monte Carlo prices use 20,000 terminal paths
per point by default. Two 401-point reference curves are built:

1. `t = 0`, varying the first asset from 0.001 to `max_S`, with the others at K.
2. Time varying from 0 to T, with every asset fixed at K.

These references are built once per dimension and shared across its runs.
With base seed `seed`, test coordinates use `seed + 101*d`, test-reference
Monte Carlo uses `seed + 202*d`, and curves use `seed + 303*d` and the next
seed. Independent run seeds are `seed`, `seed + 1`, and so on. All dimensions
use that same run-seed sequence.

The reference prices use the exact terminal jump-diffusion distribution and
antithetic Brownian draws. They still have Monte Carlo error. The saved
`reference_mc_stderr` is the discounted payoff sample standard deviation
divided by the square root of the path count. It does not account for
dependence within the antithetic pairs and should not be treated as a
calibrated confidence interval without a separate statistical check.

Reported solution errors are mean absolute errors on the fixed test set and
the two reference curves. They are not relative L2 or maximum errors.
`last_epoch_pde_loss` is the iterative-target mean-square regression loss;
`last_epoch_loss` is a mean absolute strong-residual diagnostic using 50 jump
draws per test point. The console labels these quantities `PDE Loss` and
`Test Loss`, respectively; they are different objectives, not train/test
versions of one loss. Training time includes each training phase and its
diagnostics, but excludes initial Monte Carlo reference construction and
the final curve evaluation and file writes.

### Output files

Every invocation saves `config_resolved.yaml`. Each dimension/run has:

```text
dimen_010/run_01/
  curve_t0.csv
  curve_xfixed_time.csv
  run_summary.csv
  run_outputs.npz
```

Curve CSVs contain predictions, Monte Carlo references, their reported
standard errors and absolute errors. The NPZ stores curve coordinates,
predictions, references and reference standard errors. No trained-model
checkpoint or complete epoch history is saved for this experiment.

The default `evaluation.output_tag: bs_dimension_sweep` produces these files
at the output root:

```text
all_runs_bs_dimension_sweep.csv
dimension_summary_bs_dimension_sweep.csv
dimension_summary_bs_dimension_sweep.tex
```

The raw run table is updated after each completed run. The final summary
groups by dimension and reports run count, mean/sample standard deviation of
test MAE, iterative-target loss and training time, plus mean curve errors.
With one run, sample standard deviations are undefined and appear as missing
values. Use a new output directory for each saved sweep; existing result
paths can be overwritten.
