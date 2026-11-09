# Nonlinear Black–Scholes default-risk experiment

This experiment uses INEUS for a 100-dimensional nonlinear Black–Scholes equation
with a minimum terminal payoff and solution-dependent default intensity. Both the
diffusion-only PDE and multiplicative-jump PIDE are supported. PINN and neural
deep-BSDE baselines are not implemented here; regression Monte Carlo utilities
provide approximate references, not trained neural baselines.

## Equation and default coefficients

The configured problem is

$$
\partial_t u + \mu\sum_{i=1}^{d}x_i\partial_{x_i}u
+ \frac{\sigma^2}{2}\sum_{i=1}^{d}x_i^2\partial_{x_ix_i}u
- [(1-\delta)Q(u)+R]u
+ \lambda\mathbb{E}[u(t,x\odot e^Z)-u(t,x)] = 0,
\qquad u(T,x)=\min_i x_i.
$$

The intensity is $Q(u)=\gamma_h$ below $v_h$, $Q(u)=\gamma_l$ at or
above $v_l$, and linear between the two thresholds. Jump components are

$$
Z_i=\mu_J+\sigma_J\bigl(\sqrt{\rho_J}\xi_0
+\sqrt{1-\rho_J}\xi_i\bigr),
$$

with independent standard-normal $\xi_0,\xi_1,\ldots,\xi_d$. Diffusion
coordinates have identity correlation. Set `parameters.problem.use_jumps: false`
to remove the nonlocal term. The drift used by the operator is `mu_bar`; no
additional jump-compensator drift adjustment is applied.

The full configuration uses $d=100$, $T=1$, $x_0=(100,\ldots,100)$,
$\mu=R=0.02$, $\sigma=0.20$, $\delta=2/3$, $(v_h,v_l)=(50,70)$,
$(\gamma_h,\gamma_l)=(0.20,0.02)$ and, when enabled,
$(\lambda,\mu_J,\sigma_J,\rho_J)=(0.1,-0.2,0.15,0.5)$.

## Architecture and training

The network is a normalized DGM value model with width 100 and three configured
layers (one initial state and two gated updates). Gate parameters are shared
between those two updates, with `tanh` activations. The default uses a soft terminal
loss with `model.constrained: false` and `train.w_bc: 1`. An optional wrapper
$u_\theta(t,x)=\min_i x_i+(T-t)N_\theta(t,x)$ enforces the terminal payoff;
use `model.constrained: true` and `train.w_bc: 0` for this setting.

INEUS fits the value network to the frozen-target Euler update

$$
u_{\rm targ}+\beta\bigl[\mathcal A_0u_{\rm targ}
-((1-\delta)Q(u_{\rm targ})+R)u_{\rm targ}
+\mathcal J[u_{\rm targ}]\bigr],\qquad
\beta=h/(n_{\rm inner}-1).
$$

Here $\mathcal A_0$ includes the time derivative and the diffusion/drift operator.
It is computed as the second derivative at zero of a scalar auxiliary function
$\psi(h)$, using nested automatic-differentiation tapes. This avoids explicitly
forming the full spatial Hessian, but it is **not** a derivative-free operator
evaluation. The jump operator uses one fresh correlated Gaussian jump draw per
collocation point. Its single-draw value is a stochastic estimate of the nonlocal
expectation.

The default training settings are:

- 400 outer epochs, `n_inner: 20`, 32 minibatches per inner step and batch size
  1,024: $(20-1)\times32=608$ Adam updates per outer epoch.
- `h_scale: 1`, so the inner increment is $1/19$. With `zeta_scale: 1.25`,
  the relaxation numerator is $\zeta=h/1.25=0.8$ and the block-averaging
  mixing weight is $\zeta/h=0.8$ (named `alpha` in the code).
- Initial learning rate 0.001, cosine decay to 2% of that value, and gradient
  clipping by global norm at 1.0. Exponential decay is available through
  `train.use_cosine_lr: false`.
- 2,048 initial collocation points and a FIFO replay capacity of 16,384.
  Each subsequent epoch adds `num_samples // 4` fresh points.
- Path-based sampling enabled. With jumps, half the samples are GBM states and
  half are GBM states multiplied by one sampled jump. This is a training-point
  mixture, not a simulation of the compound-Poisson process with intensity
  $\lambda$.
- Block averaging every 10 epochs. The student is
  initialized from a weighted average of model weights and then trained for
  200 steps at learning rate 0.0001 to fit the function-space ensemble. The
  current and target networks are replaced by the student.

RAD is disabled by default. Its supported configuration uses uniform sampling,
not the path-based sampler. The fixed interior test set follows diffusion-only
GBM paths even when jump-augmented training is enabled.

## Configuration and execution

The configuration is [black_scholes.yaml](../configs/black_scholes.yaml), using
experiment key `black_scholes`. From the repository root:

```bash
python -u -m ineus --config configs/black_scholes.yaml --mode ineus
```

The default is five independent runs starting at seed 1. Repeated runs use seeds
`seed`, `seed + 1`, ..., in `run_01`, `run_02`, ... directories. Each run draws
its fixed test points before reseeding for training. With `--runs 1`, files are
written directly in the output directory. Only `mode: ineus` is supported.

For HPC, adapt the resources and environment in
[run_black_scholes.slurm](../scripts/run_black_scholes.slurm) before submission.
See the [configuration guide](configuration.md) for shared YAML and CLI options.

## Evaluation and outputs

Each run saves `hpinn_default_risk.csv` with the reference-point prediction and
optional absolute error, `final.pkl`, and completed-block checkpoints such as
`epoch_0010.pkl`. `history.json` and checkpoint histories contain student metrics
at completed distillation blocks. If training stops between blocks, the final
weights may be newer than the last recorded student prediction.

Repeated-run child directories also contain `config_resolved.yaml` with the
actual seed and output path. Root `run_summaries.csv` is updated after each
completed run; `run_statistics.csv` gives means, sample standard deviations
(`ddof=1`) and nonmissing counts. With fewer than two observations, standard
deviations are empty. These are across-run statistics, not confidence intervals.

`train_loss` is a mean-square Euler-target regression loss, with the terminal
penalty added in soft-terminal mode. `student_pde_loss` is the mean absolute
operator diagnostic on the fixed test set; with jumps it uses a single random
jump estimate per point. `student_test_loss` combines that diagnostic with a
mean-square terminal penalty. These quantities have different definitions and
must not be interpreted as the same loss evaluated on train/test data. They are
not relative $L^2$ or $L^\infty$ solution errors.

The point references at `t = 0`, `x = (100, ..., 100)` are `57.300` without
jumps and `55.810` with jumps. If `evaluation.u_ref` is omitted, the loader
selects the jump value only when jumps are enabled and intensity is positive.
These benchmarks apply to the default 100D equation, not arbitrary coefficients.
Override `evaluation.u_ref` for a modified equation, or set it to `null` to
disable reference errors. Resolved YAML stores the selected numeric value;
omit it again to restore automatic selection.

### Checkpoints

Checkpoints contain weights, history and epoch number, but not optimizer,
replay-buffer or random-generator state. `train.resume_path` therefore does not
reproduce an uninterrupted training trajectory. Checkpoints use pickle: load
only trusted files. Resume is supported only with `--runs 1`.

### Approximate reference calculations

The `references` module provides backward least-squares regression and forward
Picard iteration. Both use ten polynomial features of the current minimum and
time. This is an approximate reduced feature set: exchangeability of the asset
coordinates does not make the current minimum a sufficient state statistic.
They use antithetic GBM increments and, with jumps enabled, a Bernoulli
approximation allowing at most one event per time step.

```bash
python -m ineus.black_scholes.references --config configs/black_scholes.yaml --output results/black_scholes/reference.json
```

Defaults are 400,000 paths, 1,000 backward-regression time steps and 100 Picard
time steps with up to 30 iterations. These require substantial time and memory.
Without `--config`, the reference CLI uses the **no-jump** equation.

The printed path standard errors/intervals do not include regression error,
time-discretisation bias, or fixed-point error, and the antithetic samples are
dependent. In backward regression, the common initial state can give an almost
zero cross-path standard error even when the estimate is biased. Use independent
repetitions and numerical convergence checks before assigning reference accuracy.

### Plotting

Use the resolved YAML that produced a trusted checkpoint. For the default run:

```bash
python -m ineus.black_scholes.plotting --config results/black_scholes/run_01/config_resolved.yaml --checkpoint results/black_scholes/run_01/final.pkl --output-dir figures/black_scholes/run_01
```

To aggregate prediction histories across the repeated runs:

```bash
python -m ineus.black_scholes.aggregation --config results/black_scholes/run_01/config_resolved.yaml --base-dir results/black_scholes --seeds 1 2 3 4 5 --prediction-only --output-dir figures/black_scholes
```

The aggregator maps seeds to `run_XX` directories using `run_summaries.csv`,
falling back to `seed_<seed>` for separately launched runs. Pass actual seeds,
not run indices, to `--seeds`.

Predictions align at common distillation epochs. Plot bands show population
standard deviation (`ddof=0`), unlike the sample standard deviation in CSV
summaries (`ddof=1`); neither is a confidence interval. For loss plots, omit
`--prediction-only` and provide an `.out` console log in each run directory;
the driver does not split its log automatically. Plotting `--u-ref` overrides
the reference line without changing stored results.
