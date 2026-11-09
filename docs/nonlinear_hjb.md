# Nonlinear Hamilton–Jacobi–Bellman PIDE

This experiment supports INEUS alone or a comparison with a PINN and a local
deep-BSDE baseline followed by a global solution surrogate. Its full settings
are in [configs/nonlinear.yaml](../configs/nonlinear.yaml).

## Equation and default coefficients

For centered Gaussian jump marks $E$, the implemented equation is

$$
\partial_t u+\Delta u-\eta\lVert\nabla u\rVert^2+f -\frac{\lambda}{\eta} \left(\mathbb E\left[e^{-\eta(u(t,x+E)-u(t,x))}\right]-1\right)=0, \qquad u(T,x)=\lVert x\rVert^2.
$$

| Setting | YAML field under `parameters.problem` | Default |
|---|---|---|
| State dimension | `state_dim` | 10 |
| Jump dimension | `jump_dim` | 10; must equal `state_dim` |
| Terminal time | `horizon` | 1.0 |
| Nonlinearity coefficient | `eta` | 1.0 |
| Poisson intensity | `jump_intensity` | 0.5 |
| Constant driver | `driver_constant` | 2.0 |
| Jump covariance | `jump_covariance_scale` | $0.2I_d$ |

The Brownian diffusion matrix is $\sqrt{2}I_d$. The covariance scale is a
variance, not a standard deviation. Setting `jump_intensity: 0.0` removes the
jump term and selects the corresponding HJB PDE. These options configure this
equation family; they do not select arbitrary terminal conditions or operators.

## Architecture and training

INEUS and the PINN use the same width-50 DGM value-network architecture: one
initial hidden layer and three gated stages (`n_layers: 4`). Each stage has
independent parameters, and `tanh` is applied to the gates and candidate state.
The default terminal constraint is imposed exactly through
$u_\theta(t,x)=\lVert x\rVert^2+(T-t)N_\theta(t,x)$.

Both methods use 1,000 outer epochs, batches of 1,024, and Adam with a constant
learning rate of 0.0005. The configured decay factors are 1.0. Each outer epoch
contains $(20-1)\times32=608$ main optimizer updates; an epoch is therefore
not a single gradient update. The standard configuration repeats the experiment
ten times with consecutive seeds starting at 29.

INEUS uses `h: 0.5`, `alpha: 0.4`, and a block length of ten epochs. At each
block boundary, 200 distillation steps with Adam at 0.001 fit the averaged
teacher, and the student replaces the current model. Initial collocation points
use $x\sim\mathcal N(0,0.8^2I_d)$ and uniform time sampling. Training uses a
replay buffer enriched by residual-adaptive sampling (RAD) every five epochs;
distillation draws fresh points from the base Gaussian/uniform proposal, not
from the RAD-enriched buffer. Five additional points at time zero are added
each epoch.

The comparison trains 100 independent local deep-BSDE models, each for 10,000
updates with 25 time steps, batch size 1,024, and learning rate 0.0003. The
Brownian and jump networks have three hidden layers of width 128. Their local
solution values train a terminal-constrained global surrogate with three
width-128 hidden layers, 5,000 updates, batch size 256, and learning rate 0.001.
INEUS, PINN and deep BSDE have different training objectives and update costs.

## Configuration and execution

Run from the repository root:

```bash
python -u -m ineus --config configs/nonlinear.yaml --mode ineus --output-dir results/hjb_ineus
python -u -m ineus --config configs/nonlinear.yaml --mode comparison --output-dir results/hjb_comparison
```

Edit `parameters.problem.state_dim` and `jump_dim` together for a single
dimension. In a partial YAML, omitting `jump_dim` when setting `state_dim`
automatically makes them equal. A nonempty `dimensions` list overrides both
fields for each dimension in a sweep, for example:

```yaml
parameters:
  problem:
    dimensions: [5, 10, 25, 50, 100]
    jump_intensity: 0.0
```

This block selects a no-jump sweep; `runs` applies separately to every dimension.
Alternatively, `--dimension 100 --jump-intensity 0` selects one 100D PDE and
clears any configured sweep. The [shared configuration guide](configuration.md)
explains partial YAML files, command-line precedence, seeds and output paths.

## Evaluation and outputs

All methods and repetitions use the same test set: by default, 2,000 points
uniform in $[-1.5,1.5]^d\times[0,T]$, generated with `test_seed: 2`. The
reference solution is estimated using

$
u(t,x)=f(T-t)-\frac1\eta\log\mathbb E
\left[e^{-\eta\lVert x+\sqrt2(W_T-W_t)+\sum_{j=1}^{N_{T-t}}E_j\rVert^2}\right].
$

The default reference uses 2,000,000 paths, seed 124, and chunks of 10,000 paths.
Its accuracy depends on the dimension and coefficients: it is a Monte Carlo
reference, not an exact numerical value. Memory scales with
`test_samples * reference_chunk_size * state_dim`. Reduce the chunk size to
limit memory usage; reducing the total path count instead changes reference
accuracy. The reference is calculated even in INEUS-only mode.

The principal comparison metric is test-set MAE. Each repetition writes to
`run_01/`, `run_02/`, and so on, including when `runs: 1`. INEUS-only runs save
`ineus_history.csv` and `ineus_summary.csv`; comparisons save
`three_method_history.csv`, `three_method_summary.csv`, and the local-BSDE and
surrogate outputs. At the experiment root, `ineus_summary_over_N_runs.csv` or
`summary_over_N_runs.csv` collects the repetitions, alongside aggregated
MAE-versus-runtime curves and figures.

The summary's `final_mae` fields use the last logged MAE. `best_loss_ineus` /
`best_loss_bellman` records the minimum logged squared test residual, **not** the
final INEUS regression loss; that regression loss is stored in the history's
training-loss column. The HJB trainers retain their final iterates rather than
restoring the recorded best-residual weights.

Dimension sweeps place each experiment under `dimen_DDD/` and collect scalar
results in `dimension_run_summaries.csv` and `dimension_statistics.csv` at the
sweep root. Resolved configurations record the settings used. The runner does
not automatically produce spatial/time solution-slice figures.
