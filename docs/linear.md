# Quadratic-terminal linear PIDE

This experiment compares INEUS, PINN and local deep BSDE with a global
surrogate for a linear equation with quadratic terminal data.
The main configuration is [configs/linear.yaml](../configs/linear.yaml).

## Equation and default coefficients

For state dimension $d$ and Brownian dimension $q$, the equation is

$$
\partial_t u+\beta x\cdot\nabla u
+\frac12\operatorname{Tr}(\Sigma\Sigma^\top D^2u)
+\lambda\,\mathbb E[u(t,x+E)-u(t,x)]-cu=0,
\qquad u(T,x)=\|x\|^2,
$$

where $E\sim\mathcal N(0,\Sigma_J)$. Defaults are $d=q=10$,
$T=0.5$, $\beta=1$, $c=2$, and $\lambda=0.25$.

The diffusion matrix has independent Gaussian entries with standard deviation
`diffusion_scale: 0.3`. The jump covariance is diagonal, with entries sampled
uniformly between `jump_covariance_min: 0.01` and
`jump_covariance_scale: 0.4`. Both are generated using `coefficient_seed: 23`,
independently of training seeds, and remain fixed across repetitions at a
given dimension. Covariance entries are variances, not standard deviations.

The YAML also supports `diffusion_mode: constant`, which sets every diffusion
entry to `diffusion_scale`, and `jump_covariance_mode: isotropic`, which uses
`jump_covariance_scale` times the identity. Setting `jump_intensity: 0.0`
removes the jump term. These options parameterize this equation family;
they do not replace its quadratic terminal condition.

Evaluation uses the exact solution. With $\tau=T-t$ and
$A=\|\Sigma\|_F^2+\lambda\operatorname{Tr}(\Sigma_J)$,

$$
u(t,x)=e^{(2\beta-c)\tau}\|x\|^2
+e^{-c\tau}A\frac{e^{2\beta\tau}-1}{2\beta}.
$$

For zero drift, the last fraction is replaced by $\tau$.

## Architecture and training

INEUS and PINN use a width-50 DGM network with four hidden layers: an initial
layer and three gated stages. Each stage has independent parameters. The
initial layer and gates use `tanh`; the candidate state $H$ is unactivated.
The default wrapper enforces the terminal condition exactly through
$u_\theta(t,x)=\|x\|^2+(T-t)N_\theta(t,x)$.

| Setting | Default |
|---|---|
| Independent repetitions | 10, starting at seed 29 |
| INEUS / PINN epochs | 1,000 each |
| Main optimizer | Adam, constant learning rate 0.0005 |
| Batch size | 1,024 |
| Inner grid / batches | `n_inner: 20`, `num_batches: 32` |
| INEUS scale / relaxation | $h=0.5$, relaxation 0.4 |
| Collocation / replay capacity | 2,048 / 4,096 points |
| RAD refresh | Every 5 epochs |
| INEUS distillation | Every 10 epochs, 200 Adam steps at learning rate 0.001 |
| PINN jump samples | 64 per training residual |
| RAD / evaluation jump samples | 25 / 50 |

Each outer epoch performs $(20-1)\times32=608$ main optimizer updates.
INEUS uses an inner step of $0.5/19$ and a block-averaging weight of
$0.4/0.5=0.8$. Its regression target uses a frozen network and one fresh
Gaussian jump mark per point; PINN directly minimizes a squared strong residual.

The base proposal is uniform on $[0,T]\times[-1.75,1.75]^d$. RAD adds
residual-selected points to the replay buffers used for training. Distillation
instead draws fresh points from the base proposal, not from the RAD-enriched
buffer. The student replaces the current network after each completed block.

Comparison mode starts INEUS and PINN from the same weights and collocation
points. The deep-BSDE baseline trains 100 local models, each with 10,000 Adam
updates at learning rate 0.0003, batch size 1,024 and 25 time steps. Its control
networks have three hidden layers of width 128, and its jump expectation uses
64 samples. A width-128, three-hidden-layer global surrogate then trains for
5,000 epochs at learning rate 0.001 and batch size 256.

## Configuration and execution

Run these commands from the repository root:

```bash
# INEUS only
python -u -m ineus --config configs/linear.yaml --mode ineus --output-dir results/linear_ineus

# INEUS, PINN and deep BSDE
python -u -m ineus --config configs/linear.yaml --mode comparison --output-dir results/linear_comparison
```

Both commands retain the configured ten repetitions. Use `--runs N` to change
that count. Equation settings live in `parameters.problem`; the remaining
sections configure `model`, `ineus`, `pinn`, `bsde` and `evaluation`.

For one dimension, set `state_dim` and leave `dimensions: []`. For a sweep,
set a list such as `dimensions: [10, 25, 50, 100, 150]`; `brownian_dim` stays
fixed across that sweep. State and Brownian dimensions need not be equal.
The equivalent command-line options are:

```bash
# 100-dimensional no-jump equation, with Brownian dimension from the YAML
python -u -m ineus --config configs/linear.yaml --mode ineus --dimension 100 --jump-intensity 0 --output-dir results/linear_100d_no_jumps

# A dimension sweep
python -u -m ineus --config configs/linear.yaml --mode ineus --dimensions 10 25 50 100 150 --output-dir results/linear_sweep
```

See the [shared configuration guide](configuration.md) for override precedence,
seeds and output paths. Full comparisons repeat the local-BSDE training budget
for every run and dimension.

## Evaluation and outputs

Each repetition evaluates all methods on the same 2,000-point set sampled
uniformly from $[0,T]\times[-1.5,1.5]^d$, using exact solution values.
The main outputs are:

| Output | Contents |
|---|---|
| `config_resolved.yaml` | Settings after defaults and overrides |
| `ineus_history.csv`, `ineus_summary.csv` | INEUS-only histories and scalar summaries |
| `three_method_history.csv`, `three_method_summary.csv` | Comparison histories, errors, losses and runtimes |
| MAE-versus-time figures | Training progress for the selected mode |
| `run_summaries.csv`, `run_statistics.csv` | Repetition summaries and sample standard deviations when `runs > 1` |
| `dimension_run_summaries.csv`, `dimension_statistics.csv` | Per-dimension sweep summaries |

Repeated runs use `run_01`, `run_02`, etc.; sweeps add a `dimen_DDD` directory
for each dimension. Deep-BSDE label and surrogate outputs are saved separately
within each comparison run.

INEUS `train_loss` is the iterative regression loss. The diagnostic
`test_residual` / `test_rhs_square_bellman` is a mean squared strong residual;
`best_loss_ineus` / `best_loss_bellman` is its minimum logged value, not the
final regression loss. INEUS and PINN restore the best-residual weights after
training, while summary MAEs remain the last logged MAEs before restoration.
Those summary MAEs are therefore not fresh evaluations of the restored weights.
