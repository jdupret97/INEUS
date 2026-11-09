"""Fixed test data, shared reference curves and final solution-error evaluation."""

from dataclasses import dataclass

import numpy as np

from .references import build_reference_curves_bs, mc_prices_on_grid


@dataclass(frozen=True)
class EvaluationData:
    """Reference data shared by all runs of one problem dimension."""

    xtest: np.ndarray
    ttest: np.ndarray
    values: np.ndarray
    se_mc: np.ndarray
    reference_curves: dict


def build_evaluation_data(problem, evaluation_config, seed):
    """Build the dimension-specific test set and both Monte Carlo curves."""
    dimen_value = problem.dimen
    test_rng = np.random.default_rng(seed + 101 * dimen_value)
    xtest = test_rng.uniform(
        0.001, problem.max_S, (evaluation_config.test_samples, dimen_value)
    ).astype(np.float32)
    ttest = test_rng.uniform(
        0.0, problem.T, (evaluation_config.test_samples, 1)
    ).astype(np.float32)

    print(
        f"  Building fixed test-set MC values with "
        f"{evaluation_config.mc_test_paths} paths per point..."
    )
    values_local, se_local = mc_prices_on_grid(
        ttest,
        xtest,
        problem.T,
        w_val=problem.w,
        K_val=problem.K,
        r_val=problem.r,
        sigma=problem.sigma_diag,
        Sigma_val=problem.Sigma,
        lam=problem.lam_vec,
        muZ=problem.mu_j,
        sigZ=problem.sig_j,
        n_paths=evaluation_config.mc_test_paths,
        seed=seed + 202 * dimen_value,
        antithetic=True,
        S_scale=1.0,
    )
    values = values_local.reshape(-1, 1).astype(np.float32)
    se_mc = se_local.reshape(-1, 1).astype(np.float32)

    reference_curves = build_reference_curves_bs(
        T_val=problem.T,
        w_val=problem.w,
        K_val=problem.K,
        r_val=problem.r,
        sigma_diag_val=problem.sigma_diag,
        Sigma_val=problem.Sigma,
        lam_vec_val=problem.lam_vec,
        mu_j_val=problem.mu_j,
        sig_j_val=problem.sig_j,
        dimen_val=dimen_value,
        x1_min=0.001,
        x1_max=problem.max_S,
        x1_fixed=problem.K,
        x_other_fixed=problem.K,
        n_x=evaluation_config.n_x_curve,
        n_t=evaluation_config.n_t_curve,
        n_paths_curve=evaluation_config.mc_curve_paths,
        mc_seed=seed + 303 * dimen_value,
    )
    return EvaluationData(xtest, ttest, values, se_mc, reference_curves)


def evaluate_mae_single_run_bs(model1, xtest_val, ttest_val, values_val, ref_curves):
    x1_grid = ref_curves["x1_grid"]
    x_curve_t0 = ref_curves["x_curve_t0"]
    t0 = ref_curves["t0"]
    true_t0 = ref_curves["true_t0"].reshape(-1)
    se_t0 = ref_curves["se_t0"].reshape(-1)

    pred_t0 = model1([x_curve_t0, t0], training=False).numpy().reshape(-1)
    mae_t0_curve = float(np.mean(np.abs(pred_t0 - true_t0)))

    t_grid = ref_curves["t_grid"]
    t_curve = ref_curves["t_curve"]
    x_time_curve = ref_curves["x_time_curve"]
    true_time = ref_curves["true_time"].reshape(-1)
    se_time = ref_curves["se_time"].reshape(-1)

    pred_time = model1([x_time_curve, t_curve], training=False).numpy().reshape(-1)
    mae_time_curve = float(np.mean(np.abs(pred_time - true_time)))

    pred_full = model1([xtest_val, ttest_val], training=False).numpy().reshape(-1)
    true_full = values_val.reshape(-1)
    mae_full = float(np.mean(np.abs(pred_full - true_full)))

    return {
        "x1_grid": x1_grid,
        "pred_t0": pred_t0,
        "true_t0": true_t0,
        "se_t0": se_t0,
        "t_grid": t_grid,
        "pred_time": pred_time,
        "true_time": true_time,
        "se_time": se_time,
        "mae_t0_curve": mae_t0_curve,
        "mae_time_curve": mae_time_curve,
        "mae_full": mae_full,
        "x1_fixed": ref_curves["x1_fixed"],
        "x_other_fixed": ref_curves["x_other_fixed"],
    }
