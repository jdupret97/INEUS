"""Monte Carlo basket-call references for the linear Black–Scholes PIDE."""

import numpy as np


def mc_basket_call_jumpdiff(
    S_t,
    t,
    T_val,
    w_val,
    K_val,
    r_val,
    sigma,
    Sigma_val,
    lam,
    muZ,
    sigZ,
    n_paths=200_000,
    seed=123,
    antithetic=True,
):
    S_t = np.asarray(S_t, dtype=float)
    w_val = np.asarray(w_val, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    Sigma_val = np.asarray(Sigma_val, dtype=float)
    lam = np.asarray(lam, dtype=float)
    muZ = np.asarray(muZ, dtype=float)
    sigZ = np.asarray(sigZ, dtype=float)

    d = S_t.shape[0]
    tau = float(T_val - t)
    if tau <= 0.0:
        payoff = max(np.dot(w_val, S_t) - K_val, 0.0)
        return payoff, 0.0

    rng = np.random.default_rng(seed) if seed is not None else np.random

    kappa = np.exp(muZ + 0.5 * (sigZ**2)) - 1.0
    L = np.linalg.cholesky(Sigma_val)

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal(size=(n_half, d))
        Z = np.vstack([Z, -Z])
        n_paths = Z.shape[0]
    else:
        Z = rng.standard_normal(size=(n_paths, d))

    Z_corr = Z @ L.T
    diff = (sigma * np.sqrt(tau))[None, :] * Z_corr

    N = rng.poisson(lam * tau, size=(n_paths, d))
    G = rng.standard_normal(size=(n_paths, d))
    sumZ = N * muZ[None, :] + np.sqrt(N) * sigZ[None, :] * G

    drift = ((r_val - lam * kappa) - 0.5 * sigma**2) * tau
    log_ST = np.log(S_t)[None, :] + drift[None, :] + diff + sumZ
    S_T = np.exp(log_ST)

    basket = S_T @ w_val
    payoff = np.maximum(basket - K_val, 0.0)

    disc = np.exp(-r_val * tau)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr


def mc_prices_on_grid(
    ttest_val,
    xtest_val,
    T_val,
    w_val,
    K_val,
    r_val,
    sigma,
    Sigma_val,
    lam,
    muZ,
    sigZ,
    n_paths=50_000,
    seed=123,
    antithetic=True,
    S_scale=1.0,
):
    """
    MC reference prices for each (t, S_t) in a grid.

    Returns:
        prices:  shape (N,)
        stderr:  shape (N,)
    """
    ttest_val = np.asarray(ttest_val, dtype=float).reshape(-1, 1)
    xtest_val = np.asarray(xtest_val, dtype=float)
    N, _ = xtest_val.shape

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(N)

    prices = np.empty(N, dtype=float)
    stderrs = np.empty(N, dtype=float)

    for n in range(N):
        if (n + 1) % 250 == 0:
            print(f"    MC reference point {n+1}/{N}")

        t_val = float(ttest_val[n, 0])
        S_t = S_scale * xtest_val[n, :]

        price, se = mc_basket_call_jumpdiff(
            S_t=S_t,
            t=t_val,
            T_val=T_val,
            w_val=w_val,
            K_val=K_val,
            r_val=r_val,
            sigma=sigma,
            Sigma_val=Sigma_val,
            lam=lam,
            muZ=muZ,
            sigZ=sigZ,
            n_paths=n_paths,
            seed=int(child_seeds[n].generate_state(1)[0]),
            antithetic=antithetic,
        )
        prices[n] = price
        stderrs[n] = se

    return prices, stderrs


def build_reference_curves_bs(
    T_val,
    w_val,
    K_val,
    r_val,
    sigma_diag_val,
    Sigma_val,
    lam_vec_val,
    mu_j_val,
    sig_j_val,
    dimen_val,
    x1_min=0.001,
    x1_max=100.0,
    x1_fixed=None,
    x_other_fixed=None,
    n_x=401,
    n_t=401,
    n_paths_curve=50_000,
    mc_seed=1234,
):
    """
    Build the two reference curves used for all runs of a fixed dimension:
      1) t=0, x=(x1, x_other_fixed, ..., x_other_fixed)
      2) x fixed, t in [0,T]
    """
    if x1_fixed is None:
        x1_fixed = K_val
    if x_other_fixed is None:
        x_other_fixed = K_val

    print(f"  Building reference curves with {n_paths_curve} MC paths per point...")

    x1_grid = np.linspace(x1_min, x1_max, n_x, dtype=np.float32)
    x_curve_t0 = np.full((n_x, dimen_val), x_other_fixed, dtype=np.float32)
    x_curve_t0[:, 0] = x1_grid
    t0 = np.zeros((n_x, 1), dtype=np.float32)

    true_t0, se_t0 = mc_prices_on_grid(
        t0,
        x_curve_t0,
        T_val,
        w_val=w_val,
        K_val=K_val,
        r_val=r_val,
        sigma=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam=lam_vec_val,
        muZ=mu_j_val,
        sigZ=sig_j_val,
        n_paths=n_paths_curve,
        seed=mc_seed,
        antithetic=True,
        S_scale=1.0,
    )

    t_grid = np.linspace(0.0, T_val, n_t, dtype=np.float32)
    t_curve = t_grid.reshape(-1, 1)

    x_time_curve = np.full((n_t, dimen_val), x_other_fixed, dtype=np.float32)
    x_time_curve[:, 0] = np.float32(x1_fixed)

    true_time, se_time = mc_prices_on_grid(
        t_curve,
        x_time_curve,
        T_val,
        w_val=w_val,
        K_val=K_val,
        r_val=r_val,
        sigma=sigma_diag_val,
        Sigma_val=Sigma_val,
        lam=lam_vec_val,
        muZ=mu_j_val,
        sigZ=sig_j_val,
        n_paths=n_paths_curve,
        seed=mc_seed + 1,
        antithetic=True,
        S_scale=1.0,
    )

    return {
        "x1_grid": x1_grid,
        "x_curve_t0": x_curve_t0,
        "t0": t0,
        "true_t0": true_t0.astype(np.float32),
        "se_t0": se_t0.astype(np.float32),
        "t_grid": t_grid,
        "t_curve": t_curve,
        "x_time_curve": x_time_curve,
        "true_time": true_time.astype(np.float32),
        "se_time": se_time.astype(np.float32),
        "x1_fixed": float(x1_fixed),
        "x_other_fixed": float(x_other_fixed),
    }
