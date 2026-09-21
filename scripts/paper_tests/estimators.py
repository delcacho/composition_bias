"""
estimators.py -- covariance estimators compared in the paper.

All estimators are computed in a strictly causal way (no lookahead).
All return pd.Series indexed by date.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# ── 1. Ground truth ───────────────────────────────────────────────────────────

def true_cov(w_i: np.ndarray, w_j: np.ndarray, Sigma: np.ndarray) -> float:
    """w_i' Sigma w_j using the *known* asset covariance matrix."""
    return float(w_i @ Sigma @ w_j)


def true_cov_series(
    weights_i: pd.DataFrame,
    weights_j: pd.DataFrame,
    Sigma: np.ndarray,
) -> pd.Series:
    """Point-in-time true covariance using known Sigma (the ground truth, for benchmarking)."""
    idx = weights_i.index
    vals = np.array([
        float(weights_i.iloc[t].values @ Sigma @ weights_j.iloc[t].values)
        for t in range(len(idx))
    ])
    return pd.Series(vals, index=idx, name="true_cov")


# ── 2. Position-implied covariance (the holdings-based estimator) ─────────────

def pic_series(
    weights_i: pd.DataFrame,
    weights_j: pd.DataFrame,
    asset_returns: pd.DataFrame,
    min_obs: int = 63,
    ewma_halflife: int = 0, # 0 = plain sample cov
) -> pd.Series:
    """
    Position-implied covariance: w_i(t)' Sigma_hat(t) w_j(t)

    Sigma_hat estimated from all asset returns up to t (no lookahead).
    Optionally use EWMA-weighted sample cov (ewma_halflife > 0).
    """
    cols = weights_i.columns
    R = asset_returns.reindex(index=weights_i.index, columns=cols).fillna(0.0).values
    Wi = weights_i.values
    Wj = weights_j.values
    T = len(weights_i)
    result = np.full(T, np.nan)

    for t in range(min_obs, T):
        R_hist = R[:t]
        if ewma_halflife > 0:
            decay = 0.5 ** (1.0 / ewma_halflife)
            # weights: newest observation gets weight 1, oldest gets decay^(t-1)
            ages = np.arange(t - 1, -1, -1, dtype=float)
            w = decay ** ages
            w /= w.sum()
            mu = (R_hist * w[:, None]).sum(axis=0)
            diff = R_hist - mu
            Sigma_hat = (diff * w[:, None]).T @ diff
        else:
            Sigma_hat = np.cov(R_hist.T, ddof=1)
        result[t] = Wi[t] @ Sigma_hat @ Wj[t]

    return pd.Series(result, index=weights_i.index, name="pic")


# ── 3. Return-based estimators (biased; the estimators the paper examines) ─────

def sleeve_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Realized sleeve return: r_sleeve(t) = w(t-1)' r_asset(t) (executed yesterday's weights)."""
    cols = weights.columns
    W = weights.reindex(index=asset_returns.index, columns=cols).fillna(0.0).shift(1)
    R = asset_returns.reindex(index=weights.index, columns=cols).fillna(0.0)
    return (W * R).sum(axis=1)


def rolling_return_cov_series(
    r_i: pd.Series,
    r_j: pd.Series,
    window: int = 756, # 3yr rolling window (industry standard)
    min_obs: int = 63,
) -> pd.Series:
    """Standard rolling-window return-based covariance. Industry standard."""
    df = pd.concat([r_i.rename("i"), r_j.rename("j")], axis=1)
    result = df["i"].rolling(window=window, min_periods=min_obs).cov(df["j"])
    return result.rename("rolling_cov")


def ewma_return_cov_series(
    r_i: pd.Series,
    r_j: pd.Series,
    halflife: int = 63, # ~3 months
    min_obs: int = 30,
) -> pd.Series:
    """EWMA return-based covariance. Better than rolling but still biased."""
    df = pd.concat([r_i.rename("i"), r_j.rename("j")], axis=1)
    result = df["i"].ewm(halflife=halflife, min_periods=min_obs).cov(df["j"])
    return result.rename("ewma_cov")


# ── 4. Factor-regression (FoF approximation) ──────────────────────────────────

def factor_regression_cov_series(
    r_i: pd.Series,
    r_j: pd.Series,
    factors: pd.DataFrame,
    ewma_halflife: int = 126, # 6-month halflife for EWMA weighting
    n_pca: int = 5,
    min_obs: int = 63,
) -> pd.Series:
    """
    Factor-regression implied covariance (Section 5 of paper).

    Steps:
      1. EWMA-weighted PCA on factor returns to get orthogonal scores Z(t).
      2. Regress r_i and r_j on Z(t) with EWMA-weighted OLS to get gamma_i, gamma_j.
      3. Implied cov = gamma_i' gamma_j (orthogonal factors => no extra Sigma needed).

    All estimated causally (only past data used at each t).
    """
    idx = r_i.index.intersection(r_j.index).intersection(factors.index)
    ri = r_i.reindex(idx).fillna(0.0).values
    rj = r_j.reindex(idx).fillna(0.0).values
    F = factors.reindex(idx).fillna(0.0).values
    T = len(idx)
    decay = 0.5 ** (1.0 / ewma_halflife)

    result = np.full(T, np.nan)

    for t in range(min_obs, T):
        ages = np.arange(t - 1, -1, -1, dtype=float)
        ew = decay ** ages
        ew /= ew.sum()

        F_hist = F[:t]
        ri_hist = ri[:t]
        rj_hist = rj[:t]

        # --- Step 1: EWMA-weighted PCA on factors ---
        mu_f = (F_hist * ew[:, None]).sum(axis=0)
        dF = F_hist - mu_f
        Omega = (dF * ew[:, None]).T @ dF
        # Clip n_pca to available factors
        k = min(n_pca, F_hist.shape[1])
        vals, vecs = np.linalg.eigh(Omega)
        top_k = np.argsort(vals)[::-1][:k]
        # Normalize so each PC score has unit variance: Z = dF @ P / sqrt(eigenvalue)
        # This ensures gamma_i'gamma_j is in the same units as cov(r_i, r_j)
        scales = np.sqrt(np.maximum(vals[top_k], 1e-12))
        P = vecs[:, top_k] / scales[None, :] # N_factors x k (scaled eigenvectors)
        Z_hist = dF @ P # t x k (unit-variance orthogonal scores)

        # --- Step 2: EWMA-weighted OLS in PC space ---
        sqrt_w = np.sqrt(ew)
        Z_w = Z_hist * sqrt_w[:, None]
        ri_w = ri_hist * sqrt_w
        rj_w = rj_hist * sqrt_w
        # OLS: beta = (Z_w'Z_w)^-1 Z_w' r_w
        ZtZ = Z_w.T @ Z_w
        try:
            gamma_i = np.linalg.solve(ZtZ, Z_w.T @ ri_w)
            gamma_j = np.linalg.solve(ZtZ, Z_w.T @ rj_w)
        except np.linalg.LinAlgError:
            gamma_i = np.linalg.lstsq(ZtZ, Z_w.T @ ri_w, rcond=None)[0]
            gamma_j = np.linalg.lstsq(ZtZ, Z_w.T @ rj_w, rcond=None)[0]

        # --- Step 3: Implied cov = gamma_i' gamma_j (orthogonal => identity cov) ---
        result[t] = float(gamma_i @ gamma_j)

    return pd.Series(result, index=idx, name="factor_reg_cov")
