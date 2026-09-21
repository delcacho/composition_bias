"""
synthetic.py -- controlled data generation for paper tests.

All scenarios are fully reproducible and have known ground-truth covariance,
so estimator bias can be measured precisely.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _crowding_weight(t: int, crowd_start: int, crowd_end: int, steepness: float = 8.0) -> float:
    """
    Smooth sigmoid transition from 0 (pre-crowd) to 1 (full crowd).
    Returns value in [0, 1].
    """
    mid = (crowd_start + crowd_end) / 2.0
    scale = (crowd_end - crowd_start) / 2.0
    x = steepness * (t - mid) / scale
    return float(_sigmoid(x))


@dataclass
class CrowdingScenario:
    """
    Two-sleeve controlled crowding scenario.

    Asset universe is split into two blocks (A-block, B-block).
    Initially sleeve 1 holds only A-block, sleeve 2 holds only B-block.
    During the crowding window, both sleeves gradually migrate to A-block.
    True covariance goes from near 0 to high over the crowding period.
    """
    T: int = 1500 # total trading days
    N: int = 20 # underlying assets (10 per block)
    crowd_start: int = 600 # day crowding begins
    crowd_end: int = 900 # day crowding is complete
    disperse_start: int = 1050 # optional: crowding reverses after this
    disperse_end: int = 1250
    within_block_corr: float = 0.40 # correlation inside each block
    cross_block_corr: float = 0.05 # correlation between blocks
    annual_vol: float = 0.20 # per-asset annual vol
    crowd_n_assets: int = 3 # how many assets to crowd into (< half = more concentrated)
    seed: int = 42

    # filled by build()
    dates: pd.DatetimeIndex = field(init=False, repr=False)
    Sigma: np.ndarray = field(init=False, repr=False)
    asset_returns: pd.DataFrame = field(init=False, repr=False)
    weights_1: pd.DataFrame = field(init=False, repr=False) # sleeve 1
    weights_2: pd.DataFrame = field(init=False, repr=False) # sleeve 2
    crowd_intensity: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        self.build()

    def build(self):
        rng = np.random.default_rng(self.seed)
        N = self.N
        half = N // 2

        # --- Asset covariance matrix (block structure) ---
        daily_vol = self.annual_vol / np.sqrt(252)
        Sigma = np.full((N, N), self.cross_block_corr * daily_vol**2)
        # within-block A
        Sigma[:half, :half] = self.within_block_corr * daily_vol**2
        # within-block B
        Sigma[half:, half:] = self.within_block_corr * daily_vol**2
        np.fill_diagonal(Sigma, daily_vol**2)
        # ensure PSD
        Sigma = 0.5 * (Sigma + Sigma.T)
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 0:
            Sigma += (-min_eig + 1e-8) * np.eye(N)
        self.Sigma = Sigma

        # --- Asset returns ---
        self.dates = pd.bdate_range("2000-01-01", periods=self.T)
        L = np.linalg.cholesky(Sigma)
        raw = rng.standard_normal((self.T, N)) @ L.T
        cols = [f"A{i+1}" for i in range(half)] + [f"B{i+1}" for i in range(half)]
        self.asset_returns = pd.DataFrame(raw, index=self.dates, columns=cols)

        # --- Sleeve weights ---
        # Base compositions: sleeve 1 = uniform over A-block, sleeve 2 = uniform over B-block
        w1_base = np.zeros(N); w1_base[:half] = 1.0 / half
        w2_base = np.zeros(N); w2_base[half:] = 1.0 / half
        # Crowded composition: both concentrate into the top `crowd_n_assets` of A-block
        n_crowd = max(1, min(self.crowd_n_assets, half))
        w1_crowd = np.zeros(N); w1_crowd[:n_crowd] = 1.0 / n_crowd
        w2_crowd = np.zeros(N); w2_crowd[:n_crowd] = 1.0 / n_crowd

        intensity = np.zeros(self.T)
        W1 = np.zeros((self.T, N))
        W2 = np.zeros((self.T, N))

        for t in range(self.T):
            # crowding in
            c_in = _crowding_weight(t, self.crowd_start, self.crowd_end)
            # crowding out (if disperse window defined)
            if self.disperse_start < self.T:
                c_out = _crowding_weight(t, self.disperse_start, self.disperse_end)
            else:
                c_out = 0.0
            lam = np.clip(c_in - c_out, 0.0, 1.0)
            intensity[t] = lam
            W1[t] = (1 - lam) * w1_base + lam * w1_crowd
            W2[t] = (1 - lam) * w2_base + lam * w2_crowd

        self.crowd_intensity = pd.Series(intensity, index=self.dates, name="crowd_intensity")
        self.weights_1 = pd.DataFrame(W1, index=self.dates, columns=cols)
        self.weights_2 = pd.DataFrame(W2, index=self.dates, columns=cols)

    def cosine_similarity(self) -> pd.Series:
        """Cosine similarity between sleeve weight vectors at each date."""
        W1 = self.weights_1.values
        W2 = self.weights_2.values
        dot = (W1 * W2).sum(axis=1)
        n1 = np.linalg.norm(W1, axis=1)
        n2 = np.linalg.norm(W2, axis=1)
        with np.errstate(invalid="ignore"):
            cos = np.where((n1 > 0) & (n2 > 0), dot / (n1 * n2), np.nan)
        return pd.Series(cos, index=self.dates, name="cosine_sim")

    def sleeve_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Realized sleeve returns (using lagged weights -- no lookahead)."""
        R = self.asset_returns
        r1 = (self.weights_1.shift(1).fillna(0.0) * R).sum(axis=1)
        r2 = (self.weights_2.shift(1).fillna(0.0) * R).sum(axis=1)
        r1.name = "sleeve_1"
        r2.name = "sleeve_2"
        return r1, r2


@dataclass
class FoFScenario:
    """
    Fund-of-funds scenario where manager returns are generated from actual
    asset holdings, so PIC (using holdings + asset covariance) is the
    exact ground truth.

    Model:
        r_asset(t) = B @ f(t) + eps(t) B: N×F loading matrix
        r_mgr_i(t) = w_i(t-1)' r_asset(t) manager return from holdings

    Observable factors f(t) are independent draws (not noisy versions of
    the true latent factors) -- the factor regression must learn the link
    from r_mgr_i to f(t) purely from the return history.

    During crowding, all managers shift their weights toward the first
    n_crowd_assets assets, driving up their true covariance.

    Ground truth: w_i(t)' Sigma_asset w_j(t) (PIC, computed analytically
    from current holdings and the full asset covariance).
    """
    T: int = 1500
    N_assets: int = 30
    K_managers: int = 5
    n_factors: int = 8 # observable macro factors
    n_crowd_assets: int = 5 # how many assets all managers crowd into
    crowd_start: int = 700
    crowd_end: int = 950
    assets_per_manager: int = 8 # assets each manager holds pre-crowding
    factor_asset_corr: float = 0.50 # correlation between each factor and its linked asset block
    idio_vol_asset: float = 0.008 # daily idio vol per asset (on top of factor vol)
    annual_vol: float = 0.18
    seed: int = 99
    smoothing_alpha: float = 0.0 # AR(1) NAV smoothing: 0=none, 0.25=typical hedge fund

    dates: pd.DatetimeIndex = field(init=False, repr=False)
    factor_returns: pd.DataFrame = field(init=False, repr=False)
    asset_returns: pd.DataFrame = field(init=False, repr=False)
    manager_returns: pd.DataFrame = field(init=False, repr=False)
    manager_weights: list = field(init=False, repr=False)
    Sigma_asset: np.ndarray = field(init=False, repr=False)
    crowd_intensity: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        self.build()

    def build(self):
        rng = np.random.default_rng(self.seed)
        T, N, K, F = self.T, self.N_assets, self.K_managers, self.n_factors

        self.dates = pd.bdate_range("2000-01-01", periods=T)
        daily_vol = self.annual_vol / np.sqrt(252)

        # --- Observable macro factors (F independent series) ---
        f_vol = daily_vol * 0.8
        factor_rets = rng.standard_normal((T, F)) * f_vol
        self.factor_returns = pd.DataFrame(
            factor_rets, index=self.dates,
            columns=[f"F{i+1}" for i in range(F)]
        )

        # --- Asset factor loading matrix B (N×F) ---
        # Each asset loads primarily on one factor (block structure) with noise
        assets_per_factor = max(1, N // F)
        B = np.zeros((N, F))
        for fi in range(F):
            start = fi * assets_per_factor
            end = min(start + assets_per_factor, N)
            B[start:end, fi] = self.factor_asset_corr
        # small cross-loadings
        B += rng.uniform(0.0, 0.05, size=(N, F))

        # --- Asset returns: r_asset = B @ f + eps ---
        eps_vol = self.idio_vol_asset
        eps = rng.standard_normal((T, N)) * eps_vol
        raw_assets = factor_rets @ B.T + eps
        asset_cols = [f"Asset{i+1}" for i in range(N)]
        self.asset_returns = pd.DataFrame(raw_assets, index=self.dates, columns=asset_cols)
        self.Sigma_asset = np.cov(raw_assets.T, ddof=1)

        # --- Manager weights (time-varying) ---
        # Pre-crowding: each manager i holds assets in its own niche
        # Crowding: all managers shift to the first n_crowd_assets
        n_crowd = min(self.n_crowd_assets, N)
        base_weights = []
        for k in range(K):
            w = np.zeros(N)
            # stagger the niche so managers hold different assets initially
            niche_start = (k * self.assets_per_manager) % (N - n_crowd) + n_crowd
            niche_end = min(niche_start + self.assets_per_manager, N)
            if niche_end <= niche_start:
                niche_end = niche_start + 1
            w[niche_start:niche_end] = 1.0 / (niche_end - niche_start)
            base_weights.append(w)

        crowd_w = np.zeros(N)
        crowd_w[:n_crowd] = 1.0 / n_crowd

        crowd_vals = np.zeros(T)
        W_all = [np.zeros((T, N)) for _ in range(K)]

        for t in range(T):
            lam = _crowding_weight(t, self.crowd_start, self.crowd_end)
            lam = min(lam, 1.0)
            crowd_vals[t] = lam
            for k in range(K):
                W_all[k][t] = (1 - lam) * base_weights[k] + lam * crowd_w

        self.crowd_intensity = pd.Series(crowd_vals, index=self.dates, name="crowd_intensity")
        self.manager_weights = [
            pd.DataFrame(W_all[k], index=self.dates, columns=asset_cols)
            for k in range(K)
        ]

        # --- Manager returns: r_mgr_i(t) = w_i(t-1)' r_asset(t) ---
        mgr_rets = np.zeros((T, K))
        for k in range(K):
            W_lag = W_all[k].copy()
            W_lag[1:] = W_all[k][:-1]
            W_lag[0] = base_weights[k]
            mgr_rets[:, k] = (W_lag * raw_assets).sum(axis=1)

        # Optional AR(1) NAV smoothing (Getmansky-Lo-Makarov stale-price model).
        # r_obs(t) = (1 - alpha) * r_true(t) + alpha * r_obs(t-1)
        # Induces autocorrelation that biases standard rolling-window correlations.
        if self.smoothing_alpha > 0:
            alpha = self.smoothing_alpha
            for k in range(K):
                for t in range(1, T):
                    mgr_rets[t, k] = (1 - alpha) * mgr_rets[t, k] + alpha * mgr_rets[t - 1, k]

        self.manager_returns = pd.DataFrame(
            mgr_rets, index=self.dates,
            columns=[f"Mgr{i+1}" for i in range(K)]
        )
