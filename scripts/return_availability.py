"""
return_availability.py -- missing asset returns handled as missing, not as zeros.

The returns matrix has gaps: names before they list or enter the data, names Compustat or the FMP splice cannot price
after May 2025, the odd missing day. Filling them with zero makes a name with little real history look nearly riskless:
w' Sigma w understates the risk of any book holding it, and a book rotating into it looks like no rotation at all.

Rules, applied identically in persistence (etf_race.persistence), the covariance comparison (etf_race.run_race), the
staleness experiment (etf_staleness_experiment.run_geometry) and the sizing comparison (etf_sizing_race.fund_scores):
  coverage c_i share of the estimation window's weight (EWMA, or equal weights for a fixed covariance) on days where
                 name i has a return
  unpriced c_i < MIN_COVER, or fewer than MIN_OBS actual observations: the name is dropped from every book on that
                 date and each leg of the book is rescaled to its own gross (renorm_books, leg_factors); under
                 MIN_PRICED_SHARE of the gross priced the book is undefined, the rule the clone day uses (masked_book_returns)
  covariance the exact pairwise-masked estimate, sum of weighted products over days both names have returns divided by
                 the sum of the weights of those days: masked_cov on a window (comparison, experiment, persistence),
                 MaskedEWMA as a recursion (sizing comparison). Pairs observed together on fewer than MIN_OBS days get zero
                 covariance. coverage_scaled (zero fill x 1/sqrt(c_i)) is exact on variances but shrinks the covariance of
                 a partly covered name with a fully covered one by sqrt(c_i); the pairwise-masked estimate does not.
  neutral a factor is removed on observed days only (neutral_residuals): betas fitted over days the name has a return,
                 residual zero where it has none (a zero-filled 'x - f beta' invents -f beta on every missing day). The
                 book replay is hedged with the clone's beta, as the stream and the target are (etf_race.holdings_covs).
  MaskedEWMA a pair never observed together has an unknown (NaN) covariance, not a zero one.
  factor model persistence over the whole held universe (~28k names, T ~2,500 days: an N x N estimate is rank <= T and,
                 pairwise-masked, not PSD): Sigma = B Sigma_f B' + D (fit_statistical_factor_model), loadings from the
                 masked covariance of a weight-covered subset, factor returns by daily cross-sectional OLS on observed
                 names, betas and D by per-name time-series OLS on observed days. Uncentered, as masked_cov. PSD.
"""
import numpy as np

MIN_OBS = 60 # a name is priced once it has three months of actual observations
MIN_COVER = 0.10 # and at least 10% of the window's weight: the estimates are normalized by the observed weight, so a high
                     # floor would only drop recently listed names from the holdings book while the clone already holds them
                     # (a mismatch between the two sides)


def availability(Rs):
    """T x N boolean: the name has a return that day."""
    return np.isfinite(Rs.values) if hasattr(Rs, "values") else np.isfinite(Rs)


def coverage(A_window, wts=None):
    """per-name coverage over a window: weighted share of days with a return (equal weights when wts is None)."""
    A = A_window.astype(np.float32)
    if wts is None:
        return A.mean(axis=0)
    w = np.asarray(wts, dtype=np.float32)
    return (w / w.sum()) @ A


def priced_mask(A_window, wts=None, min_cover=MIN_COVER, min_obs=MIN_OBS):
    """names priced on this window: coverage at least min_cover and at least min_obs actual observations."""
    return (coverage(A_window, wts) >= min_cover) & (A_window.sum(axis=0) >= min_obs)


def coverage_scaled(X_window, A_window, wts=None, min_cover=MIN_COVER, min_obs=MIN_OBS):
    """(zero-filled returns scaled by 1/sqrt(coverage) for priced names, zero columns for unpriced ones; priced mask)."""
    c = coverage(A_window, wts)
    keep = (c >= min_cover) & (A_window.sum(axis=0) >= min_obs)
    scale = np.where(keep, 1.0 / np.sqrt(np.clip(c, 1e-12, None)), 0.0).astype(np.float32)
    X0 = masked_zero(X_window, A_window) # the mask rules, not the values: a residual is nonzero on gaps
    return X0 * scale[None, :], keep


def masked_zero(X_window, A_window):
    """float32 returns with exact zeros wherever the availability mask says there is no return."""
    return np.where(np.asarray(A_window, bool), np.nan_to_num(np.asarray(X_window, dtype=np.float32)), np.float32(0.0))


def masked_cov(X_window, A_window, wts=None, min_overlap=MIN_OBS):
    """pairwise-masked weighted (uncentered) covariance of the window's columns: Sigma_ij = sum_s w_s x_is x_js 1{both} /
    sum_s w_s 1{both}; pairs with fewer than min_overlap common observations -> 0. float32, columns x columns. Not
    guaranteed PSD: callers treat a non-positive book variance as undefined."""
    A = np.asarray(A_window, bool); Af = A.astype(np.float32)
    X0 = masked_zero(X_window, A)
    w = np.full(len(A), 1.0 / max(len(A), 1), np.float32) if wts is None else np.asarray(wts, np.float32) / np.float32(np.sum(wts))
    num = (X0 * w[:, None]).T @ X0
    if wts is None: # equal weights: the weight of the common days is their count / T
        cnt = Af.T @ Af; den = cnt * w[0]
    else:
        den = (Af * w[:, None]).T @ Af; cnt = Af.T @ Af if min_overlap > 0 else None
    del X0
    ok = den > 0
    if min_overlap > 0:
        ok &= cnt >= min_overlap
    del cnt
    np.divide(num, den, out=num, where=ok)
    num[~ok] = 0.0
    return num


MIN_FACTOR_NAMES = 50 # a day's factor returns need max(3K, 50) observed subset names, else the day is undefined (NaN)
MAX_SOLVE_COND = 1e10 # a K x K normal matrix worse conditioned than this is not solved: the day / the name is left out
FACTOR_PRICED, SPECIFIC_ONLY, UNPRICED = "factor", "specific_only", "unpriced"


def masked_top_eigenpairs(X_window, A_window, k):
    """top-k eigenpairs of the pairwise-masked covariance (masked_cov) of the window's columns, symmetrized in float64:
    (eigenvalues descending, eigenvectors n x k, trace). Dense n x n: callers bound n (etf_race.check_dense_budget)."""
    Sig = masked_cov(X_window, A_window)
    S = Sig.astype(np.float64); S += Sig.T; S *= 0.5 # masked_cov is symmetric only up to float32 rounding
    del Sig
    n = len(S); tr = float(np.trace(S))
    if k >= n - 1: # eigsh needs k < n - 1
        lam, V = np.linalg.eigh(S); lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    else:
        from scipy.sparse.linalg import eigsh
        lam, V = eigsh(S, k=k, which="LA", v0=np.ones(n)) # fixed start vector: the same run gives the same vectors
        o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    return lam, V, tr


def eigenvalue_ratio_k(lam, kmax):
    """Ahn and Horenstein (2013) eigenvalue ratio: K = argmax over k = 1..kmax of lam_k / lam_{k+1} (lam descending, at least
    kmax + 1 of them). A ratio whose denominator is not positive is undefined (NaN; the masked estimate is not PSD).
    -> (K, ratios for k = 1..kmax)."""
    lam = np.asarray(lam, np.float64)
    if len(lam) < kmax + 1:
        raise ValueError(f"eigenvalue ratio up to k={kmax} needs {kmax + 1} eigenvalues, got {len(lam)}")
    num, den = lam[:kmax], lam[1:kmax + 1]
    ratios = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    if not np.isfinite(ratios).any():
        raise ValueError("no positive eigenvalue ratio: the subset covariance has fewer than two positive eigenvalues")
    return int(np.nanargmax(ratios)) + 1, ratios


def _solve_normal(G, r):
    """batched OLS solve G b = r (G: m x K x K, r: m x K); rows with cond(G) over MAX_SOLVE_COND -> (NaN, ok False)."""
    out = np.full(r.shape, np.nan); ok = np.zeros(len(G), bool)
    if len(G):
        ok = np.linalg.cond(G) < MAX_SOLVE_COND
        if ok.any():
            out[ok] = np.linalg.solve(G[ok], r[ok][..., None])[..., 0]
    return out, ok


def factor_returns(X_sub, A_sub, V, min_names, chunk=256):
    """T x K daily cross-sectional OLS of the observed subset names' returns on their loadings (rows of V): f_t =
    (V_o' V_o)^-1 V_o' x_o over the names with a return that day. NaN on a day with fewer than min_names observed or an
    ill-conditioned V_o' V_o. float64; memory chunk x n."""
    T = len(X_sub); n, K = V.shape
    VV = (V[:, :, None] * V[:, None, :]).reshape(n, K * K)
    f = np.full((T, K), np.nan)
    for s in range(0, T, chunk):
        Ac = np.asarray(A_sub[s:s + chunk], bool)
        Xc = masked_zero(X_sub[s:s + chunk], Ac).astype(np.float64)
        G = (Ac.astype(np.float64) @ VV).reshape(-1, K, K); r = Xc @ V
        enough = Ac.sum(axis=1) >= min_names
        b, ok = _solve_normal(G[enough], r[enough])
        rows = np.flatnonzero(enough)[ok]
        f[s + rows] = b[ok]
    return f


OBS_PER_FACTOR = 10 # joint days per factor a name needs for betas; below it the name is specific-risk only


def name_factor_regressions(X, A, f, min_obs=MIN_OBS, block=2000):
    """per column: no-intercept OLS of its returns on the factor returns over the days both are observed. Uncentered
    throughout, as masked_cov, except that d_i = sum of squared residuals / (n_i - K), the regression's degrees of freedom
      FACTOR_PRICED >= max(min_obs, 10 K) joint days (and a well-conditioned F'F): beta, d from the regression; ten
      SPECIFIC_ONLY fewer joint days but >= min_obs own returns: beta 0, d = its own uncentered variance
      UNPRICED fewer than min_obs own returns: beta 0, d 0 (masked_cov's zero for a pair under min_overlap)
    -> (B N x K, d N, status N, R2 N uncentered, NaN unless FACTOR_PRICED). float64; memory T x block."""
    T, N = X.shape; K = f.shape[1]
    fdef = np.isfinite(f).all(axis=1); F = np.where(fdef[:, None], f, 0.0)
    FF = (F[:, :, None] * F[:, None, :]).reshape(T, K * K)
    B = np.zeros((N, K)); d = np.zeros(N); r2 = np.full(N, np.nan); status = np.full(N, UNPRICED, dtype=object)
    for s in range(0, N, block):
        e = min(s + block, N)
        Ab = np.asarray(A[:, s:e], bool); Xb = masked_zero(X[:, s:e], Ab).astype(np.float64)
        n_own = Ab.sum(axis=0); ss_own = np.einsum("ti,ti->i", Xb, Xb)
        Xb[~fdef] = 0.0; Aj = Ab & fdef[:, None]
        n = Aj.sum(axis=0); ss = np.einsum("ti,ti->i", Xb, Xb)
        G = (FF.T @ Aj.astype(np.float64)).T.reshape(-1, K, K); r = Xb.T @ F
        del Xb, Aj
        cand = n >= max(min_obs, OBS_PER_FACTOR * K)
        b, ok = _solve_normal(G[cand], r[cand])
        fit = np.flatnonzero(cand)[ok]; b = b[ok]
        rss = np.clip(ss[fit] - np.einsum("ik,ik->i", b, r[fit]), 0.0, None) # b'G b = b'r at the solution
        B[s + fit] = b; d[s + fit] = rss / (n[fit] - K); status[s + fit] = FACTOR_PRICED # residual degrees of freedom
        r2[s + fit] = np.where(ss[fit] > 0, 1.0 - rss / np.where(ss[fit] > 0, ss[fit], 1.0), np.nan)
        spec = np.setdiff1d(np.flatnonzero(n_own >= min_obs), fit)
        d[s + spec] = ss_own[spec] / n_own[spec]; status[s + spec] = SPECIFIC_ONLY
    return B, d, status, r2


class StatisticalFactorModel:
    """Sigma = B Sigma_f B' + diag(d) on the columns it was fitted on, never formed. Factor 0 is the top eigenvector (the
    market); the neutral geometry drops its row/column of Sigma_f and its column of B, D unchanged."""

    def __init__(self, B, f, Sigma_f, d, status, r2, lam, trace, subset):
        self.B, self.f, self.Sigma_f, self.d, self.status, self.r2 = B, f, Sigma_f, d, status, r2
        self.lam, self.trace, self.subset = lam, trace, subset
        self.K = B.shape[1]


def fit_statistical_factor_model(X, A, subset_idx, K, min_obs=MIN_OBS, min_names=None, eig=None):
    """statistical factor model of the columns of X (T x N returns, A availability): loadings of the subset columns
    subset_idx = top K eigenvectors of their pairwise-masked covariance; daily factor returns by cross-sectional OLS on the
    observed subset names (factor_returns); every column's beta and specific variance by time-series OLS on them
    (name_factor_regressions); Sigma_f = uncentered covariance of f over the days it is defined. eig: (lam, V, trace) from
    masked_top_eigenpairs on the same subset with at least K vectors, to share one eigendecomposition across K."""
    subset_idx = np.asarray(subset_idx, np.int64)
    if eig is None:
        eig = masked_top_eigenpairs(X[:, subset_idx], A[:, subset_idx], K)
    lam, V, tr = eig
    if V.shape[1] < K:
        raise ValueError(f"{K} factors requested, {V.shape[1]} eigenvectors given")
    V = V[:, :K]
    f = factor_returns(X[:, subset_idx], A[:, subset_idx], V, max(3 * K, MIN_FACTOR_NAMES) if min_names is None else min_names)
    fdef = np.isfinite(f).all(axis=1)
    if fdef.sum() < min_obs:
        raise ValueError(f"factor returns defined on {int(fdef.sum())} days, fewer than {min_obs}")
    Sigma_f = f[fdef].T @ f[fdef] / fdef.sum()
    B, d, status, r2 = name_factor_regressions(X, A, f, min_obs)
    return StatisticalFactorModel(B, f, Sigma_f, d, status, r2, np.asarray(lam[:K], np.float64), tr, subset_idx)


def neutral_residuals(X_window, A_window, f, wts):
    """residuals of each column on the factor f, betas fitted over the days the column has a return (weights wts), and
    exactly zero where it has none. Returns (residuals float32, betas)."""
    A = np.asarray(A_window, bool); X0 = masked_zero(X_window, A)
    f = np.asarray(f, np.float32); wf = np.asarray(wts, np.float32) * f
    den = (wf * f) @ A.astype(np.float32)
    beta = np.where(den > 0, (wf @ X0) / np.where(den > 0, den, 1.0), 0.0).astype(np.float32)
    return np.where(A, X0 - np.outer(f, beta), np.float32(0.0)), beta


MIN_PRICED_SHARE = 0.95 # a clone day needs 95% of the book's gross priced (the admission rule's threshold)


def leg_factors(W, Wk, min_share):
    """per-row rescaling of the priced part Wk of books W: each leg (long, short) back to its own gross, so dropping an
    unpriced short does not lever the net ([.5, .5, -.3] rescaled by gross read net 1.3 for 0.7); a leg with nothing priced
    cannot be rescaled and its gross goes to the other leg. (long factor, short factor, valid row); valid = something priced
    and at least min_share of the gross priced."""
    pos0 = np.clip(W, 0, None).sum(axis=1); neg0 = np.clip(-W, 0, None).sum(axis=1)
    posk = np.clip(Wk, 0, None).sum(axis=1); negk = np.clip(-Wk, 0, None).sum(axis=1)
    gross = pos0 + neg0; g2 = posk + negk
    with np.errstate(divide="ignore", invalid="ignore"):
        fl = np.where(posk > 0, pos0 / np.where(posk > 0, posk, 1.0), 0.0)
        fs = np.where(negk > 0, neg0 / np.where(negk > 0, negk, 1.0), 0.0)
        kept = pos0 * (posk > 0) + neg0 * (negk > 0)
        tot = np.where(kept > 0, gross / np.where(kept > 0, kept, 1.0), 0.0)
    valid = (g2 > 0) & (g2 >= min_share * gross)
    return fl * tot, fs * tot, valid


def renorm_books(Wm, keep, min_share=0.0):
    """drop unpriced names from each book (rows) and rescale each leg to its original gross (leg_factors); rows with nothing
    priced, or with less than min_share of the gross priced (the clone's rule, MIN_PRICED_SHARE, so a long/short book cannot
    lose a leg and come back as a gross-sized one-sided book), -> NaN."""
    Wm = np.asarray(Wm, dtype=np.float32)
    W0 = np.nan_to_num(Wm)
    Wk = W0 * np.asarray(keep, bool)[None, :]
    fl, fs, valid = leg_factors(W0, Wk, min_share)
    out = np.where(Wk > 0, Wk * fl[:, None], Wk * fs[:, None]).astype(np.float32)
    out[~valid] = np.nan
    out[~np.isfinite(Wm).all(axis=1)] = np.nan # a missing book stays missing
    return out


def masked_book_returns(W, X_next, A_next, min_share=MIN_PRICED_SHARE):
    """book returns over names that have a return that day, each leg rescaled to its own gross (leg_factors, the rule
    renorm_books applies to the holdings book); NaN when the priced share of the gross is under min_share. W: days x names
    (dense), X_next: the matching days' returns (NaN or zero-filled), A_next: their availability."""
    W = np.asarray(W, dtype=np.float64); A = np.asarray(A_next, dtype=bool)
    X0 = np.where(A, np.nan_to_num(np.asarray(X_next, dtype=np.float64)), 0.0)
    Wk = W * A
    fl, fs, valid = leg_factors(W, Wk, min_share)
    num = np.einsum("ij,ij->i", np.where(Wk > 0, Wk * fl[:, None], Wk * fs[:, None]), X0)
    return np.where(valid, num, np.nan)


def masked_ewma_var(x, halflife):
    """EWMA variance of a series with gaps, normalized by the weight of the observed days (the same normalization as
    MaskedEWMA, so the start of the sample and the gaps are treated identically on both sides of a comparison)."""
    lam = 0.5 ** (1.0 / halflife); num = 0.0; den = 0.0; out = np.full(len(x), np.nan)
    for i, v in enumerate(x):
        a = np.isfinite(v)
        num = lam * num + (1 - lam) * (v * v if a else 0.0); den = lam * den + (1 - lam) * (1.0 if a else 0.0)
        out[i] = num / den if den > 1e-12 else np.nan
    return out


class MaskedEWMA:
    """exact pairwise-masked EWMA covariance for a small support: Sigma_ij = sum_s w_s x_i x_j 1{both} / sum_s w_s 1{both}.
    Also tracks coverage (the diagonal of the denominator, normalized by the total weight) and observation counts."""

    def __init__(self, n, halflife):
        self.lam = 0.5 ** (1.0 / halflife)
        self.num = np.zeros((n, n), np.float64); self.den = np.zeros((n, n), np.float64)
        self.tot = 0.0; self.obs = np.zeros(n, np.int64)

    def update(self, x):
        a = np.isfinite(x); x0 = np.where(a, x, 0.0)
        self.num = self.lam * self.num + (1 - self.lam) * np.outer(x0, x0)
        af = a.astype(np.float64)
        self.den = self.lam * self.den + (1 - self.lam) * np.outer(af, af)
        self.tot = self.lam * self.tot + (1 - self.lam); self.obs += a

    def cov(self, ix=None):
        """the covariance (of the names ix only, when given); never observed together = unknown (NaN), not a zero covariance
        that reads as a riskless pair."""
        num, den = (self.num, self.den) if ix is None else (self.num[np.ix_(ix, ix)], self.den[np.ix_(ix, ix)])
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)
        return S

    def priced(self, min_cover=MIN_COVER, min_obs=MIN_OBS):
        cov_share = np.diag(self.den) / max(self.tot, 1e-12)
        return (cov_share >= min_cover) & (self.obs >= min_obs)
