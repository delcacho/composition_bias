"""etf_factor_benchmark.py -- a return-based factor benchmark for the young-manager comparison.

The return stream of a young fund is estimated on its own short record; the holdings estimator projects
today's book through an asset covariance estimated on the assets' full window. This module gives the
return-based route the same window: statistical factors are extracted from the priced names of the
estimator support over the trailing window (the covariance the holdings replay already uses), each fund's
loadings are estimated on the K factor returns from whatever record the fund's clone has inside the
window, the factor covariance from the factors' full window, and the pairwise correlation is

    rho^F_ij = b_i' S_f b_j / sqrt((b_i' S_f b_i + d_i) (b_j' S_f b_j + d_j)).

Everything is uncentered and EWMA-weighted like the stream (return_availability.masked_cov conventions).
A fund needs max(MIN_OBS, OBS_PER_FACTOR * K) days with a clone return and defined factors, else its
row and column are NaN. The design was fixed before the run.
"""
import numpy as np
import return_availability as ra

KS = (5, 10)                      # statistical factors: K = 5 and K = 10 (fixed before the run)
MIN_FACTOR_NAMES = 50             # a day's factor returns need this many observed support names

# Named proxy factors, ETFs the panel prices (returns-matrix CUSIP8 columns, present every trading day from 2017-01-03),
# fixed before any outcome was read. Core = the first 13 (style, international
# equity, rates, credit, commodities, dollar); extended = core plus the nine sector SPDRs.
PROXIES = [("IWM", "46428765"), ("IWD", "46428759"), ("IWF", "46428761"), ("MTUM", "46432F39"),
           ("EFA", "46428746"), ("EEM", "46428723"), ("IEF", "46428744"), ("TLT", "46428743"),
           ("LQD", "46428724"), ("HYG", "46428851"), ("GLD", "78463V10"), ("COMT", "46431W85"), ("UUP", "46141D20"),
           ("XLK", "81369Y80"), ("XLF", "81369Y60"), ("XLE", "81369Y50"), ("XLV", "81369Y20"), ("XLI", "81369Y70"),
           ("XLP", "81369Y30"), ("XLU", "81369Y88"), ("XLY", "81369Y40"), ("XLB", "81369Y10")]
PROXY_SETS = (13, 22)             # core, extended: the first P columns of PROXIES


def proxy_columns(names):
    """column positions of the PROXIES in the returns matrix, in PROXIES order; a missing proxy raises."""
    ix = {c: k for k, c in enumerate(names)}
    missing = [t for t, c in PROXIES if c not in ix]
    if missing:
        raise KeyError(f"proxy factors not in the returns matrix: {missing}")
    return np.array([ix[c] for _, c in PROXIES])


def proxy_correlations(Xp, Ap, rw, hw, wts, sets=PROXY_SETS):
    """{P: rho n x n} correlations from the named proxy factors. Xp/Ap: the proxies' window returns (residuals on the
    market factor in the neutral geometry) and availability, T x len(PROXIES) in PROXIES order; rw/hw/wts as in
    factor_correlations. A day with any of the first P proxies missing is undefined for that set."""
    F = np.where(np.asarray(Ap, bool), np.asarray(Xp, np.float64), np.nan)
    out = {}
    for P in sets:
        Fk = F[:, :P]; fdef = np.isfinite(Fk).all(axis=1)
        w = np.asarray(wts, np.float64) * fdef
        Fz = np.where(fdef[:, None], Fk, 0.0)
        S_f = (Fz * w[:, None]).T @ Fz / max(w.sum(), 1e-12)
        B, d, ok = fund_loadings(F, wts, rw, hw, P)
        cov = B @ S_f @ B.T
        var = np.diag(cov) + d
        den = np.sqrt(np.outer(var, var))
        rho = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
        rho[~ok, :] = np.nan; rho[:, ~ok] = np.nan
        out[P] = rho
    return out


def factor_eigenvectors(Sig, priced, kmax):
    """top-kmax eigenvectors of the priced block of the support covariance Sig (dense, symmetrized in float64).
    Returns (V |priced| x kmax, indices of the priced names in Sig's order)."""
    p = np.flatnonzero(priced)
    S = np.asarray(Sig, np.float64)[np.ix_(p, p)]
    S = 0.5 * (S + S.T)
    n = len(S)
    if n < 2:
        return np.zeros((n, 0)), p
    k = min(kmax, n - 2) if n > 3 else 1
    if k >= n - 1:
        lam, V = np.linalg.eigh(S); V = V[:, ::-1][:, :k]
    else:
        from scipy.sparse.linalg import eigsh
        lam, V = eigsh(S, k=k, which="LA", v0=np.ones(n))
        V = V[:, np.argsort(lam)[::-1]]
    return V, p


def fund_loadings(F, wts, rw, hw, K):
    """weighted OLS of each clone column of rw (T x n, zero where absent) on the first K columns of F (T x kmax factor
    returns, NaN rows undefined), over the days the clone exists and the factors are defined, EWMA weights wts.
    Returns (B n x K, d n, ok n): loadings, specific variance with the K-parameter degrees-of-freedom correction, and the
    mask of funds with at least max(MIN_OBS, OBS_PER_FACTOR K) usable days."""
    Fk = F[:, :K]; fdef = np.isfinite(Fk).all(axis=1)
    Fk = np.where(fdef[:, None], Fk, 0.0).astype(np.float64)
    w = np.asarray(wts, np.float64) * fdef
    H = (np.asarray(hw, bool) & fdef[:, None])                          # T x n usable days
    Wn = H * w[:, None]                                                  # T x n weights
    n = rw.shape[1]
    R = np.where(H, np.asarray(rw, np.float64), 0.0)
    G = np.einsum("tk,tl,ti->ikl", Fk, Fk, Wn)                            # n x K x K
    r = np.einsum("tk,ti->ik", Fk, Wn * R)                               # n x K
    ss = np.einsum("ti,ti->i", Wn, R ** 2)                                # weighted sum of squares
    nobs = H.sum(axis=0); wsum = Wn.sum(axis=0)
    ok = nobs >= max(ra.MIN_OBS, ra.OBS_PER_FACTOR * K)
    B = np.zeros((n, K)); d = np.full(n, np.nan)
    if ok.any():
        b, solved = ra._solve_normal(G[ok], r[ok])
        ix = np.flatnonzero(ok)[solved]
        B[ix] = b[solved]
        rss = np.clip(ss[ix] - np.einsum("ik,ik->i", b[solved], r[ix]), 0.0, None)
        d[ix] = rss / np.where(wsum[ix] > 0, wsum[ix], np.nan) * nobs[ix] / np.maximum(nobs[ix] - K, 1)
        ok = np.zeros(n, bool); ok[ix] = True
    return B, d, ok


def factor_correlations(Xres, Aw, Sig, priced, rw, hw, wts, Ks=KS, kmax=None):
    """{K: rho n x n} factor-model correlations of the clones. Xres/Aw: the support names' window returns (residuals in
    the neutral geometry) and availability; Sig/priced: their pairwise-masked covariance and priced mask (the holdings
    replay's); rw/hw: the clones' window returns (residuals in the neutral geometry, zero where absent) and availability;
    wts: the EWMA weights of the window. rho is NaN on a fund without enough usable days."""
    kmax = max(Ks) if kmax is None else kmax
    V, p = factor_eigenvectors(Sig, priced, kmax)
    out = {}
    if V.shape[1] == 0:
        return {K: np.full((rw.shape[1], rw.shape[1]), np.nan) for K in Ks}
    F = ra.factor_returns(np.asarray(Xres)[:, p], np.asarray(Aw, bool)[:, p], V, max(3 * V.shape[1], MIN_FACTOR_NAMES))
    for K in Ks:
        if K > V.shape[1]:
            out[K] = np.full((rw.shape[1], rw.shape[1]), np.nan); continue
        Fk = F[:, :K]; fdef = np.isfinite(Fk).all(axis=1)
        w = np.asarray(wts, np.float64) * fdef
        Fz = np.where(fdef[:, None], Fk, 0.0)
        S_f = (Fz * w[:, None]).T @ Fz / max(w.sum(), 1e-12)             # uncentered EWMA factor covariance
        B, d, ok = fund_loadings(F, wts, rw, hw, K)
        cov = B @ S_f @ B.T
        var = np.diag(cov) + d
        den = np.sqrt(np.outer(var, var))
        rho = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
        rho[~ok, :] = np.nan; rho[:, ~ok] = np.nan
        out[K] = rho
    return out
