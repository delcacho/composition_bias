"""allocator_pnl_attribution.py -- pathwise (Aumann-Shapley) attribution of the holdings-vs-stream
realized P&L difference to the individual pairwise correlation changes that drove the allocation apart.

Exhibit 7 says the holdings-correlation book (c2) has shallower drawdowns than the return-stream book;
the online appendix says the mechanism is a stale-EWMA correlation, so in a fast correlation spike the
stream is sized on pre-spike correlations and over-concentrated. This turns that mechanism from a story
into an accounting identity: on each rebalance date it attributes the c2-minus-stream realized P&L to
each sleeve pair's correlation change, exactly.

THE CLEAN INTERVENTION. Both books already share the stream volatilities D_R (c2 = D_R . R_holdings . D_R,
stream = D_R . R_stream . D_R), so the ONLY covariance input that differs is the correlation matrix. Move
it along a straight line and hold the vols fixed:

    Sigma(lambda) = D_R [ R_stream + lambda (R_holdings - R_stream) ] D_R,   0 <= lambda <= 1.

At each lambda solve the ERC weights, differentiate the ERC first-order conditions implicitly to get the
exact local sensitivity of the realized payoff to every pair correlation rho_ij (no invalid one-pair-
changed matrix is ever built), and integrate along the path:

    A_ij = (rho_ij^H - rho_ij^R) * integral_0^1 [ d PnL / d rho_ij ](lambda) d lambda,

    sum_{i<j} A_ij  ==  ( w_holdings - w_stream )' r_next       (to numerical precision).

Because leverage differs between the two books (each is scaled to the 10% target off its OWN ex-ante
vol), the attribution is run at a COMMON scale so the reconstructed object is exactly the correlation-
driven ALLOCATION channel. The leftover leverage channel is reported as scale_channel_bp = (full book
c2-minus-stream period P&L) - (attributed allocation difference), the fine-grained pair-level companion
of allocator_tilt_decomp.py's scale/allocation split.

Reads the per-rebalance-date inputs persisted by allocator_multiasset.py
(allocator_attribution_inputs{tag}.pkl). Writes a pair x date table and a per-date reconstruction log.

usage: python research/allocator_pnl_attribution.py [inputs.pkl] [--ngrid=41] [--tail=0.05]
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


# --------------------------------------------------------------------------------------------------
# 1. ERC solver -- long-only risk budgeting  min_x  1/2 x' Sigma x - sum_i b_i log x_i,  w = x/sum(x).
#    Solving on the covariance (not the correlation) returns the UNNORMALISED x the Jacobian needs.
#    For equal budgets its normalised weights equal allocator_multiasset.risk_parity's to ~1e-9.
# --------------------------------------------------------------------------------------------------
def solve_erc_x(cov, budgets=None, x0=None):
    from scipy.optimize import minimize
    cov = np.asarray(cov, float); n = cov.shape[0]
    b = np.ones(n) / n if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    if x0 is None:
        x0 = np.sqrt(b / np.diag(cov))
    x0 = np.maximum(x0, 1e-10)

    def obj(x):
        return 0.5 * x @ cov @ x - np.sum(b * np.log(x))

    def grad(x):
        return cov @ x - b / x

    res = minimize(obj, x0, jac=grad, method="L-BFGS-B", bounds=[(1e-12, None)] * n,
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
    if (not res.success) and np.linalg.norm(grad(res.x), ord=np.inf) > 1e-7:
        raise RuntimeError(res.message)
    x = res.x
    return x, x / x.sum(), b


# --------------------------------------------------------------------------------------------------
# 2. Realized payoff of allocator weights w and its gradient wrt w. future_returns is (n,) one-period
#    or (h,n) daily over the holding interval; scales is a scalar (common return-stream leverage).
# --------------------------------------------------------------------------------------------------
def payoff_and_grad_w(w, future_returns, scales=1.0):
    R = np.asarray(future_returns, float)
    if R.ndim == 1:
        s = float(np.asarray(scales))
        return s * (w @ R), s * R
    if R.ndim != 2:
        raise ValueError("future_returns must have dimension 1 or 2.")
    sc = np.asarray(scales, float)
    if sc.ndim == 0:
        sc = np.full(R.shape[0], float(sc))
    daily = sc * (R @ w)                          # portfolio return each day of the interval
    gross = 1.0 + daily
    wealth = np.prod(gross)
    payoff = wealth - 1.0
    grad_w = wealth * np.sum((sc[:, None] * R) / gross[:, None], axis=0)   # d/dw prod_d[1 + scale_d r_d'w]
    return payoff, grad_w


# --------------------------------------------------------------------------------------------------
# 3. Exact local ERC sensitivity of the realized payoff to every pair correlation at the CURRENT matrix.
#    Differentiates the FOC g(x)=Sigma x - b/x = 0 implicitly: dx/drho = -[Sigma+diag(b/x^2)]^{-1} (dSigma/drho) x,
#    with dSigma/drho_ij = sigma_i sigma_j (E_ij+E_ji) since D is held fixed. One factorisation for all pairs.
# --------------------------------------------------------------------------------------------------
def local_pair_pnl_gradients(cov, x, budgets, vols, grad_w):
    n = len(x)
    H = cov + np.diag(budgets / x ** 2)
    factor = cho_factor(H, lower=True, check_finite=False)
    s = x.sum(); w = x / s
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            v = np.zeros(n); sij = vols[i] * vols[j]
            v[i] = sij * x[j]; v[j] = sij * x[i]        # (dSigma/drho_ij) x
            dx = -cho_solve(factor, v, check_finite=False)
            dw = (dx - w * dx.sum()) / s
            out.append((i, j, float(grad_w @ dw)))
    return out


def local_pair_weight_jacobian(cov, x, budgets, vols):
    """The ERC weight sensitivity dw/drho_ij for every pair, as a (n_pairs, n) array (same implicit
    differentiation as local_pair_pnl_gradients, but returns the weight vectors so a caller can dot them
    into a whole holding period of daily returns -- what the expected-shortfall decomposition needs).
    Returns (pairs, dW) with pairs the list of (i,j) in the same order."""
    n = len(x)
    H = cov + np.diag(budgets / x ** 2)
    factor = cho_factor(H, lower=True, check_finite=False)
    s = x.sum(); w = x / s
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dW = np.empty((len(pairs), n))
    for k, (i, j) in enumerate(pairs):
        v = np.zeros(n); sij = vols[i] * vols[j]
        v[i] = sij * x[j]; v[j] = sij * x[i]
        dx = -cho_solve(factor, v, check_finite=False)
        dW[k] = (dx - w * dx.sum()) / s
    return pairs, dW


# --------------------------------------------------------------------------------------------------
# 4. Pathwise attribution R_stream -> R_holdings at fixed vols.
# --------------------------------------------------------------------------------------------------
def _floor_corr(C, kappa_max):
    """Floor a correlation matrix's eigenvalues at max/kappa_max and renormalise to unit diagonal --
    exactly allocator_multiasset.condition()'s correlation step. The book conditions BOTH the stream and
    c2 covariances before solving ERC, so the attribution must interpolate between the SAME conditioned
    correlations, or its endpoints are not the book's weights and the path runs through the near-singular
    holdings matrix the book never uses (2009-recovery dates: raw residual up to 112 bp, conditioned <1)."""
    d = np.sqrt(np.maximum(np.diag(C), 1e-30)); Rm = C / np.outer(d, d)
    ev, V = np.linalg.eigh(Rm); ev = np.maximum(ev, ev.max() / kappa_max); Rm = V @ np.diag(ev) @ V.T
    dd = np.sqrt(np.diag(Rm)); return Rm / np.outer(dd, dd)


def path_attribution(R_stream, R_holdings, vols_stream, future_returns,
                     budgets=None, scales=1.0, n_grid=41, kappa_max=1e3):
    R0 = np.asarray(R_stream, float); R1 = np.asarray(R_holdings, float)
    if kappa_max is not None:                        # match the book's condition(); identity when flooring does not bind
        R0 = _floor_corr(R0, kappa_max); R1 = _floor_corr(R1, kappa_max)
    vols = np.asarray(vols_stream, float); n = R0.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    lambdas = np.linspace(0.0, 1.0, n_grid)
    gradients = np.empty((n_grid, len(pairs)))
    endpoints = {}
    x0 = None
    for k, lam in enumerate(lambdas):
        R = R0 + lam * (R1 - R0)
        cov = np.outer(vols, vols) * R
        x, w, b = solve_erc_x(cov, budgets=budgets, x0=x0)
        x0 = x
        payoff, grad_w = payoff_and_grad_w(w, future_returns, scales=scales)
        if k == 0:
            endpoints["stream"] = {"weights": w.copy(), "pnl": payoff}
        if k == n_grid - 1:
            endpoints["holdings"] = {"weights": w.copy(), "pnl": payoff}
        local = local_pair_pnl_gradients(cov, x, b, vols, grad_w)
        gradients[k, :] = [d for _, _, d in local]
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz"))   # np.trapz renamed to np.trapezoid in numpy 2
    integrated = _trapz(gradients, lambdas, axis=0)
    delta_rho = np.array([R1[i, j] - R0[i, j] for i, j in pairs])
    attribution = delta_rho * integrated
    direct = endpoints["holdings"]["pnl"] - endpoints["stream"]["pnl"]
    table = pd.DataFrame({"i": [i for i, _ in pairs], "j": [j for _, j in pairs],
                          "delta_rho": delta_rho, "integrated_dPnl_dRho": integrated,
                          "pnl_attribution": attribution})
    return table, direct, float(attribution.sum()), endpoints


# --------------------------------------------------------------------------------------------------
# Driver: read the per-rebalance-date inputs from allocator_multiasset.py and attribute every date.
# --------------------------------------------------------------------------------------------------
def _run(inputs_path, n_grid, tail):
    with open(inputs_path, "rb") as f:
        blob = pickle.load(f)
    names = blob["sleeves"]; dates = blob["dates"]; recs = blob["records"]
    bp = 1e4
    pair_rows, date_rows = [], []
    for rec in recs:
        ix = rec["ix"]; Ds = rec["Ds"]; Rst = rec["R_stream"]; Rh = rec["R_holdings"]
        fut = rec["future_returns"]                     # (h, k) realized sleeve returns after the rebalance
        sc = 0.5 * (rec["sc_stream"] + rec["sc_c2"])    # common leverage: symmetric, realistic bp units
        table, direct, attributed, ep = path_attribution(
            Rst, Rh, Ds, fut, budgets=None, scales=sc, n_grid=n_grid)
        sl = [names[k] for k in ix]
        for _, row in table.iterrows():
            i, j = int(row["i"]), int(row["j"])
            pair_rows.append({"date": rec["date"], "sleeve_i": sl[i], "sleeve_j": sl[j],
                              "rho_stream": Rst[i, j], "rho_holdings": Rh[i, j],   # staleness (gap) vs crowding (level)
                              "delta_rho": row["delta_rho"],
                              "integrated_dPnl_dRho": row["integrated_dPnl_dRho"],
                              "pnl_attribution_bp": row["pnl_attribution"] * bp})
        # full book period P&L difference (each book at its OWN leverage) and the leverage channel
        wC = rec["w_c2"]; wS = rec["w_stream"]
        full_c2 = np.prod(1.0 + rec["sc_c2"] * (fut @ wC)) - 1.0
        full_st = np.prod(1.0 + rec["sc_stream"] * (fut @ wS)) - 1.0
        full_diff = full_c2 - full_st
        date_rows.append({"date": rec["date"],
                          "direct_bp": direct * bp, "attributed_bp": attributed * bp,
                          "recon_err_bp": (attributed - direct) * bp,
                          "full_book_diff_bp": full_diff * bp,
                          "scale_channel_bp": (full_diff - direct) * bp,
                          "stream_period_pnl": full_st,
                          "composition_intensity": rec.get("composition_intensity", np.nan)})
    pair = pd.DataFrame(pair_rows); dlog = pd.DataFrame(date_rows)
    tag = inputs_path.replace("allocator_attribution_inputs", "").replace(".pkl", "")
    pair.to_parquet(f"allocator_pnl_attribution{tag}_pairs.parquet")
    dlog.to_csv(f"allocator_pnl_attribution{tag}_datelog.csv", index=False, float_format="%.5f")

    # ---- reconstruction diagnostic (the identity the method must satisfy) ----
    med_abs_direct = dlog["direct_bp"].abs().median()
    med_abs_err = dlog["recon_err_bp"].abs().median()
    print(f"dates={len(dlog)}  n_grid={n_grid}")
    print(f"reconstruction of the allocation channel (should be ~0):")
    print(f"  median |direct|      {med_abs_direct:8.3f} bp   median |recon err| {med_abs_err:8.4f} bp"
          f"   ({med_abs_err/max(med_abs_direct,1e-9):.2%} of the signal)")
    print(f"  leverage (scale) channel: mean {dlog['scale_channel_bp'].mean():+.3f} bp/reb,"
          f" allocation (direct) mean {dlog['direct_bp'].mean():+.3f} bp/reb")

    # ---- which pairs carry the allocation difference, overall and in the adverse tail ----
    tot = pair.groupby(["sleeve_i", "sleeve_j"])["pnl_attribution_bp"].sum().sort_values(key=np.abs, ascending=False)
    print("\ntop sleeve pairs by |cumulative allocation attribution| (bp, full sample):")
    for (a, b), v in tot.head(10).items():
        print(f"  {a:14s} {b:14s} {v:+9.1f}")

    # tail dates = worst `tail` of the STREAM book's realized period P&L (the book the mechanism protects)
    if "stream_period_pnl" in dlog:
        thr = dlog["stream_period_pnl"].quantile(tail)
        tail_dates = set(dlog.loc[dlog["stream_period_pnl"] <= thr, "date"])
        tp = pair[pair["date"].isin(tail_dates)]
        tt = tp.groupby(["sleeve_i", "sleeve_j"])["pnl_attribution_bp"].sum().sort_values(key=np.abs, ascending=False)
        share = dlog.loc[dlog["date"].isin(tail_dates), "direct_bp"].sum() / dlog["direct_bp"].sum()
        print(f"\nadverse tail = worst {tail:.0%} of stream-book rebalances ({len(tail_dates)} dates), "
              f"{share:.0%} of the total allocation difference:")
        for (a, b), v in tt.head(8).items():
            print(f"  {a:14s} {b:14s} {v:+9.1f}")
    print(f"\nwrote allocator_pnl_attribution{tag}_pairs.parquet and _datelog.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    src = next((a for a in args if not a.startswith("--")), "allocator_attribution_inputs_L34_reb21.pkl")
    ng = next((int(a.split("=")[1]) for a in args if a.startswith("--ngrid=")), 41)
    tl = next((float(a.split("=")[1]) for a in args if a.startswith("--tail=")), 0.05)
    _run(src, ng, tl)
