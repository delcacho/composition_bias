"""allocator_es_attribution.py -- pathwise decomposition of the EXPECTED-SHORTFALL gap ES_H - ES_R
between the holdings-correlation book (c2) and the return-stream book, attributed to each sleeve-pair
correlation change.

WHY THIS AND NOT THE PER-DATE ATTRIBUTION. allocator_pnl_attribution.py attributes the realized c2-minus-
stream P&L per rebalance; summing it over "tail dates picked by the stream book's losses" CONDITIONS on one
treatment's outcome (the stream's), which is circular for an expected-shortfall claim. Expected shortfall
has a clean pathwise derivative instead (Gourieroux-Laurent-Scaillet):

    d ES_alpha(lambda) / d lambda = E[ dP_t(lambda)/d lambda  |  P_t(lambda) <= VaR_alpha(lambda) ],

where the tail set { P_t <= VaR(lambda) } is RE-EVALUATED at every lambda -- the stream's tail at lambda=0,
the holdings' tail at lambda=1, moving with the covariance in between, conditioning on NO book's outcome.
With the correlation moving along Sigma(lambda)=D_R[R_R+lambda(R_H-R_R)]D_R (vols fixed, the blend's clean
intervention) and dP_t/d lambda = sum_ij (dP_t/drho_ij) delta_rho_ij from the ERC weight sensitivity,

    ES_H - ES_R  =  sum_ij  integral_0^1  E[ (dP_t/drho_ij) delta_rho_ij | tail(lambda) ] d lambda
                 =  sum_ij  A^ES_ij .

Matched leverage: like the P&L attribution, each rebalance is run at a COMMON scale (the average of the two
books' leverages), fixed across lambda, so the decomposed ES gap is the CORRELATION channel at matched vol --
the comparison ex:mdd_es makes. The separate leverage channel (c2's ~6% higher gross) is not in these numbers.

CAVEAT this is RETURN expected shortfall (shortfall of the daily return distribution), which has the exact
envelope derivative above. ex:mdd_es reports DRAWDOWN shortfall (a path-max), which is differentiable only via
Danskin and is a heavier, separate build; this return-ES decomposition is the clean first cut.

Reads allocator_multiasset.py's allocator_attribution_inputs*.pkl (per-rebalance correlation matrices, vols,
leverages and the holding-period daily returns). Writes a pair-level ES attribution table per alpha.

usage: python research/allocator_es_attribution.py [inputs.pkl] [--ngrid=81] [--alphas=0.20,0.10,0.05,0.02]
"""
import sys
import pickle
import numpy as np
import pandas as pd
from allocator_pnl_attribution import solve_erc_x, _floor_corr, local_pair_weight_jacobian


def _build_paths(recs, names, lam, kappa_max, x0cache):
    """At correlation-path position `lam`, build the full daily portfolio P&L and the per-day, per-GLOBAL-pair
    sensitivity dP_t/d lambda (attributed to each pair). Returns (P_all (T,), S_all (T, n_pairs), pair_index)."""
    n = len(names)
    gpairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    gidx = {p: k for k, p in enumerate(gpairs)}
    P_list, S_list = [], []
    for kk, rec in enumerate(recs):
        ix = rec["ix"]; Ds = rec["Ds"]; R0 = rec["R_stream"]; R1 = rec["R_holdings"]
        fut = rec["future_returns"]                      # (h, k_active) holding-period daily returns
        sc = 0.5 * (rec["sc_stream"] + rec["sc_c2"])     # common leverage, fixed across lambda
        R = _floor_corr(R0, kappa_max) + lam * (_floor_corr(R1, kappa_max) - _floor_corr(R0, kappa_max))
        cov = np.outer(Ds, Ds) * R
        x, w, b = solve_erc_x(cov, budgets=None, x0=x0cache.get(kk))
        x0cache[kk] = x
        pairs, dW = local_pair_weight_jacobian(cov, x, b, Ds)     # (n_local_pairs, k_active)
        drho = np.array([(_floor_corr(R1, kappa_max) - _floor_corr(R0, kappa_max))[i, j] for i, j in pairs])
        h = fut.shape[0]
        P = sc * (fut @ w)                               # (h,) daily book return this window
        # per-day sensitivity of each local pair: sc * delta_rho_ij * (dW_ij . r_d)
        contrib = sc * (fut @ dW.T) * drho[None, :]      # (h, n_local_pairs)
        S = np.zeros((h, len(gpairs)))
        for lp, (li, lj) in enumerate(pairs):
            S[:, gidx[(ix[li], ix[lj])]] = contrib[:, lp]
        P_list.append(P); S_list.append(S)
    return np.concatenate(P_list), np.concatenate(S_list), gpairs


def es_attribution(inputs, n_grid=81, alphas=(0.20, 0.10, 0.05, 0.02), kappa_max=1e3):
    if isinstance(inputs, dict):                          # accept an in-memory blob (robustness resampling)
        blob = inputs
    else:
        with open(inputs, "rb") as f:
            blob = pickle.load(f)
    names = blob["sleeves"]; recs = blob["records"]
    lambdas = np.linspace(0.0, 1.0, n_grid)
    # daily P&L path and per-pair sensitivity at every lambda (warm-started across lambda)
    x0cache = {}
    P_grid, S_grid, gpairs = [], [], None
    for lam in lambdas:
        P, S, gpairs = _build_paths(recs, names, lam, kappa_max, x0cache)
        P_grid.append(P); S_grid.append(S)
    P_grid = np.array(P_grid)                            # (n_grid, T)
    S_grid = np.array(S_grid)                            # (n_grid, T, n_pairs)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
    bp = 1e4
    results = {}
    for a in alphas:
        dES = np.empty((n_grid, len(gpairs)))
        es_endpoints = {}
        for g in range(n_grid):
            P = P_grid[g]; var = np.quantile(P, a); tail = P <= var
            dES[g] = S_grid[g][tail].mean(axis=0)        # d ES/d lambda per pair = mean of dP/dlambda over tail(lambda)
            if g == 0:
                es_endpoints["R"] = P[tail].mean()
            if g == n_grid - 1:
                es_endpoints["H"] = P[tail].mean()
        A = _trapz(dES, lambdas, axis=0)                 # per-pair ES attribution
        direct = es_endpoints["H"] - es_endpoints["R"]
        tbl = pd.DataFrame({"sleeve_i": [names[i] for i, _ in gpairs],
                            "sleeve_j": [names[j] for _, j in gpairs],
                            "es_attribution_bp": A * bp}).sort_values("es_attribution_bp", key=np.abs, ascending=False)
        results[a] = {"table": tbl, "ES_R_bp": es_endpoints["R"] * bp, "ES_H_bp": es_endpoints["H"] * bp,
                      "direct_bp": direct * bp, "attributed_bp": A.sum() * bp}
    return results, gpairs


def _run(inputs_path, n_grid, alphas):
    results, _ = es_attribution(inputs_path, n_grid=n_grid, alphas=alphas)
    tag = inputs_path.replace("allocator_attribution_inputs", "").replace(".pkl", "")
    print(f"ES decomposition (return shortfall, matched leverage, n_grid={n_grid})")
    print(f"{'alpha':>6} {'ES_R':>9} {'ES_H':>9} {'gap H-R':>9} {'attributed':>11} {'recon err':>10}")
    frames = []
    for a, r in results.items():
        rec = (r['attributed_bp'] - r['direct_bp'])
        print(f"{a:>6.2f} {r['ES_R_bp']:>9.2f} {r['ES_H_bp']:>9.2f} {r['direct_bp']:>+9.2f} "
              f"{r['attributed_bp']:>+11.2f} {rec:>+10.3f}  bp"
              f"   ({abs(rec)/max(abs(r['direct_bp']),1e-9):.1%} of gap)")
        t = r["table"].copy(); t.insert(0, "alpha", a); frames.append(t)
        if abs(a - 0.05) < 1e-9:
            print("   top pairs (alpha=0.05), bp of the ES gap:")
            for _, row in r["table"].head(8).iterrows():
                print(f"     {row['sleeve_i']:14s} {row['sleeve_j']:14s} {row['es_attribution_bp']:+8.2f}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(f"allocator_es_attribution{tag}.csv", index=False, float_format="%.4f")
    print(f"\nwrote allocator_es_attribution{tag}.csv")
    print("gap H-R > 0 means the holdings book's shortfall is SHALLOWER (less negative) than the stream's.")


if __name__ == "__main__":
    args = sys.argv[1:]
    src = next((a for a in args if not a.startswith("--")), "allocator_attribution_inputs_L34_reb21.pkl")
    ng = next((int(a.split("=")[1]) for a in args if a.startswith("--ngrid=")), 81)
    al = next((tuple(float(x) for x in a.split("=")[1].split(",")) for a in args if a.startswith("--alphas=")),
              (0.20, 0.10, 0.05, 0.02))
    _run(src, ng, al)
