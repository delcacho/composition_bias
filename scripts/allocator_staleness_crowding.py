"""allocator_staleness_crowding.py -- does the c2-minus-stream tail protection come from STALENESS or
from CROWDING?

The pathwise attribution (allocator_pnl_attribution.py) gives, per rebalance date and sleeve pair, the
realized P&L attributable to that pair's correlation moving from the stream's (stale, backward EWMA)
value to the holdings' (current) value. This asks WHICH KIND of pair carries the drawdown protection,
because two different stories predict the same aggregate tail advantage:

  STALENESS: the stream's correlation is out of date -- a large gap |delta_rho| = |rho_holdings - rho_stream|.
             The mechanism in the paper: the stream under-estimates a correlation SPIKE (delta_rho > 0),
             so risk parity thinks the pair diversifies and over-concentrates into it; c2 already saw the
             spike and held them apart, so it loses less when the correlated move lands in the tail.
  CROWDING:  the holdings correlation LEVEL rho_holdings is high -- the two sleeves genuinely converged,
             independent of whether the stream had caught up.

These are separable (a pair can be stale without being crowded, or crowded without being stale). The
discriminator runs three ways, on the adverse tail (worst `tail` of stream-book rebalances) and, as a
falsification, on the calm body:

  1. SIGN split -- share of protection from delta_rho > 0 (stream under-estimated) vs < 0. The paper's
     mechanism predicts the positive side dominates in the tail.
  2. 2x2 SORT -- median split of |delta_rho| (staleness) x rho_holdings (crowding); mean protection per
     cell says whether it sits in the stale column or the crowded row.
  3. JOINT REGRESSION with date fixed effects -- protection ~ z(|delta_rho|) + z(rho_holdings), demeaned
     within date so it is a purely cross-pair comparison, date-cluster bootstrap CIs. The INCREMENTAL
     coefficient is the honest test: |delta_rho| loads on the attribution partly by construction
     (A = delta_rho x sensitivity), so the question that is NOT tautological is whether crowding adds
     explanatory power BEYOND the divergence size, and vice versa. Reported for the attribution and,
     tautology-free, for the integrated sensitivity (the economic part, without the delta_rho multiplier).

The tail should show a stronger loading than the body -- that is the "bites the tail, dilutes in the
body" claim. If protection turns out to be generic rotation (neither feature, or the crowding level with
the wrong sign), that is reported plainly; it is arguably the more interesting finding, not a
failure.

usage: python research/allocator_staleness_crowding.py [pairs.parquet] [--datelog=..] [--tail=0.05]
                                                       [--nboot=2000] [--abscrowd]
"""
import sys
import numpy as np
import pandas as pd


def _demean_within(df, cols, by="date"):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def _ols(y, X):
    """OLS coefficients through the given design (no added intercept); X columns already chosen."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _cluster_bootstrap(df, yc, fcols, nboot, rng):
    """Date-clustered bootstrap of standardized coefficients of y ~ features (date-demeaned, no intercept).
    Returns dict feature -> (point, lo, hi) and the point/CI of the coefficient DIFFERENCE f0 - f1."""
    groups = [g for _, g in df.groupby("date")]           # keyed by integer position (datetime keys are fragile)
    ng = len(groups)
    # point estimate on the full sample
    dd = _demean_within(df, [yc] + fcols)
    sy = dd[yc].std() or 1.0
    Z = np.column_stack([dd[c] / (dd[c].std() or 1.0) for c in fcols])
    b_point = _ols(dd[yc].to_numpy() / sy, Z)
    draws = np.empty((nboot, len(fcols)))
    for b in range(nboot):
        pick = rng.integers(0, ng, ng)
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        dd = _demean_within(sample, [yc] + fcols)
        sy = dd[yc].std() or 1.0
        Z = np.column_stack([dd[c] / (dd[c].std() or 1.0) for c in fcols])
        try:
            draws[b] = _ols(dd[yc].to_numpy() / sy, Z)
        except np.linalg.LinAlgError:
            draws[b] = np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    diff = draws[:, 0] - draws[:, 1]
    dlo, dhi = np.nanpercentile(diff, [2.5, 97.5])
    return ({fcols[k]: (b_point[k], lo[k], hi[k]) for k in range(len(fcols))},
            (b_point[0] - b_point[1], dlo, dhi))


def _report(df, label, fcols, nboot, rng):
    print(f"\n================  {label}  ({len(df)} pair-days, {df['date'].nunique()} dates)  ================")
    A = df["pnl_attribution_bp"]
    print(f"total protection (sum A) = {A.sum():+.1f} bp   mean per pair-day = {A.mean():+.3f} bp")

    # 1. sign of the staleness gap
    pos = df[df["delta_rho"] > 0]["pnl_attribution_bp"].sum()
    neg = df[df["delta_rho"] < 0]["pnl_attribution_bp"].sum()
    tot = pos + neg
    print("\n1. SIGN of the gap (mechanism predicts delta_rho>0 dominates in the tail):")
    print(f"   delta_rho>0 (stream UNDER-estimated correlation): {pos:+.1f} bp  ({pos/tot:>6.0%} of total)")
    print(f"   delta_rho<0 (stream OVER-estimated correlation) : {neg:+.1f} bp  ({neg/tot:>6.0%} of total)")

    # 2. 2x2 median sort of staleness x crowding
    stale_hi = df["abs_dr"] >= df["abs_dr"].median()
    crowd_hi = df["crowd"] >= df["crowd"].median()
    print("\n2. 2x2 SORT, mean protection per pair-day (bp)   [staleness = |delta_rho|, crowding = rho_holdings]:")
    print(f"   {'':16s}{'crowd LOW':>12s}{'crowd HIGH':>12s}")
    for slab, smask in [("stale LOW", ~stale_hi), ("stale HIGH", stale_hi)]:
        lo = df[smask & ~crowd_hi]["pnl_attribution_bp"].mean()
        hi = df[smask & crowd_hi]["pnl_attribution_bp"].mean()
        print(f"   {slab:16s}{lo:>12.3f}{hi:>12.3f}")
    corr = np.corrcoef(df["abs_dr"], df["crowd"])[0, 1]
    print(f"   (feature correlation |delta_rho| vs rho_holdings = {corr:+.2f}; near 0 = cleanly separable)")

    # 3. joint regression, date fixed effects, incremental (non-tautological) test
    print("\n3. JOINT REGRESSION, date-demeaned, standardized coefficients, date-cluster bootstrap 95% CI:")
    for yc, tag in [("pnl_attribution_bp", "attribution A  (|delta_rho| term partly mechanical)"),
                    ("integrated_dPnl_dRho", "sensitivity    (economic part, delta_rho-free)")]:
        coefs, (dpt, dlo, dhi) = _cluster_bootstrap(df, yc, fcols, nboot, rng)
        (sp, slo, shi) = coefs["abs_dr"]; (cp, clo, chi) = coefs["crowd"]
        print(f"   {tag}")
        print(f"     staleness |delta_rho| : {sp:+.3f}  [{slo:+.3f}, {shi:+.3f}]")
        print(f"     crowding  rho_holdings: {cp:+.3f}  [{clo:+.3f}, {chi:+.3f}]")
        verdict = ("STALENESS" if dlo > 0 else "CROWDING" if dhi < 0 else "UNDECIDABLE (CIs overlap)")
        print(f"     staleness - crowding  : {dpt:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]  -> {verdict}")


def _run(pairs_path, datelog_path, tail, nboot, abscrowd):
    pair = pd.read_parquet(pairs_path)
    dlog = pd.read_csv(datelog_path, parse_dates=["date"])
    pair["date"] = pd.to_datetime(pair["date"])
    pair = pair.merge(dlog[["date", "stream_period_pnl"]], on="date", how="left")
    pair["abs_dr"] = pair["delta_rho"].abs()
    pair["crowd"] = pair["rho_holdings"].abs() if abscrowd else pair["rho_holdings"]

    thr = dlog["stream_period_pnl"].quantile(tail)
    tail_dates = set(dlog.loc[dlog["stream_period_pnl"] <= thr, "date"])
    tp = pair[pair["date"].isin(tail_dates)].copy()
    bp = pair[~pair["date"].isin(tail_dates)].copy()

    rng = np.random.default_rng(11)
    fcols = ["abs_dr", "crowd"]
    print(f"crowding feature = {'|rho_holdings|' if abscrowd else 'rho_holdings (signed)'};  "
          f"adverse tail = worst {tail:.0%} of stream-book rebalances")
    _report(tp, f"ADVERSE TAIL (worst {tail:.0%})", fcols, nboot, rng)
    _report(bp, "CALM BODY (rest)", fcols, nboot, rng)
    print("\nfalsification: the tail should load more strongly (and with the mechanism's sign) than the body.")

    # ---- the insurance table: every input is flat across states; the cost moves through ALIGNMENT ----
    # protection = mean(A) = mean|A| * alignment, with alignment = sum(A)/sum|A| in [-1,1] the sign structure
    # (whether the pairs' economic effects cancel or point together). Exact factorization.
    def alignment(s):
        A = s.pnl_attribution_bp; return A.sum() / A.abs().sum()
    metrics = [("staleness |delta_rho|", lambda s: s.abs_dr.mean()),
               ("current dependence |rho_holdings|", lambda s: s.rho_holdings.abs().mean()),
               ("P&L sensitivity |dP/drho|", lambda s: s.integrated_dPnl_dRho.abs().mean()),
               ("gross effect mean|A| (bp)", lambda s: s.pnl_attribution_bp.abs().mean()),
               ("alignment sumA/sum|A|", alignment),
               ("protection (bp/pair-day)", lambda s: s.pnl_attribution_bp.mean())]
    rows = []
    print(f"\nINSURANCE TABLE (every input flat; the cost moves through ALIGNMENT):\n"
          f"  {'metric':34s}{'tail':>9}{'calm':>9}{'ratio':>8}")
    for name, fn in metrics:
        tv, cv = fn(tp), fn(bp)
        rr = tv / cv if abs(cv) > 1e-12 else float("nan")
        print(f"  {name:34s}{tv:>9.4f}{cv:>9.4f}{rr:>7.2f}x")
        rows.append({"metric": name, "tail": tv, "calm": cv, "ratio": rr})
    out = pairs_path.replace("_pairs.parquet", "_insurance.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5f")
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    src = next((a for a in args if not a.startswith("--")), "allocator_pnl_attribution_L34_reb21_pairs.parquet")
    dl = next((a.split("=", 1)[1] for a in args if a.startswith("--datelog=")),
              src.replace("_pairs.parquet", "_datelog.csv"))
    tl = next((float(a.split("=")[1]) for a in args if a.startswith("--tail=")), 0.05)
    nb = next((int(a.split("=")[1]) for a in args if a.startswith("--nboot=")), 2000)
    ac = "--abscrowd" in args
    _run(src, dl, tl, nb, ac)
