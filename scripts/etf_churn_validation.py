"""Validate the churn detector (Exhibit 9 recipe) on the active-ETF panel, the one place with a true
daily phibar(63). For each fund, mimic an opaque manager: subsample its daily holdings to a periodic
filing (every DELTA active days), hold each filing frozen, price it forward (the frozen-filing
return), and estimate persistence as the detector would: regress the frozen-filing return and the
reported return on the market factor and keep the residuals (recipe step 4), aggregate to monthly
(21-day) blocks, correlate, pool. Compare to the fund's true phibar_neutral_63.

Reported return: the fund's real daily NAV (massive_flows) when present; otherwise the fund's own
daily-book clone, the paper's clone-not-NAV world. Headline number: P(read >= 0.75 | true < 0.75),
the costly upward misread. Neutralization uses the panel's equal-weight return as the market factor
and one beta per fund (the same-hedge convention of the staleness experiment).

Reads existing ETF data (books, returns, massive_flows NAVs if fetched). No new fetch.
Usage: python research/etf_churn_validation.py            (DELTA=63, quarterly filings)
       python research/etf_churn_validation.py 21         (monthly filings)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex

DELTA = int(sys.argv[1]) if len(sys.argv) > 1 else 63    # filing cadence in active days
BLOCK = 21                                               # monthly aggregation
CUT = 0.75
MIN_BLOCKS = 12                                          # >= a year of monthly blocks to read a fund

truth = pd.read_csv("etf_results/persistence_daily.csv").set_index("ticker")
sample = pd.read_csv(er.U)
admit = sample.in_sample_staleness if "in_sample_staleness" in sample.columns else sample.in_sample
tickers = sorted(sample[admit].ticker.astype(str))

Rs = er.load_returns()
names = list(Rs.columns); name_ix = {n: i for i, n in enumerate(names)}
books = er.load_books(tickers, names, name_ix, Rs)
navs = er.load_navs(list(books.keys()), Rs.index)       # real NAV log-returns, daily (empty if not fetched)
print(f"funds with books {len(books)}; with a real NAV {navs.shape[1]}; filing cadence {DELTA}d")

# frozen-filing books: hold each fund's book at its most recent snapshot (every DELTA of its own rows)
frozen = {t: (dates, W[(np.arange(W.shape[0]) // DELTA) * DELTA], pr) for t, (dates, W, pr) in books.items()}
fro = er.clone_returns(frozen, Rs)                      # frozen-filing return, daily
act = er.clone_returns(books, Rs)                       # the fund's own daily-book clone (NAV proxy)
market = act.mean(axis=1)                               # panel equal-weight return: the market factor

# clone-to-NAV fidelity: how well the daily-book clone tracks the reported NAV, day by day
fid = []
for t in books:
    if t in navs.columns:
        d = pd.concat([act[t].rename("c"), navs[t].rename("n")], axis=1).dropna()
        if len(d) > 250 and d.c.std() > 0 and d.n.std() > 0:
            fid.append(float(np.corrcoef(d.c, d.n)[0, 1]))
fid = np.array(fid)
if len(fid):
    print(f"clone-to-NAV daily-return correlation: median {np.median(fid):.3f}, "
          f"IQR [{np.percentile(fid, 25):.3f}, {np.percentile(fid, 75):.3f}], "
          f"5th pct {np.percentile(fid, 5):.3f}, n={len(fid)}")
    pd.DataFrame({"fidelity": np.round(fid, 4)}).to_csv("etf_results/clone_nav_fidelity.csv", index=False)

def beta_on_market(s):
    d = pd.concat([s.rename("s"), market.rename("m")], axis=1).dropna()
    if len(d) < 250 or d.m.var() < 1e-12:
        return 0.0
    return float(np.cov(d.s, d.m)[0, 1] / d.m.var())

def neutral_block_corr(frozen_s, report_s, beta):
    """residualize both on the market (one beta), sum 21-day blocks, correlate."""
    d = pd.concat([frozen_s.rename("f"), report_s.rename("r"), market.rename("m")], axis=1).dropna()
    if len(d) < BLOCK * MIN_BLOCKS:
        return np.nan, 0
    fr = (d.f - beta * d.m).values
    rr = (d.r - beta * d.m).values
    m = len(d) // BLOCK
    A = fr[:m * BLOCK].reshape(m, BLOCK).sum(1)
    B = rr[:m * BLOCK].reshape(m, BLOCK).sum(1)
    if A.std() < 1e-9 or B.std() < 1e-9:
        return np.nan, m
    return float(np.corrcoef(A, B)[0, 1]), m

rows = []
for t in books:
    if t not in truth.index:
        continue
    beta = beta_on_market(act[t])
    tru = float(truth.loc[t, "phibar_neutral_63"])
    est_nav, m_nav = (neutral_block_corr(fro[t], navs[t], beta) if t in navs.columns else (np.nan, 0))
    est_cln, m_cln = neutral_block_corr(fro[t], act[t], beta)
    rows.append({"ticker": t, "true_phibar63": tru, "beta": round(beta, 3),
                 "est_nav": est_nav, "n_nav": m_nav, "est_clone": est_cln, "n_clone": m_cln})
R = pd.DataFrame(rows)
cls_of = ex.fund_classes()
R["class"] = R.ticker.map(lambda t: cls_of.get(t, "longonly"))
R.to_csv(f"etf_results/churn_validation_delta{DELTA}.csv", index=False)

def report(col, label):
    d = R[["true_phibar63", col]].dropna()
    d = d[np.isfinite(d[col])]
    if len(d) < 20:
        print(f"\n[{label}] too few funds ({len(d)}) -- likely no NAVs fetched"); return
    tv, ev = d.true_phibar63.values, d[col].values
    print(f"\n===== churn detector estimate vs truth: {label}  (n={len(d)} funds) =====")
    print(f"  bias mean(est - true): {np.mean(ev - tv):+.3f}   median {np.median(ev - tv):+.3f}")
    print(f"  Spearman(est, true): {pd.Series(ev).corr(pd.Series(tv), 'spearman'):+.3f}   "
          f"Pearson: {np.corrcoef(ev, tv)[0,1]:+.3f}")
    for lo, hi, nm in [(0, 0.75, "true rotators <0.75"), (0.75, 0.85, "0.75-0.85"),
                       (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, "frozen >0.95")]:
        g = d[(tv >= lo) & (tv < hi)]
        if len(g):
            print(f"  {nm:22s} n={len(g):3d}  mean est {g[col].mean():.3f}  read >= {CUT}: {(g[col]>=CUT).mean()*100:4.0f}%")
    rot = d[tv < CUT]
    if len(rot):
        print(f"  costly upward misread  P(read >= {CUT} | true < {CUT}) = {(rot[col]>=CUT).mean()*100:.0f}%  (n={len(rot)})")
    print(f"  class accuracy at {CUT} cut: {((tv<CUT)==(ev<CUT)).mean()*100:.0f}%")

report("est_nav", "frozen filing vs real NAV (recipe)")
report("est_clone", "frozen filing vs own clone (paper's clone world)")

def summary_rows():
    """The two-row summary shipped as data/etf_panel/churn_validation_summary.csv (accuracy at the
    cut, the no-skill majority-class baseline, Spearman, and the upward and downward misread rates),
    written from the same per-fund estimates as report()."""
    out = []
    for col, label in [("est_nav", "real_nav"), ("est_clone", "clone")]:
        d = R[["true_phibar63", col]].dropna(); d = d[np.isfinite(d[col])]
        if len(d) < 20:
            continue
        tv, ev = d.true_phibar63.values, d[col].values; rot = tv < CUT
        out.append({"reported_return": label, "n": len(d),
                    "accuracy_pct": round(((tv < CUT) == (ev < CUT)).mean() * 100, 1),
                    "noskill_baseline_pct": round(max(rot.mean(), 1 - rot.mean()) * 100, 1),
                    "spearman": round(pd.Series(ev).corr(pd.Series(tv), "spearman"), 3),
                    "P_up_given_rotator_pct": round((ev[rot] >= CUT).mean() * 100, 1),
                    "P_persistent_read_rotator_pct": round((ev[~rot] < CUT).mean() * 100, 1)})
    pd.DataFrame(out).to_csv("etf_results/churn_validation_summary.csv", index=False)

summary_rows()

def by_class(col, label):
    d = R[["true_phibar63", col, "class"]].dropna()
    d = d[np.isfinite(d[col])]
    if len(d) < 20:
        return
    print(f"\n----- {label}: by strategy class (composition risk share differs) -----")
    print(f"  {'class':12s} {'n':>4s} {'rotators':>8s} {'P(up|rot)':>10s} {'accuracy':>9s}")
    for cl, g in d.groupby("class"):
        rot = g[g.true_phibar63 < CUT]
        up = f"{(rot[col]>=CUT).mean()*100:.0f}%" if len(rot) else "  -"
        acc = ((g.true_phibar63 < CUT) == (g[col] < CUT)).mean() * 100
        print(f"  {cl:12s} {len(g):4d} {len(rot):8d} {up:>10s} {acc:8.0f}%")

by_class("est_nav", "real NAV")
print(f"\nwrote etf_results/churn_validation_delta{DELTA}.csv")
