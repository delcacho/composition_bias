"""divergence_tail_book.py -- does holdings-vs-stream divergence flag the RISK-PARITY BOOK's tail?

The object is the stream risk-parity book (the stale estimator an allocator runs), not a single pair.
At each date t, from information through t:
  stream RP weights w  = risk parity on the 252d sleeve-return covariance (what the allocator holds)
  book divergence gvar = (w' Sigma_H w) / (w' Sigma_stream w) - 1
        how much the stream UNDER-states the book's own variance; Sigma_H is the holdings-implied
        sleeve covariance (books priced through the 252d asset covariance), Sigma_stream the sleeve
        return covariance. gvar > 0 = the book is riskier than the stream thinks = under-hedged.
  (alt) max_gap = largest pairwise holdings-minus-stream sleeve correlation gap.
Forward: freeze w and take the book's 63-day path; dd = its max drawdown.

Instead of quintiles, cut at the extreme: forward drawdown by fine signal band (top 5% vs the rest),
and the share of the worst-decile drawdown dates that entered with the signal in its top 20/5/1 percent.
The result is a null: at the book level the divergence does not concentrate in the tail, so the
holdings correction's tail value is the alignment of Exhibit A7, not a forecastable divergence flag.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allocator_multiasset import load_commodity, _risk_parity_builtin

LOOKBACK, FWD = 252, 63


def corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None)); return C / np.outer(d, d)


dates, sleeves, assets, Wt, Rv = load_commodity()
T, K = len(dates), len(sleeves)
rs = np.zeros((T, K))
for i in range(K):
    rs[1:, i] = np.einsum("ta,ta->t", Wt[:-1, i, :], Rv[1:, :])
print(f"panel {dates[0].date()}..{dates[-1].date()}  T={T}  sleeves={K}: {sleeves}")

rows = []
for t in range(LOOKBACK, T - FWD):
    Sig = np.cov(Rv[t - LOOKBACK:t].T)
    Wc = Wt[t]
    G = np.array([[float(Wc[i] @ Sig @ Wc[j]) for j in range(K)] for i in range(K)])
    Cs = np.cov(rs[t - LOOKBACK:t].T)
    if np.any(np.diag(Cs) < 1e-14) or np.any(np.diag(G) < 1e-14):
        continue
    w = _risk_parity_builtin(Cs)
    vs_str = float(w @ Cs @ w); vs_hold = float(w @ G @ w)
    if vs_str <= 0:
        continue
    gvar = vs_hold / vs_str - 1.0
    iu = np.triu_indices(K, 1)
    max_gap = float((corr(G)[iu] - corr(Cs)[iu]).max())
    pr = rs[t + 1:t + 1 + FWD] @ w
    cum = np.cumprod(1.0 + pr); dd = float((cum / np.maximum.accumulate(cum) - 1.0).min())
    rows.append((dates[t], gvar, max_gap, dd))

d = pd.DataFrame(rows, columns=["date", "gvar", "maxgap", "dd"]).set_index("date")
n = len(d)
big = d.dd <= d.dd.quantile(0.10)
print(f"forecast dates {n} (~{n//FWD} quarters); worst-decile drawdown threshold {d.dd.quantile(0.10):+.3f}")

for signame, s in [("book divergence gvar", d.gvar), ("max pairwise correlation gap", d.maxgap)]:
    p = s.rank(pct=True)
    bands = [("0-50%", p <= 0.50), ("50-80%", (p > 0.50) & (p <= 0.80)),
             ("80-95%", (p > 0.80) & (p <= 0.95)), ("95-99%", (p > 0.95) & (p <= 0.99)),
             ("top 1%", p > 0.99)]
    print(f"\n=== forward drawdown by {signame} band ===")
    for lab, m in bands:
        g = d[m]
        print(f"  {lab:8s} n={len(g):4d}  mean fwd dd {g.dd.mean():+.4f}  "
              f"P(worst-decile dd) {(g.dd <= d.dd.quantile(0.10)).mean():.2f}")
    print(f"  worst-decile-dd dates with the signal in top quintile {(p[big] >= 0.80).mean():.2f} / "
          f"top 5% {(p[big] >= 0.95).mean():.2f} / top 1% {(p[big] >= 0.99).mean():.2f}   (base 0.20/0.05/0.01)")

d.to_csv("divergence_tail_book_dates.csv", float_format="%.6g")
