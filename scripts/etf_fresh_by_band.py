"""etf_fresh_by_band.py -- is the fresh holdings edge on real managers graded by rotation?

On the pooled full-window chunks (halflife 63, market-neutral), the fresh (age 0) pair-date loss difference of the
holdings correlation against the stream, cH - cS, is regressed on the four fixed persistence-band dummies of the
pair's less persistent fund, with standard errors two-way clustered by that fund and by date (twoway_ols of the
staleness experiment). Reported: the four band coefficients, the fastest-minus-frozen contrast, the slope of the
same loss difference on 1 - phibar(63) with an intercept, and the fastest-minus-frozen contrast for the oracle book
(the book FUTURE days ahead). Reads existing chunks only.

  python research/etf_fresh_by_band.py [dir]      default etf_results/experiment/pooled_neutral_all_L63_fullwin
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_staleness_experiment as ex

d = sys.argv[1] if len(sys.argv) > 1 else "etf_results/experiment/pooled_neutral_all_L63_fullwin"
FUTURE = -ex.FUTURE if hasattr(ex, "FUTURE") else -10
BANDS = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]


def band_design(age):
    def design(C):
        Xd = np.zeros((len(C), 4)); m = (C.age.values == age) & (C.qmin.values >= 1)
        for i, (lo, hi, _) in enumerate(BANDS):
            Xd[:, i] = (m & (C.phimin.values >= lo) & (C.phimin.values < hi)).astype(float)
        return Xd
    return design


def slope_design(age):
    def design(C):
        m = ((C.age.values == age) & (C.qmin.values >= 1)).astype(float)
        return np.column_stack([m, m * (1.0 - C.phimin.values)])
    return design


y_of = lambda C: (C.cH - C.cS).values
future_age = sorted(a for a in pd.read_parquet(next(iter(sorted(__import__("glob").glob(f"{d}/chunks/*.parquet")))), columns=["age"]).age.unique() if a < 0)
future_age = future_age[0] if future_age else FUTURE
rows = []
for label, age in [("fresh", 0), ("oracle", future_age)]:
    beta, V, nobs, used = ex.twoway_ols(d, y_of, band_design(age), 4)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    for i, (_, _, b) in enumerate(BANDS):
        rows.append({"book": label, "term": b, "coef": beta[i], "se": se[i], "t": beta[i] / se[i] if se[i] > 0 else np.nan, "nobs": nobs})
    c = np.array([1.0, 0, 0, -1.0]); dlt = c @ beta; s = float(np.sqrt(max(c @ V @ c, 0)))
    rows.append({"book": label, "term": "fastest - frozen", "coef": dlt, "se": s, "t": dlt / s if s > 0 else np.nan, "nobs": nobs})
    beta, V, nobs, used = ex.twoway_ols(d, y_of, slope_design(age), 2)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    rows.append({"book": label, "term": "slope on 1 - phibar", "coef": beta[1], "se": se[1], "t": beta[1] / se[1] if se[1] > 0 else np.nan, "nobs": nobs})
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
R.to_csv(f"{d}/fresh_by_band.csv", index=False, float_format="%.6f")
print("wrote", f"{d}/fresh_by_band.csv")
