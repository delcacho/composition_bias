"""
divergence_predicts_losses.py -- the allocator-relevant test. Does book-level divergence forecast
losses in the return-stream risk-parity book (the stale estimator an allocator would run)?

At each date t: build risk-parity weights on the stream sleeve covariance, freeze them, and measure the
frozen book's forward 63-day path. Signal = book divergence (max pairwise holdings-minus-stream gap).
Outcomes, top-minus-bottom divergence quintile (63-day block bootstrap): forward 63-day drawdown; a
large forward loss (21-day return in its bottom decile); forward 63-day vol; and, as a control, the
mean forward return (should be roughly flat: divergence forecasts risk, not direction).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from allocator_multiasset import load_commodity, _risk_parity_builtin

LOOKBACK, FWD, TAIL_PCT = 252, 63, 10 # bottom decile = a large loss
SEED, NBOOT, BLOCK = 7, 2000, 63


def block_boot_diff(f, top, bot, rng):
    n = len(f); nb = int(np.ceil(n / BLOCK)); out = []
    for _ in range(NBOOT):
        st = rng.integers(0, n - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n]
        a, tp, bt = f[idx], top[idx], bot[idx]
        if tp.sum() and bt.sum():
            out.append(a[tp].mean() - a[bt].mean())
    return np.quantile(out, [0.05, 0.95])


def corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None)); return C / np.outer(d, d)


def main():
    cdates, sleeves, assets, Wt, Rv = load_commodity()
    T, K = len(cdates), len(sleeves)
    rs = np.zeros((T, K))
    for i in range(K):
        rs[1:, i] = np.einsum("ta,ta->t", Wt[:-1, i, :], Rv[1:, :])

    rows = []
    for t in range(LOOKBACK, T - FWD):
        Sig = np.cov(Rv[t - LOOKBACK:t].T)
        Wc = Wt[t]; vs = np.array([float(Wc[i] @ Sig @ Wc[i]) for i in range(K)])
        if np.any(vs <= 0):
            continue
        RH = np.array([[float(Wc[i] @ Sig @ Wc[j]) / np.sqrt(vs[i] * vs[j]) for j in range(K)] for i in range(K)])
        Cstream = np.cov(rs[t - LOOKBACK:t].T)
        if np.any(np.diag(Cstream) < 1e-14):
            continue
        RR = corr(Cstream)
        iu = np.triu_indices(K, 1)
        max_gap = float((RH[iu] - RR[iu]).max())
        w = _risk_parity_builtin(Cstream) # stream risk-parity book, frozen at t
        pr = rs[t + 1:t + 1 + FWD] @ w # frozen forward path
        cum = np.cumprod(1.0 + pr); dd = float((cum / np.maximum.accumulate(cum) - 1.0).min())
        r21 = float(pr[:21].sum()); r63 = float(pr.sum())
        vol = float(pr.std(ddof=0) * np.sqrt(252))
        rows.append((max_gap, dd, r21, r63, vol))
    d = pd.DataFrame(rows, columns=["gap", "dd", "r21", "r63", "vol"])
    d["bigloss"] = (d.r21 < np.percentile(d.r21, TAIL_PCT)).astype(float) # left tail
    d["q"] = pd.qcut(d.gap, 5, labels=False)
    print(f"forecast dates {len(d)} (~{len(d)//FWD} quarters); stream risk-parity book, weights frozen forward 63d")

    print("\nquintile of book divergence: mean fwd drawdown | P(large 21d loss) | mean fwd 63d ret | fwd vol")
    for q, g in d.groupby("q"):
        print(f" Q{q+1}: dd {g.dd.mean():+.3f} | bigloss {g.bigloss.mean():.3f} | r63 {g.r63.mean():+.3f} | vol {g.vol.mean():.3f}")

    rng = np.random.default_rng(SEED); top, bot = (d.q == 4).values, (d.q == 0).values
    print("\n=== top-minus-bottom divergence quintile (63-day block bootstrap 90% CI) ===")
    d.to_csv("divergence_predicts_losses_dates.csv", index=False, float_format="%.6g") # per-date panel, persisted
    summary = []
    for k, lbl in (("dd", "fwd drawdown"), ("bigloss", "P(large 21d loss)"), ("vol", "fwd vol"), ("r63", "mean fwd 63d ret (control)")):
        f = d[k].values; diff = f[top].mean() - f[bot].mean(); lo, hi = block_boot_diff(f, top, bot, rng)
        print(f" {lbl:28s}: top {f[top].mean():+.3f} bottom {f[bot].mean():+.3f} diff {diff:+.3f} "
              f"CI [{lo:+.3f},{hi:+.3f}] {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")
        summary.append({"measure": k, "label": lbl, "top": f[top].mean(), "bottom": f[bot].mean(), "diff": diff, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(summary).to_csv("divergence_predicts_losses.csv", index=False, float_format="%.6g") # summary table, persisted


if __name__ == "__main__":
    main()
