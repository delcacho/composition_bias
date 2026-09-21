"""
divergence_portfolio.py -- the allocator-facing version of divergence-as-information: one book-level
signal, not per pair. At each date, aggregate the disagreement between the whole book's holdings-implied
sleeve correlation matrix and its return-stream correlation matrix, and ask whether that aggregate gap
forecasts the whole book's forward joint risk. Book = equal-weight of the 5 commodity sleeves.

signal(t) = book-variance gap = w' R_H(t) w - w' R_R(t) w, with w the sleeves' stream-vol weights
(so it isolates the correlation disagreement); also reported: mean and max pairwise gap.
forward risk(t): PVbook = book's realized 63-day vol; JMbook = |book's next-21-day return| in top decile.
Top-minus-bottom signal quintile, 63-day block bootstrap. Same LOOKBACK/FWD as the pairwise script.

usage: python research/divergence_portfolio.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from allocator_multiasset import load_commodity

LOOKBACK, FWD, TAIL_PCT = 252, 63, 90
SEED, NBOOT, BLOCK = 7, 2000, 63


def block_boot_diff(flag, top, bot, rng):
    n = len(flag); nb = int(np.ceil(n / BLOCK)); out = []
    for _ in range(NBOOT):
        starts = rng.integers(0, n - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        f, tp, bt = flag[idx], top[idx], bot[idx]
        if tp.sum() and bt.sum():
            out.append(f[tp].mean() - f[bt].mean())
    return np.quantile(out, [0.05, 0.95])


def corr_from_cov(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None)); return C / np.outer(d, d)


def main():
    cdates, sleeves, assets, Wt, Rv = load_commodity()
    T, K = len(cdates), len(sleeves)
    rs = np.zeros((T, K))
    for i in range(K):
        rs[1:, i] = np.einsum("ta,ta->t", Wt[:-1, i, :], Rv[1:, :])
    book = rs.mean(axis=1) # equal-weight book return
    print(f"sleeves {sleeves} book = equal weight T={T}")

    rows = []
    for t in range(LOOKBACK, T - FWD):
        Sig = np.cov(Rv[t - LOOKBACK:t].T)
        Wc = Wt[t] # K x N current books
        RH = np.zeros((K, K)); vs = np.zeros(K)
        ok = True
        for i in range(K):
            vs[i] = float(Wc[i] @ Sig @ Wc[i])
            if vs[i] <= 0:
                ok = False; break
        if not ok:
            continue
        for i in range(K):
            for j in range(K):
                RH[i, j] = float(Wc[i] @ Sig @ Wc[j]) / np.sqrt(vs[i] * vs[j])
        Rstream = np.cov(rs[t - LOOKBACK:t].T)
        if np.any(np.diag(Rstream) < 1e-14):
            continue
        RR = corr_from_cov(Rstream)
        sv = rs[t - LOOKBACK:t].std(axis=0) # stream sleeve vols
        w = sv / sv.sum()
        gap_var = float(w @ RH @ w - w @ RR @ w)
        iu = np.triu_indices(K, 1)
        mean_gap = float((RH[iu] - RR[iu]).mean()); max_gap = float((RH[iu] - RR[iu]).max())
        b21 = float(book[t + 1:t + 22].sum())
        pv63 = float(book[t + 1:t + 1 + FWD].std(ddof=0) * np.sqrt(252))
        rows.append((gap_var, mean_gap, max_gap, b21, pv63))
    d = pd.DataFrame(rows, columns=["gap_var", "mean_gap", "max_gap", "b21", "pv63"])
    d["jm"] = (d.b21.abs() > np.percentile(d.b21.abs(), TAIL_PCT)).astype(float)
    print(f"forecast dates {len(d)} (~{len(d)//FWD} quarters)")

    rng = np.random.default_rng(SEED)
    for sig in ("gap_var", "mean_gap", "max_gap"):
        d["q"] = pd.qcut(d[sig], 5, labels=False)
        top, bot = (d.q == 4).values, (d.q == 0).values
        print(f"\n=== signal = {sig}: forward book joint risk, top-minus-bottom quintile ===")
        print(" quintile P(large book move) | fwd book vol:")
        for q, g in d.groupby("q"):
            print(f" Q{q+1}: JM {g.jm.mean():.3f} vol {g.pv63.mean():.3f}")
        for k in ("jm", "pv63"):
            f = d[k].values; diff = f[top].mean() - f[bot].mean(); lo, hi = block_boot_diff(f, top, bot, rng)
            print(f" {k}: top {f[top].mean():.3f} bottom {f[bot].mean():.3f} diff {diff:+.3f} "
                  f"CI [{lo:+.3f},{hi:+.3f}] {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")


if __name__ == "__main__":
    main()
