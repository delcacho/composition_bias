"""
divergence_across_pairs.py -- generalize the divergence-as-information result (carry x momentum,
divergence_as_information.py) to all ten commodity sleeve pairs of the public book. Same construction:
per date, gap = holdings-implied correlation (current books through the 252-day asset covariance) minus
the 252-day rolling return correlation; forward joint risk = P(large 21-day pair move, either sign) and
the pair's forward 63-day vol; top-minus-bottom gap quintile with a 63-day block bootstrap.

usage: python research/divergence_across_pairs.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)))) # research/
from allocator_multiasset import load_commodity

LOOKBACK, FWD, TAIL_PCT = 252, 63, 90
SEED, NBOOT, BLOCK = 7, 2000, 63


def block_boot_diff(flag, top, bot, rng, nboot=NBOOT, block=BLOCK):
    n = len(flag); nb = int(np.ceil(n / block)); out = []
    for _ in range(nboot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        f, tp, bt = flag[idx], top[idx], bot[idx]
        if tp.sum() and bt.sum():
            out.append(f[tp].mean() - f[bt].mean())
    return np.quantile(out, [0.05, 0.95])


def main():
    cdates, sleeves, assets, Wt, Rv = load_commodity()
    T, K = len(cdates), len(sleeves)
    print(f"sleeves {sleeves} dates {cdates[0].date()}..{cdates[-1].date()} T={T}")
    # sleeve return series: weights decided at t-1, applied to t
    rs = np.zeros((T, K))
    for i in range(K):
        rs[1:, i] = np.einsum("ta,ta->t", Wt[:-1, i, :], Rv[1:, :])

    # shared per-date asset covariance
    Sig = [None] * T
    for t in range(LOOKBACK, T - FWD):
        Sig[t] = np.cov(Rv[t - LOOKBACK:t].T)

    rng = np.random.default_rng(SEED)
    results = []
    for i in range(K):
        for j in range(i + 1, K):
            rows = []
            for t in range(LOOKBACK, T - FWD):
                wi, wj = Wt[t, i, :], Wt[t, j, :]
                S = Sig[t]
                vi, vj = float(wi @ S @ wi), float(wj @ S @ wj)
                if vi <= 0 or vj <= 0:
                    continue
                pic = float(wi @ S @ wj) / np.sqrt(vi * vj)
                a, b = rs[t - LOOKBACK:t, i], rs[t - LOOKBACK:t, j]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    continue
                roll = float(np.corrcoef(a, b)[0, 1])
                pair = (rs[:, i] + rs[:, j]) / 2
                p21 = float(pair[t + 1:t + 22].sum())
                pv63 = float(pair[t + 1:t + 1 + FWD].std(ddof=0) * np.sqrt(252))
                fi, fj = Rv[t + 1:t + 1 + FWD] @ wi, Rv[t + 1:t + 1 + FWD] @ wj
                rcorr = float(np.corrcoef(fi, fj)[0, 1]) if fi.std() > 0 and fj.std() > 0 else np.nan
                rows.append((pic - roll, p21, pv63, rcorr))
            if len(rows) < 500:
                continue
            d = pd.DataFrame(rows, columns=["gap", "p21", "pv63", "rcorr"])
            d["jm21"] = (d.p21.abs() > np.percentile(d.p21.abs(), TAIL_PCT)).astype(float)
            d["q"] = pd.qcut(d.gap, 5, labels=False)
            top, bot = (d.q == 4).values, (d.q == 0).values
            mean_rho = float(d.rcorr.mean())
            out = {"pair": f"{sleeves[i]} x {sleeves[j]}", "n": len(d), "mean_rho": mean_rho}
            for k in ("jm21", "pv63"):
                f = d[k].values
                diff = f[top].mean() - f[bot].mean()
                lo, hi = block_boot_diff(f, top, bot, rng)
                out[k] = (f[top].mean(), f[bot].mean(), diff, lo, hi, (lo > 0 or hi < 0))
            results.append(out)

    # Exhibit A7, persisted: one row per pair with both lifts and their 90% block-bootstrap intervals
    pd.DataFrame([{"pair": o["pair"], "n": o["n"], "mean_rho": o["mean_rho"],
                   **{f"{k}_{f}": v for k in ("jm21", "pv63") for f, v in zip(("top", "bot", "diff", "lo", "hi", "sig"), o[k])}}
                  for o in results]).to_csv("divergence_across_pairs.csv", index=False, float_format="%.6g")
    print("\n=== top-minus-bottom gap quintile, per pair (63-day block bootstrap 90% CI) ===")
    print(f"{'pair':28s} {'rho':>6s} {'JM21 top':>8s} {'bot':>6s} {'diff':>7s} {'CI':>18s} {'PV63 diff':>9s} {'CI':>18s}")
    for o in sorted(results, key=lambda x: -abs(x["mean_rho"])):
        jt, jb, jd, jlo, jhi, jsig = o["jm21"]
        _, _, pd_, plo, phi, psig = o["pv63"]
        print(f"{o['pair']:28s} {o['mean_rho']:+.2f} {jt:8.3f} {jb:6.3f} {jd:+7.3f} "
              f"[{jlo:+.3f},{jhi:+.3f}]{'*' if jsig else ' '} {pd_:+9.3f} [{plo:+.3f},{phi:+.3f}]{'*' if psig else ' '}")

    mat = [o for o in results if abs(o["mean_rho"]) > 0.15]
    jm_diffs = [o["jm21"][2] for o in mat]
    print(f"\nmaterial-correlation pairs (|rho|>0.15): {len(mat)} of {len(results)}")
    print(f" JM21 top>bottom (positive diff) on {sum(dd > 0 for dd in jm_diffs)} of {len(mat)}; "
          f"CI excludes 0 on {sum(o['jm21'][5] for o in mat)}")
    print(f" median JM21 diff on material pairs: {np.median(jm_diffs):+.3f}")
    all_jm = [o["jm21"][2] for o in results]
    print(f" all {len(results)} pairs: positive on {sum(dd > 0 for dd in all_jm)}, median {np.median(all_jm):+.3f}")


if __name__ == "__main__":
    main()
