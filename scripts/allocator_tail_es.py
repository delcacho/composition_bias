"""allocator_tail_es.py -- expected shortfall of the forward drawdown for the holdings blend and the return
stream books (Exhibit ex:mdd_es). A max drawdown is a tail statistic, so it is judged by the shape of the
loss distribution, not a per-window significance test: the mean of the worst q of forward-drawdown outcomes
is reported for each book at q in {20,10,5,2}%. The blend's tail is shallower at every depth and the
protection deepens toward the extreme, where the stream's stale correlation over-concentrates its risk-parity
book in a fast correlation spike. Reads allocator_multiasset.py's phasedaily parquet (phase 0).

usage: python research/allocator_tail_es.py [parquet] [--h=63]
"""
import sys
import numpy as np
import pandas as pd

args = sys.argv[1:]
SRC = next((a for a in args if not a.startswith("--")), "allocator_multiasset_phasedaily_L34_reb21.parquet")
H = next((int(a.split("=")[1]) for a in args if a.startswith("--h=")), 63)

d = pd.read_parquet(SRC)
d = d[d["phase"] == 0]
b = d["c2_daily"].to_numpy(); s = d["stream"].to_numpy()


def fwd_maxdd(r):
    out = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        c = np.cumprod(1.0 + r[t + 1:t + 1 + H])
        out[t] = (c / np.maximum.accumulate(c) - 1.0).min()
    return out


bdd = fwd_maxdd(b); sdd = fwd_maxdd(s)
m = np.isfinite(bdd) & np.isfinite(sdd)
bdd, sdd = bdd[m], sdd[m]


def es(x, q):
    t = np.sort(x)
    return t[:max(1, int(len(t) * q))].mean()          # mean of the worst q fraction (most negative)


rows = []
print(f"file: {SRC}   forecast dates {m.sum()}   horizon {H}d")
print(f"{'worst':>8s} {'stream ES':>10s} {'blend ES':>10s} {'protection (pts)':>17s}")
for q in [0.20, 0.10, 0.05, 0.02]:
    se, be = es(sdd, q), es(bdd, q)
    prot = (be - se) * 100                              # points shallower (positive = blend protects)
    print(f"{q:>7.0%} {se*100:>9.1f}% {be*100:>9.1f}% {prot:>16.2f}")
    rows.append({"tail_q": q, "stream_es": se, "blend_es": be, "protection_pts": prot})
out = SRC.replace(".parquet", "_tail_es.csv")
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5f")
print("wrote", out)
