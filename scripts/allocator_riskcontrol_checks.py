"""allocator_riskcontrol_checks.py -- is the drawdown/under-water advantage of the holdings books over
the stream book a reduction in risk, or a return edge?

The under-water shortfall of allocator_mdd_inference.py is a MEAN RETURN on days the stream is under water,
so this script separates the return reading from the risk reading under the SAME paired circular-block
bootstrap over time, per rebalance offset (offsets are overlapping calendars, nothing pooled across them;
report the median offset, its interval, and the fraction of offsets whose interval excludes zero), positive
always meaning the holdings book is better:

  uncond   mean(h - g) on ALL days                          -- is the edge unconditional (a return edge)?
  uw_t0    mean(h - g) on days the stream is under water     -- the inference statistic, same-day state
  uw_tm1   mean(h - g) on days under water AS OF t-1         -- removes any same-day selection
  semidev  downside semideviation of g minus that of h       -- lower downside dispersion?
  cvar5    5% CVaR of g minus that of h                       -- thinner left tail?
  gapday   mean(h - g) on the top quintile of the covariance-gap state, composition_intensity, an exogenous
           state defined off returns (the divergence-day mechanism of the drawdown attribution)

usage: python allocator_riskcontrol_checks.py [parquet] [--nboot=1000] [--blocks=63,126] [--seed=7]
"""
import sys
import numpy as np
import pandas as pd

BOOKS = ["c2_daily", "pic_daily", "invvol"]   # exact set/order of allocator_mdd_inference.py, so the shared
# bootstrap draws match: c2 = the recommended blend (holdings correlation, stream vol); pic = holdings
# covariance; invvol = the no-correlation control. uw_t0 here reproduces ex:mdd_inference cell for cell.


def _semidev(r):
    d = np.minimum(r, 0.0); return float(np.sqrt(np.mean(d * d)))


def _cvar5(r):
    q = np.quantile(r, 0.05); tail = r[r <= q]
    return float(-tail.mean()) if tail.size else np.nan               # positive = larger tail loss


def metrics(h, g, s):
    """All six paired statistics for one path. s: the exogenous gap state aligned to h, g. Positive = h better."""
    c = np.cumprod(1.0 + g); uw = c < np.maximum.accumulate(c)
    uw_lag = np.concatenate(([False], uw[:-1]))                       # under water decided at t-1
    d = h - g
    thr = np.quantile(s, 0.80); gap = s >= thr
    return (float(d.mean()),
            float(d[uw].mean()) if uw.any() else np.nan,
            float(d[uw_lag].mean()) if uw_lag.any() else np.nan,
            _semidev(g) - _semidev(h),
            _cvar5(g) - _cvar5(h),
            float(d[gap].mean()) if gap.any() else np.nan)


NAMES = ["uncond", "uw_t0", "uw_tm1", "semidev", "cvar5", "gapday"]


def block_indices(n, block, rng):
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n, nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def infer(df, books=BOOKS, blocks=(63, 126), nboot=1000, seed=7):
    rng = np.random.default_rng(seed); phases = sorted(df["phase"].unique()); rows = []
    for nm in books:
        for block in blocks:
            for ph in phases:
                d = df[df["phase"] == ph]
                g = d["stream"].values; h = d[nm].values; s = d["composition_intensity"].values
                obs = metrics(h, g, s)
                n = len(h); boot = np.empty((nboot, len(NAMES)))
                for b in range(nboot):
                    idx = block_indices(n, block, rng)
                    boot[b] = metrics(h[idx], g[idx], s[idx])
                lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                row = {"book": nm, "block": block, "phase": ph}
                for k, nmk in enumerate(NAMES):
                    row[f"{nmk}"] = obs[k]; row[f"{nmk}_lo"] = lo[k]; row[f"{nmk}_hi"] = hi[k]
                    row[f"{nmk}_excl0"] = bool(lo[k] > 0 or hi[k] < 0)
                rows.append(row)
    B = pd.DataFrame(rows); agg = []
    for (nm, block), grp in B.groupby(["book", "block"]):
        rec = {"book": nm, "block": block}
        for nmk in NAMES:
            med = grp.iloc[(grp[nmk] - grp[nmk].median()).abs().argsort().iloc[0]]      # the median offset
            scale = 1e4 if nmk not in ("semidev", "cvar5") else 1e4
            rec[f"{nmk}_med_bp"] = grp[nmk].median() * scale
            rec[f"{nmk}_lo_bp"] = med[f"{nmk}_lo"] * scale
            rec[f"{nmk}_hi_bp"] = med[f"{nmk}_hi"] * scale
            rec[f"{nmk}_frac_excl0"] = grp[f"{nmk}_excl0"].mean()
        agg.append(rec)
    return pd.DataFrame(agg), B


def main():
    args = sys.argv[1:]
    src = next((a for a in args if not a.startswith("--")), "allocator_multiasset_phasedaily_L34_reb21.parquet")
    nboot = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--nboot=")), 1000)
    blocks = tuple(int(x) for x in next((a.split("=", 1)[1] for a in args if a.startswith("--blocks=")), "63,126").split(","))
    seed = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--seed=")), 7)
    df = pd.read_parquet(src)
    civ = df["composition_intensity"]; lo, hi = civ[civ.notna()].index.min(), civ[civ.notna()].index.max()
    df = df[(df.index >= lo) & (df.index <= hi)].copy()
    agg, byoff = infer(df, blocks=blocks, nboot=nboot, seed=seed)
    pd.set_option("display.width", 200, "display.max_columns", 60)
    print(f"file: {src}   offsets: {df['phase'].nunique()}   resamples: {nboot}   blocks: {blocks}")
    print("units: bp/day for return diffs (uncond/uw_t0/uw_tm1/gapday); semidev/cvar5 diffs also x1e4. positive = holdings book better.\n")
    for block in blocks:
        print(f"--- block {block} ---")
        sub = agg[agg.block == block]
        for _, r in sub.iterrows():
            print(f"{r['book']:>11}: " + "  ".join(
                f"{nmk}={r[f'{nmk}_med_bp']:+.2f}[{r[f'{nmk}_lo_bp']:+.2f},{r[f'{nmk}_hi_bp']:+.2f}]f{r[f'{nmk}_frac_excl0']:.2f}"
                for nmk in NAMES))
        print()
    agg.to_csv("allocator_riskcontrol_checks.csv", index=False)
    print("wrote allocator_riskcontrol_checks.csv")


if __name__ == "__main__":
    main()
