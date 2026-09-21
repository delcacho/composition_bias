"""
allocator_mdd_inference.py -- inference on the drawdown advantage of the holdings-based books
over the return-stream book (the online appendix's inference exhibit).

Paired circular block bootstrap over time, per rebalance offset. For each offset and book the
same day-block indices are applied to the book and to the stream, so the comparison stays
paired; max drawdown is recomputed on each resampled path and the 95% percentile interval of
the difference (positive = book shallower than the stream) is reported. A second statistic does
the same for the mean daily advantage on days the stream book is below its running peak.

Offsets are overlapping calendars, not independent draws, so nothing is pooled across them: the
output is the observed median over offsets, the interval of the median offset, and the fraction
of offsets whose interval excludes zero.

usage: python scripts/allocator_mdd_inference.py [parquet] [--nboot=1000] [--blocks=63,126] [--seed=7]
"""
import sys
import numpy as np
import pandas as pd

BOOKS = ["c2_daily", "pic_daily", "invvol"]     # c2 = the recommended row; invvol = the no-correlation comparison


def max_dd(r):
    c = np.cumprod(1.0 + r); return float((c / np.maximum.accumulate(c) - 1.0).min())


def underwater_adv(h, g):
    c = np.cumprod(1.0 + g); uw = c < np.maximum.accumulate(c)
    return float(np.mean(h[uw] - g[uw])) if uw.any() else np.nan


def block_indices(n, block, rng):
    """Circular block bootstrap indices: ceil(n/block) random starts, consecutive blocks, wrapped."""
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n, nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def boot(h, g, block, nboot, rng):
    """Bootstrap distributions of (MDD_h - MDD_g) and of the underwater advantage, paired."""
    n = len(h); dd = np.empty(nboot); ua = np.empty(nboot)
    for b in range(nboot):
        idx = block_indices(n, block, rng); hh, gg = h[idx], g[idx]
        dd[b] = max_dd(hh) - max_dd(gg); ua[b] = underwater_adv(hh, gg)
    return dd, ua


def infer(df, books=BOOKS, blocks=(63, 126), nboot=1000, seed=7):
    """df: columns phase, stream, <books>; index date repeated per offset. Returns (aggregate, by-offset)."""
    rng = np.random.default_rng(seed); phases = sorted(df["phase"].unique())
    rows = []
    for nm in books:
        for block in blocks:
            for ph in phases:
                d = df[df["phase"] == ph]; g = d["stream"].values; h = d[nm].values
                obs_dd = max_dd(h) - max_dd(g); obs_ua = underwater_adv(h, g)
                dd, ua = boot(h, g, block, nboot, rng)
                lo_dd, hi_dd = np.percentile(dd, [2.5, 97.5]); lo_ua, hi_ua = np.nanpercentile(ua, [2.5, 97.5])
                rows.append({"book": nm, "block": block, "phase": ph,
                             "mdd_diff": obs_dd, "mdd_ci_lo": lo_dd, "mdd_ci_hi": hi_dd,
                             "mdd_excl0": bool(lo_dd > 0 or hi_dd < 0),
                             "uw_adv": obs_ua, "uw_ci_lo": lo_ua, "uw_ci_hi": hi_ua,
                             "uw_excl0": bool(lo_ua > 0 or hi_ua < 0)})
    B = pd.DataFrame(rows); agg = []
    for (nm, block), g in B.groupby(["book", "block"]):
        med = g.iloc[(g["mdd_diff"] - g["mdd_diff"].median()).abs().argsort().iloc[0]]   # the median offset
        medu = g.iloc[(g["uw_adv"] - g["uw_adv"].median()).abs().argsort().iloc[0]]
        agg.append({"book": nm, "block": block,
                    "mdd_diff_median": g["mdd_diff"].median(),
                    "mdd_ci_lo_median_offset": med["mdd_ci_lo"], "mdd_ci_hi_median_offset": med["mdd_ci_hi"],
                    "mdd_frac_offsets_excl0": g["mdd_excl0"].mean(),
                    "uw_adv_median_bp": g["uw_adv"].median() * 1e4,
                    "uw_ci_lo_median_offset_bp": medu["uw_ci_lo"] * 1e4, "uw_ci_hi_median_offset_bp": medu["uw_ci_hi"] * 1e4,
                    "uw_frac_offsets_excl0": g["uw_excl0"].mean()})
    return pd.DataFrame(agg), B


def main():
    args = sys.argv[1:]
    src = next((a for a in args if not a.startswith("--")), "allocator_multiasset_phasedaily_L34_reb21.parquet")
    nboot = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--nboot=")), 1000)
    blocks = tuple(int(x) for x in next((a.split("=", 1)[1] for a in args if a.startswith("--blocks=")), "63,126").split(","))
    seed = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--seed=")), 7)
    df = pd.read_parquet(src)
    civ = df["composition_intensity"]
    lo, hi = civ[civ.notna()].index.min(), civ[civ.notna()].index.max()
    df = df[(df.index >= lo) & (df.index <= hi)].copy()
    agg, byoff = infer(df, blocks=blocks, nboot=nboot, seed=seed)
    print(f"file: {src}   offsets: {df['phase'].nunique()}   resamples: {nboot}   blocks: {blocks}\n")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nDecision rule (c2_daily, block 63): pass if the max-drawdown interval of the median offset excludes 0 and "
          "mdd_frac_offsets_excl0 >= 0.70, and block 126 does not reverse it; the same rule on the underwater advantage.")
    agg.to_csv("allocator_mdd_inference.csv", index=False)
    byoff.to_csv("allocator_mdd_inference_byoffset.csv", index=False)


if __name__ == "__main__":
    main()
