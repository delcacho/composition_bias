"""
allocator_mdd_attribution.py -- attributes the drawdown and Sharpe advantage of the holdings-based
risk-parity books over the return-stream book to divergence days (the online appendix's drawdown
attribution exhibit).

Reads allocator_multiasset_phasedaily_L34_reb21.parquet, written by allocator_multiasset.py: the
daily return of each estimator's book for every rebalance offset, plus composition_intensity, the
mean absolute gap between the holdings correlation and the stream correlation over the active
sleeve pairs. A divergence day is one in the top quintile of that gap.

Max drawdown in the article is the median over 21 monthly rebalance offsets, so every metric here
is computed per offset and reported as the median across offsets, with the fraction of offsets on
which the book beats the stream on drawdown. The primary concentration metric pools the drawdown-day
advantage across all offsets. Offsets overlap and are not independent; the across-offset spread is
descriptive, not a significance test.

Writes allocator_mdd_attribution.csv (per-book aggregate) and allocator_mdd_attribution_byoffset.csv.

usage: python scripts/allocator_mdd_attribution.py [phasedaily_parquet]
"""
import sys
import numpy as np
import pandas as pd

HOLDINGS = ["pic_daily", "c2_daily", "blend_daily"]   # books whose advantage is attributed; invvol is the control
Q = 5                                                  # a divergence day is in the top quintile of composition_intensity


def sharpe(r):
    s = np.nanstd(r)
    return float(np.nanmean(r) / s * np.sqrt(252)) if s > 0 else np.nan


def max_dd(r):
    c = np.cumprod(1.0 + r); return float((c / np.maximum.accumulate(c) - 1.0).min())


def underwater(r):
    c = np.cumprod(1.0 + r); return (c < np.maximum.accumulate(c))     # the stream book is below its running peak


def attribute(df, books=HOLDINGS + ["invvol"]):
    """df: columns phase, stream, <books>, composition_intensity; index date, repeated per offset.
    Returns (per-book aggregate rows, per-offset rows)."""
    phases = sorted(df["phase"].unique())
    ci_by_date = df[df["phase"] == phases[0]]["composition_intensity"]
    thr = np.nanquantile(ci_by_date.values, 1 - 1.0 / Q)               # one global top-quintile threshold
    per_offset, agg = [], []
    for nm in books:
        rows = []
        pool_shallow_top, pool_shallow_all = 0.0, 0.0                  # pooled numerator and denominator
        loss_hi, loss_lo = [], []                                      # advantage on the stream's large-loss days
        for ph in phases:
            d = df[df["phase"] == ph]
            g = d["stream"].values; h = d[nm].values; ci = d["composition_intensity"].values
            a = h - g; top = np.isfinite(ci) & (ci >= thr)
            uw = underwater(g)
            sh_h, sh_s = sharpe(h), sharpe(g); lead = sh_h - sh_s
            surv = float((sharpe(np.where(top, g, h)) - sh_s) / lead) if abs(lead) > 1e-9 else np.nan
            m1 = float(a[uw & top].sum() / a[uw].sum()) if a[uw].sum() != 0 else np.nan
            m2 = float(a[top].sum() / a.sum()) if a.sum() != 0 else np.nan
            rows.append({"phase": ph, "mdd": max_dd(h), "mdd_minus_stream": max_dd(h) - max_dd(g),
                         "M1_dd_topq_share": m1, "M2_fullsample_topq_share": m2,
                         "sharpe_lead": lead, "M4_lead_surviving": surv})
            pool_shallow_top += a[uw & top].sum(); pool_shallow_all += a[uw].sum()
            worst = g <= np.nanquantile(g, 0.10)
            loss_hi.append(a[worst & top]); loss_lo.append(a[worst & ~top])
        R = pd.DataFrame(rows); R.insert(0, "book", nm); per_offset.append(R)
        beats = float((R["mdd_minus_stream"] > 0).mean())
        m1_pool = float(pool_shallow_top / pool_shallow_all) if pool_shallow_all != 0 else np.nan
        # direction against magnitude: on divergence days the two books differ most, so the absolute
        # return difference is largest by construction; the mean advantage must be positive there.
        d0 = df[df["phase"] == phases[0]]; g0 = d0["stream"].values; a0 = d0[nm].values - g0
        ci0 = d0["composition_intensity"].values; top0 = np.isfinite(ci0) & (ci0 >= thr); bot0 = np.isfinite(ci0) & ~top0
        adv_top, adv_bot = float(np.mean(a0[top0])), float(np.mean(a0[bot0]))
        absratio = float(np.mean(np.abs(a0[top0])) / np.mean(np.abs(a0[bot0]))) if np.mean(np.abs(a0[bot0])) else np.nan
        hi = np.concatenate(loss_hi); lo = np.concatenate(loss_lo)
        agg.append({"book": nm,
                    "mdd_median": R["mdd"].median(), "mdd_minus_stream_median": R["mdd_minus_stream"].median(),
                    "frac_offsets_beats_stream_mdd": beats,
                    "M1_pooled_dd_topq_share": m1_pool,
                    "M1_offset_median": R["M1_dd_topq_share"].median(),
                    "M2_offset_median": R["M2_fullsample_topq_share"].median(),
                    "M4_offset_median": R["M4_lead_surviving"].median(),
                    "interaction_loss_adv_hi_minus_lo": float(np.nanmean(hi) - np.nanmean(lo)),
                    "adv_mean_topci": adv_top, "adv_mean_botci": adv_bot,
                    "absadv_topci_over_botci": absratio,
                    "sharpe_lead_median": R["sharpe_lead"].median()})
    return pd.DataFrame(agg), pd.concat(per_offset, ignore_index=True)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "allocator_multiasset_phasedaily_L34_reb21.parquet"
    df = pd.read_parquet(src)
    # the index repeats dates across offsets, so the trim to the window where the signal is defined is a mask
    civ = df["composition_intensity"]
    lo, hi = civ[civ.notna()].index.min(), civ[civ.notna()].index.max()
    df = df[(df.index >= lo) & (df.index <= hi)].copy()
    agg, byoff = attribute(df)
    print(f"file: {src}   offsets: {df['phase'].nunique()}   divergence day = top quintile of composition_intensity\n")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    agg.to_csv("allocator_mdd_attribution.csv", index=False)
    byoff.to_csv("allocator_mdd_attribution_byoffset.csv", index=False)


if __name__ == "__main__":
    main()
