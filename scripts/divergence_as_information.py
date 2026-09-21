"""
divergence_as_information.py -- is the holdings-minus-stream gap informative about forward risk?

Carry x momentum pair on the 14-commodity public book (panel_oos_benchmarks.load_signals),
1990 start. At each date t, from information through t:
    rho_PIC = correlation of the two current books through the 252-day asset covariance
    rho_str = 252-day rolling correlation of the two sleeves' realized returns
    GAP = rho_PIC - rho_str (scale-free headline; covariance gap alongside)
    cos = cosine similarity of the two books; crowded = cos > 0.80 (the paper's cut)
Forward, weights frozen at t (Exhibit 4's target): ERR = realized fwd corr (63d) - rho_str.
Forward, sleeves as run (joint risk, either sign; a converged pair moves together, and which way
is a coin): JM21 = |the pair's mean 21-day return| in its unconditional top decile;
PV63 = realized vol of the equal-weight pair over the next 63 days.

Tests: T1 ERR on GAP (Newey-West, lag 62) + GAP-quintile table; T2 top-minus-bottom quintile in
P(JM21), PV63 and forward correlation, 63-day block bootstrap; T3 GAP inside vs outside crowding.
Writes divergence_as_information.csv and figures/divergence_gap.png (one axis: the gap over
time, crowding windows shaded, forward joint tail losses as a rug).

usage: python research/divergence_as_information.py [start] [end]
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("research"))
from panel_oos_benchmarks import load_signals, gross_norm, SYMBOLS # noqa: E402

_POS = [a for a in sys.argv[1:] if not a.startswith("--")] # flags (--monthly) are not dates
START = _POS[0] if _POS else "1990-01-01"
END = _POS[1] if len(_POS) > 1 else "2025-12-31"
LOOKBACK, FWD = 252, 63
TAIL_PCT = 90 # |pair 21-day return| above this percentile = a large joint move, either sign
CROWD = 0.80
SEED, NBOOT, BLOCK = 7, 2000, 63


def nw_ols(x, y, lag):
    """y = a + b x, Newey-West HAC (Bartlett, `lag`). Returns b, se_b, t, p (normal)."""
    from scipy.stats import norm
    X = np.column_stack([np.ones(len(x)), x]); XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y; e = y - X @ beta; Xe = X * e[:, None]; S = Xe.T @ Xe
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); G = Xe[l:].T @ Xe[:-l]; S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv; se = np.sqrt(np.diag(V))
    t = beta[1] / se[1]; return beta[1], se[1], t, 2 * (1 - norm.cdf(abs(t)))


def block_boot_diff(flag, top, bot, rng, nboot=NBOOT, block=BLOCK):
    """top-minus-bottom frequency of `flag`, resampling 63-day blocks of dates (labels fixed)."""
    n = len(flag); nb = int(np.ceil(n / block)); out = []
    for _ in range(nboot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        f, tp, bt = flag[idx], top[idx], bot[idx]
        if tp.sum() and bt.sum():
            out.append(f[tp].mean() - f[bt].mean())
    return np.quantile(out, [0.05, 0.95])


def main():
    sig = {s: load_signals(s, START, END) for s in SYMBOLS}
    R = pd.concat({s: sig[s]["daily_ret"] for s in SYMBOLS}, axis=1).dropna()
    Wc = gross_norm(pd.concat({s: sig[s]["carry_vt"] for s in SYMBOLS}, axis=1).reindex(R.index)).dropna()
    Wm = gross_norm(pd.concat({s: sig[s]["mom_vt"] for s in SYMBOLS}, axis=1).reindex(R.index)).dropna()
    idx = R.index.intersection(Wc.index).intersection(Wm.index); R, Wc, Wm = R.loc[idx], Wc.loc[idx], Wm.loc[idx]
    if "--monthly" in sys.argv: # rebalance monthly: hold each month-end book
        me = pd.PeriodIndex(idx, freq="M"); is_me = np.array([me[i] != me[i + 1] for i in range(len(idx) - 1)] + [True])
        for Wx in (Wc, Wm):
            last = None
            for t in range(len(idx)):
                if is_me[t] or last is None: last = Wx.iloc[t].copy()
                Wx.iloc[t] = last
    rc = (Wc.shift(1) * R).sum(axis=1).fillna(0.0).values; rm = (Wm.shift(1) * R).sum(axis=1).fillna(0.0).values
    Rv, T = R.values, len(idx)
    print(f"dates {idx[0].date()}..{idx[-1].date()} T={T}")

    rows = []
    for t in range(LOOKBACK, T - FWD):
        wc, wm = Wc.iloc[t].values, Wm.iloc[t].values
        Sig = np.cov(Rv[t - LOOKBACK:t].T)
        pic_cov = float(wc @ Sig @ wm); pic_corr = pic_cov / np.sqrt(float(wc @ Sig @ wc) * float(wm @ Sig @ wm))
        a, b = rc[t - LOOKBACK:t], rm[t - LOOKBACK:t]
        roll_cov = float(np.cov(a, b)[0, 1]); roll_corr = float(np.corrcoef(a, b)[0, 1])
        cos = float(wc @ wm / (np.linalg.norm(wc) * np.linalg.norm(wm)))
        Rf = Rv[t + 1:t + 1 + FWD]; fc, fm = Rf @ wc, Rf @ wm
        real_cov = float(np.cov(fc, fm)[0, 1]); real_corr = float(np.corrcoef(fc, fm)[0, 1])
        row = dict(date=idx[t], pic_corr=pic_corr, roll_corr=roll_corr, gap=pic_corr - roll_corr,
                   pic_cov=pic_cov, roll_cov=roll_cov, gap_cov=pic_cov - roll_cov, cos=cos,
                   real_corr=real_corr, err=real_corr - roll_corr, err_cov=real_cov - roll_cov)
        pair = (rc + rm) / 2
        row["pair21"] = float(pair[t + 1:t + 22].sum())
        row["pv63"] = float(pair[t + 1:t + 1 + FWD].std(ddof=0) * np.sqrt(252))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    df["jm21"] = (df.pair21.abs() > np.percentile(df.pair21.abs(), TAIL_PCT)).astype(float)
    df["crowded"] = (df.cos > CROWD).astype(float)
    df["q"] = pd.qcut(df.gap, 5, labels=False)
    df.to_csv("divergence_as_information.csv", float_format="%.6g")
    print(f"forecast dates {len(df)} (~{len(df)//FWD} non-overlapping quarters); crowded dates {int(df.crowded.sum())}")

    # ---- T1: does the gap forecast the stream's forward error?
    print("\n=== T1: stream forward error on the gap (Newey-West, lag 62) ===")
    for lab, x, y in (("correlation", df.gap.values, df.err.values), ("covariance", df.gap_cov.values, df.err_cov.values)):
        b, se, t, p = nw_ols(x, y, FWD - 1)
        print(f" {lab:11s}: b = {b:+.3f} (se {se:.3f}) t = {t:+.2f} p = {p:.4f}")
    from scipy.stats import spearmanr
    print(f" Spearman(gap, err) = {spearmanr(df.gap, df.err)[0]:+.3f}")

    print("\n=== GAP quintiles: mean gap | mean fwd err | share err>0 | mean fwd corr | P(JM21) | mean fwd pair vol | mean cos | crowded share ===")
    for q, g in df.groupby("q"):
        print(f" Q{q+1}: {g.gap.mean():+.3f} | {g.err.mean():+.3f} | {(g.err > 0).mean():.2f} | {g.real_corr.mean():+.3f} | "
              f"{g.jm21.mean():.3f} | {g.pv63.mean():.3f} | {g.cos.mean():+.2f} | {g.crowded.mean():.2f}")
    uncond = {k: df[k].mean() for k in ("jm21", "pv63", "real_corr")}
    print(" unconditional: " + " ".join(f"{k} {v:.3f}" for k, v in uncond.items()))

    # ---- T2: joint losses, top minus bottom quintile, block bootstrap
    print("\n=== T2: forward joint risk, top minus bottom gap quintile (63-day block bootstrap 90% CI) ===")
    rng = np.random.default_rng(SEED); top = (df.q == 4).values; bot = (df.q == 0).values
    for k in ("jm21", "pv63", "real_corr"):
        f = df[k].values; d = f[top].mean() - f[bot].mean(); lo, hi = block_boot_diff(f, top, bot, rng)
        print(f" {k}: top {f[top].mean():.3f} bottom {f[bot].mean():.3f} diff {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}] {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")

    # ---- T3: mechanism, crowding windows
    print("\n=== T3: the gap inside vs outside crowding windows (cos > 0.80) ===")
    cr, nc = df[df.crowded == 1], df[df.crowded == 0]
    print(f" mean gap: crowded {cr.gap.mean():+.3f} (n={len(cr)}) vs not {nc.gap.mean():+.3f} (n={len(nc)})")
    print(f" share of crowded dates in the top gap quintile: {(cr.q == 4).mean():.2f} (top-quintile dates that are crowded: {(df[df.q == 4].crowded).mean():.2f})")
    print(f" fwd err crowded {cr.err.mean():+.3f} vs not {nc.err.mean():+.3f}; JM21 crowded {cr.jm21.mean():.3f} vs not {nc.jm21.mean():.3f}; fwd pair vol crowded {cr.pv63.mean():.3f} vs not {nc.pv63.mean():.3f}")

    # ---- figure: one axis, the gap over time; crowding shaded; forward tail losses as a rug
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(df.index, df.gap, color="#1f4e79", lw=0.5, alpha=0.35) # daily reading
        ax.plot(df.index, df.gap.rolling(21, min_periods=5).mean(), color="#1f4e79", lw=1.4) # 21-day mean of the same series
        ax.axhline(0.0, color="black", lw=1, ls="--")
        c = df.crowded.values; d = df.index
        i = 0
        while i < len(c): # contiguous crowding runs
            if c[i]:
                j = i
                while j + 1 < len(c) and c[j + 1]:
                    j += 1
                ax.axvspan(d[i], d[j], color="0.80", lw=0, zorder=0); i = j + 1
            else:
                i += 1
        lo = df.gap.min(); span = df.gap.max() - lo
        tail = df.index[df.jm21 == 1]
        ax.plot(tail, np.full(len(tail), lo - 0.04 * span), "|", color="#c0392b", ms=7, mew=1.0)
        ax.set_ylim(lo - 0.08 * span, df.gap.max() + 0.05 * span)
        ax.set_ylabel("holdings-implied minus stream correlation\ncarry x trend")
        ax.grid(alpha=0.3)
        ax.legend(handles=[Line2D([], [], color="#1f4e79", lw=0.5, alpha=0.35, label="gap, daily (information through the day)"),
                           Line2D([], [], color="#1f4e79", lw=1.4, label="gap, 21-day mean"),
                           Patch(color="0.80", label="crowding window (cosine > 0.80)"),
                           Line2D([], [], color="#c0392b", marker="|", ls="none", mew=1, ms=7, label="large joint 21-day move ahead, either sign")],
                  fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig("figures/divergence_gap.png", dpi=200)
        print("\nwrote figures/divergence_gap.png")
    except Exception as ex:
        print(f"\nfigure skipped: {ex}")
    print("wrote divergence_as_information.csv")


if __name__ == "__main__":
    main()
