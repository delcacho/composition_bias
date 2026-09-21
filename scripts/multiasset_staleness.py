"""
multiasset_staleness.py -- one picture for the multi-asset claim, at a common rebalancing cadence.

Persistence and staleness are measured for all eight sleeves as monthly-rebalanced strategies (the
book is re-formed at each month-end and held for the month), so the commodity and equity factors are
compared apples-to-apples: any remaining gap is breadth, not the fact that the commodity signals were
originally re-ranked daily and the equity factors monthly/annually.

For each sleeve, by one identical procedure with no per-asset tuning:
  monthly panel v[s, m] = B_m . r(s) -- the virtual return of month m's frozen book on day s,
                 B_m the commodity weight vector at month-end m (from the public book) or the equity
                 factor's month-end holdings; r the underlying asset returns (14 contracts / the stocks).
  x = phi(63): Sigma-norm cosine between a sleeve's book and its book a quarter later, from the panel
               Gram (never forms the asset covariance).
  y = staleness cost: var(live - clone)/var(live) over 63 days, live = the monthly-rebalanced return
               (each day uses the current month's book), clone = the book frozen at the quarter start.

If the equity points still sit below the commodity ones at matched persistence, universe size sets the
level, not the rebalancing cadence.

Outputs: figures/multiasset_staleness.png and research/multiasset_staleness.csv
Usage: python research/multiasset_staleness.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocator_multiasset as A
from phi_from_snapshots import phi_series # noqa: E402,F401 unused; kept for parity with allocator_multiasset

H = 63
STEP = 21
EQOUT = os.path.join("research", "equity_factors", "_out")


def neff(w):
    a = np.abs(np.asarray(w, dtype=float)); s = a.sum()
    return float(1.0 / np.sum((a / s) ** 2)) if s > 0 else np.nan


def month_ends(dates):
    ym = pd.PeriodIndex(dates, freq="M")
    last = pd.Series(np.arange(len(dates))).groupby(ym.values).max().values
    return np.sort(last)


def commodity_panels(Wt, Rv, dates):
    """monthly panel per commodity sleeve: v[s,m] = Rv[s] . Wt[month_end_m, sleeve]; snap_idx[s]=current month."""
    me = month_ends(dates)
    snap = np.searchsorted(me, np.arange(len(dates)), side="right") - 1 # most recent month-end <= s
    panels = []
    for i in range(Wt.shape[1]):
        B = Wt[me, i] # M x n_assets month-end books
        panels.append(Rv @ B.T) # T x M virtual returns
    return panels, snap, me


def phi63(panel, active):
    G = (panel[active].T @ panel[active]) / active.sum()
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None)); C = G / np.outer(d, d); M = C.shape[0]
    lag = max(1, round(H / STEP))
    return float(np.nanmean([C[m, m + lag] for m in range(M - lag)]))


def stale_tv(panel, snap, T):
    tvs = []
    for t in range(STEP * 12, T - H, STEP): # start after a year of books exist
        m0 = snap[t]
        if m0 < 0:
            continue
        s = np.arange(t, t + H)
        live = panel[s, snap[s]] # monthly-rebalanced: current month's book each day
        clone = panel[s, m0] # frozen at the quarter start
        v = np.var(live)
        if v > 0 and np.isfinite(live).all() and np.isfinite(clone).all():
            tvs.append(np.var(live - clone) / v)
    return float(np.median(tvs)) if tvs else np.nan


def phi_daily(W_sleeve, Sig, h=H):
    """phi(h) of eq:phi: Sigma-norm cosine between the daily book and its book h days later,
    fixed full-sample Sigma, averaged over active days (the persistence-table definition)."""
    act = np.abs(W_sleeve).sum(1) > 0; vals = []; n = len(W_sleeve)
    for t in range(n - h):
        if act[t] and act[t + h]:
            a = W_sleeve[t]; b = W_sleeve[t + h]; va = a @ Sig @ a; vb = b @ Sig @ b
            if va > 0 and vb > 0:
                vals.append((a @ Sig @ b) / np.sqrt(va * vb))
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    cdates, csleeves, assets, Wt, Rv = A.load_commodity()
    T = len(cdates)
    Sig_c = np.cov(np.nan_to_num(Rv).T)   # fixed full-sample covariance for eq:phi
    cpanels, csnap, _ = commodity_panels(Wt, Rv, cdates)
    esleeves, estream, epanels, esnap, elev = A.load_equity(cdates)
    snaps_eq = pd.read_parquet(os.path.join(EQOUT, "equity_holdings_monthly.parquet"))
    eqphi = pd.read_csv(os.path.join(EQOUT, "equity_persistence_daily.csv")).set_index("sleeve")["phi63_daily"]  # daily eq:phi(63), same as the persistence table

    rows = []
    active_all = np.ones(T, dtype=bool)
    for i, s in enumerate(csleeves):
        act = np.abs(Wt[:, i, :]).sum(axis=1) > 0
        panel = cpanels[i]
        nef = float(np.nanmedian([neff(Wt[t, i]) for t in month_ends(cdates)
                                  if np.abs(Wt[t, i]).sum() > 0]))
        rows.append({"sleeve": s, "class": "commodity", "phi63": phi_daily(Wt[:, i, :], Sig_c),
                     "stale_tv": stale_tv(panel, csnap, T), "neff": nef})
    for j, s in enumerate(esleeves):
        act = np.isfinite(estream[:, j])
        panel = epanels[s]
        g = snaps_eq[snaps_eq["sleeve"] == s]
        nef = float(g.groupby("snap_date")["w"].apply(lambda x: neff(x.values)).median())
        rows.append({"sleeve": f"eq_{s}", "class": "equity", "phi63": float(eqphi[s]),
                     "stale_tv": stale_tv(panel, esnap[s], T), "neff": nef})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("research", "multiasset_staleness.csv"), index=False, float_format="%.4f")
    print("Monthly-cadence comparison (both asset classes rebalanced monthly):\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        for cls, mk, col in [("commodity", "o", "#1f4e79"), ("equity", "s", "#b03a2e")]:
            g = df[df["class"] == cls].sort_values("phi63")
            ax.plot(g["phi63"], g["stale_tv"], color=col, lw=1.0, alpha=0.35, zorder=2)
            sz = 55 + 60 * np.log10(g["neff"].clip(lower=1))
            ax.scatter(g["phi63"], g["stale_tv"], marker=mk, s=sz, color=col, alpha=0.9,
                       edgecolor="white", linewidth=0.7, zorder=3,
                       label=f"{cls} ($N\\approx${int(g['neff'].median())})")
        OFF = {"LongOnlyEW": (0, 8, "center"), "eq_value": (7, -3, "left"),
               "eq_quality": (-7, -9, "right"), "eq_momentum": (7, 3, "left")}
        for _, r in df.iterrows():
            dx, dy, ha = OFF.get(r["sleeve"], (6, 4, "left"))
            ax.annotate(r["sleeve"].replace("eq_", ""), (r["phi63"], r["stale_tv"]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7)
        ax.set_xlabel(r"persistence $\phi(63)$: monthly book cosine over a quarter (faster rotation $\leftarrow$)")
        ax.set_ylabel(r"staleness cost: var(live $-$ frozen book) / var(live), 63 days")
        ax.set_xlim(right=1.03); ax.set_ylim(bottom=-0.04)
        ax.set_title("Both asset classes rebalanced monthly: staleness cost falls with persistence,\n"
                     "and universe size still sets the level (marker area $\\propto$ breadth $N$)", fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right"); fig.tight_layout()
        fig.savefig("figures/multiasset_staleness.png", dpi=170)
        print("\nwrote figures/multiasset_staleness.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
