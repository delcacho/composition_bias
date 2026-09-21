"""
panel_oos_all_pairs.py -- Exhibit 3's frozen-book forecast comparison on EVERY pair of the monthly
commodity book (Online Appendix C).

Per pair and forecast date, the construction of panel_oos_benchmarks.py:
  PIC      : current weights through a 252-day sample asset covariance, w_i' S w_j
  ROLL252  : 252-day sample covariance of the two sleeves' as-run returns
  EWMA63   : exponentially weighted cross product of the as-run returns, halflife 63
  EWMA252  : same, halflife 252
  target   : covariance of the two CURRENT books' returns over the next 63 days, weights frozen
Per pair: RMSE ratio benchmark/PIC (>1 = holdings wins), the quarterly-subsample win share and
sign test, a Diebold-Mariano t (Newey-West bandwidth 62), the pair's average as-run correlation and
the pair-minimum phi(63) of Exhibit 6 (phi_persistence_intent.csv). Then the ordering of the ratio
on persistence (Spearman) across the ten pairs. A date enters a pair only when both sleeves hold a
book.

usage: python scripts/panel_oos_all_pairs.py <run> <stacked_weights.csv> [--start=2000-01-01]
       (the same positional arguments as allocator_multiasset.py; writes <run>/panel_oos_all_pairs.csv)
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import binomtest, norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocator_multiasset as A  # noqa: E402  (reads argv[1], argv[2] for the run and the weights)

LOOKBACK, FWD = 252, 63
START = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--start=")), "2000-01-01")
NAME = {"CSCarry": "carry", "TSMomentum": "trend", "CSMomentum": "mom.", "CSValue": "value", "LongOnlyEW": "basket"}


def dm_test(d, h=FWD):
    """panel_oos_benchmarks.dm_test verbatim: DM on a loss differential (positive = second model better)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d); m = d.mean()
    g0 = np.mean((d - m) ** 2); s = g0
    for k in range(1, h):
        w = 1 - k / h; s += 2 * w * np.mean((d[k:] - m) * (d[:-k] - m))
    se = np.sqrt(max(s, 1e-30) / n); t = m / se
    return t, 2 * (1 - norm.cdf(abs(t)))


def ewma_cross(x, y, hl):
    lam = 0.5 ** (1 / hl); out = np.empty(len(x)); s = 0.0
    for t in range(len(x)):
        s = lam * s + (1 - lam) * x[t] * y[t]; out[t] = s
    return out


def pair_forecasts(Rv, Wt, i, j, sleeve_ret, dates, start):
    """Per-date frozen-book target and the four forecasts for sleeves i, j. Returns a DataFrame."""
    ri, rj = sleeve_ret[:, i], sleeve_ret[:, j]
    ew63, ew252 = ewma_cross(ri, rj, 63), ewma_cross(ri, rj, 252)
    held = (np.abs(Wt[:, i]).sum(axis=1) > 0) & (np.abs(Wt[:, j]).sum(axis=1) > 0)
    held[0] = False                     # no as-run return on the first day (no book held the day before)
    rows = []
    for t in range(LOOKBACK, len(dates) - FWD):
        if dates[t] < start or not held[t - LOOKBACK:t + 1].all():
            continue
        wi, wj = Wt[t, i], Wt[t, j]
        Sig = np.cov(Rv[t - LOOKBACK:t].T)
        pic = float(wi @ Sig @ wj)
        roll = float(np.cov(ri[t - LOOKBACK:t], rj[t - LOOKBACK:t])[0, 1])
        Rf = Rv[t + 1:t + 1 + FWD]; real = float(np.cov(Rf @ wi, Rf @ wj)[0, 1])
        na, nb = np.linalg.norm(wi), np.linalg.norm(wj)
        cos = float(wi @ wj / (na * nb)) if na > 0 and nb > 0 else np.nan
        rows.append({"date": dates[t], "realized": real, "PIC": pic, "ROLL252": roll,
                     "EWMA63": ew63[t], "EWMA252": ew252[t], "cos_sim": cos})
    return pd.DataFrame(rows).set_index("date")


def score(df):
    """Exhibit 3's statistics for one pair's forecast frame."""
    out = {"n_dates": len(df), "n_quarters": len(df.iloc[::FWD])}
    e_pic = df.realized - df.PIC; out["rmse_pic"] = float(np.sqrt((e_pic ** 2).mean()))
    q = df.iloc[::FWD]; eq_pic = q.realized - q.PIC
    for m in ["ROLL252", "EWMA63", "EWMA252"]:
        e = df.realized - df[m]; out[f"ratio_{m}"] = float(np.sqrt((e ** 2).mean()) / out["rmse_pic"])
        out[f"win_{m}"] = float((e_pic.abs() < e.abs()).mean())
        out[f"dm_t_{m}"], out[f"dm_p_{m}"] = dm_test(e ** 2 - e_pic ** 2)
        eq = q.realized - q[m]; out[f"win_q_{m}"] = float((eq_pic.abs() < eq.abs()).mean())
        w_q = int((eq_pic.abs() < eq.abs()).sum()); l_q = int((eq_pic.abs() > eq.abs()).sum())
        out[f"sign_p_q_{m}"] = float(binomtest(w_q, w_q + l_q, 0.5).pvalue) if (w_q + l_q) else np.nan
    return out


def main():
    dates, sleeves, assets, Wt, Rv = A.load_commodity()
    dates = pd.DatetimeIndex(dates); start = pd.Timestamp(START)
    sleeve_ret = np.einsum("tsa,ta->ts", np.vstack([np.zeros((1,) + Wt.shape[1:]), Wt[:-1]]), Rv)  # as run: yesterday's book
    phi = pd.read_csv(f"{A.RUN}/phi_persistence_intent.csv").set_index("sleeve").phi63
    rows = []; frames = []
    for i, j in combinations(range(len(sleeves)), 2):
        si, sj = sleeves[i], sleeves[j]
        df = pair_forecasts(Rv, Wt, i, j, sleeve_ret, dates, start)
        if len(df) < 2 * FWD:
            print(f"{NAME[si]} | {NAME[sj]}: {len(df)} dates, skipped"); continue
        r = {"pair": f"{NAME[si]} | {NAME[sj]}", "sleeve_i": si, "sleeve_j": sj,
             "rho": float(np.corrcoef(sleeve_ret[df.index.map(dates.get_loc), i], sleeve_ret[df.index.map(dates.get_loc), j])[0, 1]),
             "phi63_min": float(min(phi[si], phi[sj])), "phi63_geo": float(np.sqrt(phi[si] * phi[sj]))}
        r.update(score(df)); rows.append(r); frames.append(df.assign(pair=r["pair"]))
        print(f"{r['pair']:16s} rho {r['rho']:+.2f} phi63min {r['phi63_min']:.2f} n {r['n_dates']:5d} q {r['n_quarters']:3d} | "
              f"ROLL/PIC {r['ratio_ROLL252']:.2f} win_q {r['win_q_ROLL252']:.0%} DM t {r['dm_t_ROLL252']:+.2f} | "
              f"EWMA63/PIC {r['ratio_EWMA63']:.2f} | EWMA252/PIC {r['ratio_EWMA252']:.2f}", flush=True)
    T = pd.DataFrame(rows).sort_values("phi63_min")
    out = f"{A.RUN}/panel_oos_all_pairs.csv"; T.to_csv(out, index=False, float_format="%.6g")
    n_win = int((T.ratio_ROLL252 > 1).sum())
    rs, ps = spearmanr(T.ratio_ROLL252, T.phi63_min)
    print(f"\nROLL252/PIC above 1.00 on {n_win} of {len(T)} pairs; Spearman(ratio, phi63 min) {rs:+.2f} (p {ps:.2f}); "
          f"ratio range {T.ratio_ROLL252.min():.2f} to {T.ratio_ROLL252.max():.2f}; pairs above 1.10: {int((T.ratio_ROLL252 > 1.1).sum())}")
    print("wrote", out)

    # ---- POOLED over all pairs (the promoted main exhibit ex:oos): RMSE, win vs PIC, DM (daily) and
    # one non-overlapping date per 63 within each pair (quarterly), sign test ----
    allp = pd.concat(frames); qq = pd.concat([g.iloc[::FWD] for _, g in allp.groupby("pair")])
    ePIC = allp.realized - allp.PIC; ePICq = qq.realized - qq.PIC
    print(f"\nPOOLED over {len(T)} pairs: {len(allp)} pair-dates, {allp.index.nunique()} dates "
          f"{allp.index.min().date()}..{allp.index.max().date()}, {len(qq)} pair-quarters")
    prows = [{"model": "PIC", "rmse": float(np.sqrt((ePIC ** 2).mean())), "rmse_q": float(np.sqrt((ePICq ** 2).mean()))}]
    print(f"{'model':8s} {'RMSE':>11s} {'win':>7s} {'DM t':>7s} {'p':>7s} | quarterly RMSE  win  sign p")
    print(f"{'PIC':8s} {prows[0]['rmse']:11.3e} {'':>7s} {'':>7s} {'':>7s} | {prows[0]['rmse_q']:11.3e}")
    for m in ["ROLL252", "EWMA63", "EWMA252"]:
        e = allp.realized - allp[m]; rmse = float(np.sqrt((e ** 2).mean()))
        eq = qq.realized - qq[m]; rmse_q = float(np.sqrt((eq ** 2).mean()))
        win = float((ePIC.abs() < e.abs()).mean()); t, p = dm_test((e ** 2 - ePIC ** 2).values)
        w_q = int((ePICq.abs() < eq.abs()).sum()); l_q = int((ePICq.abs() > eq.abs()).sum())
        sign_p = float(binomtest(w_q, w_q + l_q, 0.5).pvalue) if (w_q + l_q) else float("nan")
        print(f"{m:8s} {rmse:11.3e} {win:7.1%} {t:7.2f} {p:7.3f} | {rmse_q:11.3e} {w_q/(w_q+l_q):4.0%} {sign_p:6.3f}")
        prows.append({"model": m, "rmse": rmse, "win": win, "dm_t": t, "dm_p": p, "rmse_q": rmse_q,
                      "win_q": w_q / (w_q + l_q), "sign_p_q": sign_p})
    prows[0].update({"n": len(allp), "n_q": len(qq)})
    pd.DataFrame(prows).to_csv(f"{A.RUN}/panel_oos_pooled.csv", index=False, float_format="%.6g")

    # ---- diagnostic 1: Newey-West bandwidth sweep on the pooled daily loss differential ----
    print("\nDM bandwidth sweep (pooled daily), t (p):")
    for m in ["ROLL252", "EWMA63", "EWMA252"]:
        d = ((allp.realized - allp[m]) ** 2 - ePIC ** 2).values
        cells = []
        for bw in (31, 62, 126, 252):
            t, p = dm_test(d, h=bw + 1)
            cells.append(f"bw{bw}: {t:.1f} ({p:.3f})")
        print(f"  {m:8s} " + "  ".join(cells))

    # ---- diagnostic 2: where the advantage sits -- distance of overlap from its trailing average ----
    # dev is built per pair frame and concatenated in the SAME order as allp, so it aligns positionally
    dev = np.concatenate([(df["cos_sim"] - df["cos_sim"].rolling(252, min_periods=60).mean()).abs().values
                          for df in frames])
    real = allp.realized.values; pic = allp.PIC.values; cos = allp.cos_sim.values; ePIC2 = (real - pic) ** 2
    ok = np.isfinite(dev); thr = np.nanquantile(dev, 0.80); top = dev >= thr
    print("\nadvantage concentration (top-quintile |overlap - trailing average| share, and corr of daily "
          "advantage with that distance and with the overlap level):")
    for m in ["ROLL252", "EWMA63", "EWMA252"]:
        adv = (real - allp[m].values) ** 2 - ePIC2
        v, tt, dv, cs = adv[ok], top[ok], dev[ok], cos[ok]
        print(f"  {m:8s} top-quintile share {v[tt].sum()/v.sum():.0%} | corr(adv, distance) {np.corrcoef(v, dv)[0,1]:+.2f} | "
              f"corr(adv, overlap) {np.corrcoef(v, cs)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
