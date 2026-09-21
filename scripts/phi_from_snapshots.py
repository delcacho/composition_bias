"""
phi_from_snapshots.py -- can an allocator recover mix persistence phi(h) from periodic holdings?

Truth: daily intent holdings (stacked pre-QP weights, sleeve||ticker) -> phi(h) for h = 1..HMAX
in each sleeve's own active days, fixed full-sample per-sleeve Sigma (same object as
phi_persistence.py). Allocator: snapshots every DELTA active days + the sleeve's daily return.

  DELTA=21: decay fit phi(h) = phi_inf + (1-phi_inf) exp(-h/tau) on snapshot lags
      {21,42,...} recovers daily phi(5) and phibar(21) within +-0.10 for rotating sleeves
      (BMS, TSMS_Com, CSMom). Naive "frozen until next snapshot" misses them by more.
  DELTA=63: within +-0.15 for TSMS_Com, CSMom; fails for BMS (daily phi(63)=0.28
      ~ asymptote: points on the floor cannot recover phi(5)=0.84), the aliasing limit.
  Churn detector: corr(frozen-snapshot return, reported return) over DELTA-intervals
      ~ daily phibar(DELTA) within +-0.10 per sleeve; Spearman across sleeves >= 0.8.
  13F: table only -- phi(45), phibar over [45, 66].

usage: python research/phi_from_snapshots.py <run_dir> <stacked_intent_weights.csv>
writes phi_from_snapshots.csv (fit table) and phi_from_snapshots_churn.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
HMAX = 252
H_REPORT = [5, 21, 63, 126]
DELTAS = [21, 63]
LAG_13F = 45


def decay(h, phi_inf, tau):
    return phi_inf + (1.0 - phi_inf) * np.exp(-h / tau)


def phi_series(A, Sig, hmax):
    """phi(h), h=1..hmax, on the active-day rows of A (n x k). Fixed Sigma."""
    Q = A @ Sig
    qa = np.einsum("ij,ij->i", Q, A)
    out = np.full(hmax + 1, np.nan)
    for h in range(1, hmax + 1):
        if len(A) <= h + 10:
            break
        num = np.einsum("ij,ij->i", Q[:-h], A[h:])
        den = np.sqrt(qa[:-h] * qa[h:])
        ok = den > 0
        out[h] = float(np.mean(num[ok] / den[ok]))
    return out


def phi_snapshots(A, Sig, delta, hmax):
    """phi at lags m*delta using only every delta-th active row."""
    S = A[::delta]
    Q = S @ Sig
    qa = np.einsum("ij,ij->i", Q, S)
    lags, vals = [], []
    for m in range(1, hmax // delta + 1):
        if len(S) <= m + 3:
            break
        num = np.einsum("ij,ij->i", Q[:-m], S[m:])
        den = np.sqrt(qa[:-m] * qa[m:])
        ok = den > 0
        lags.append(m * delta); vals.append(float(np.mean(num[ok] / den[ok])))
    return np.array(lags), np.array(vals)


def phibar(phi, h):
    return float(np.nanmean(phi[1:h + 1]))


def main():
    print(f"run: {RUN}\nweights: {WFILE}")
    W = pd.read_csv(WFILE, index_col=0); W.index = pd.to_datetime(W.index)
    R = pd.read_csv(f"{RUN}/asset_returns_by_asset.csv", index_col=0); R.index = pd.to_datetime(R.index)
    R = R.clip(-0.5, 0.5)
    sleeves = sorted({c.split("||")[0] for c in W.columns})
    rows, churn_rows = [], []
    for s in sleeves:
        cols = [c for c in W.columns if c.startswith(s + "||")]
        tick = [c.split("||")[1] for c in cols]
        keep = [i for i, t in enumerate(tick) if t in R.columns]
        cols = [cols[i] for i in keep]; tick = [tick[i] for i in keep]
        Ws = W[cols].reindex(R.index).fillna(0.0); Ws.columns = tick
        Rs = R[tick].fillna(0.0)
        act = Ws.abs().sum(axis=1) > 0
        if act.sum() < 500:
            print(f"{s:20s} skipped (active {int(act.sum())})"); continue
        Sig = Rs[act].cov().values
        A = Ws[act].values
        phi_d = phi_series(A, Sig, HMAX)
        rec = {"sleeve": s, "active": int(act.sum())}
        for h in H_REPORT:
            rec[f"phi{h}_daily"] = phi_d[h]; rec[f"phibar{h}_daily"] = phibar(phi_d, h)
        rec["phi45_daily"] = phi_d[LAG_13F]
        rec["phibar45_66_daily"] = float(np.nanmean(phi_d[LAG_13F + 1:LAG_13F + 22]))
        for d in DELTAS:
            lags, vals = phi_snapshots(A, Sig, d, HMAX)
            rec[f"d{d}_npts"] = len(lags)
            rec[f"d{d}_snap"] = " ".join(f"{v:.2f}" for v in vals)
            fit_ok = len(lags) >= 3
            if fit_ok:
                try:
                    p, _ = curve_fit(decay, lags.astype(float), vals, p0=[max(vals.min(), -0.5), float(d)],
                                     bounds=([-1.0, 1.0], [1.0, 5000.0]), maxfev=20000)
                except Exception:
                    fit_ok = False
            if fit_ok:
                rec[f"d{d}_phi_inf"], rec[f"d{d}_tau"] = float(p[0]), float(p[1])
                fit = np.array([np.nan] + [decay(h, *p) for h in range(1, HMAX + 1)])
            else:
                rec[f"d{d}_phi_inf"] = rec[f"d{d}_tau"] = np.nan
                fit = np.full(HMAX + 1, np.nan)
            # naive allocator: frozen inside the interval, snapshot value at the multiples
            naive = np.full(HMAX + 1, np.nan)
            for h in range(1, HMAX + 1):
                if h < d:
                    naive[h] = 1.0
                else:
                    m = h // d
                    naive[h] = vals[m - 1] if m - 1 < len(vals) else np.nan
            for h in H_REPORT:
                rec[f"d{d}_phi{h}_fit"] = fit[h]
                rec[f"d{d}_phibar{h}_fit"] = phibar(fit, h)
                rec[f"d{d}_phibar{h}_naive"] = phibar(naive, h)
            # churn detector: frozen-snapshot return vs actual over each interval, pooled
            idx = np.where(act.values)[0]
            Rv = Rs.values; Wv = Ws.values
            fr, ac = [], []
            for k in range(0, len(idx) - d, d):
                w0 = Wv[idx[k]]
                seg = idx[k:k + d]
                fr.append(Rv[seg] @ w0); ac.append(np.einsum("ij,ij->i", Wv[seg], Rv[seg]))
            fr = np.concatenate(fr); ac = np.concatenate(ac)
            ok = np.isfinite(fr) & np.isfinite(ac)
            c = float(np.corrcoef(fr[ok], ac[ok])[0, 1])
            vr = float(np.var(ac[ok] - fr[ok]) / np.var(ac[ok]))
            rec[f"d{d}_churn_corr"] = c; rec[f"d{d}_churn_varratio"] = vr
            churn_rows.append({"sleeve": s, "delta": d, "churn_corr": c, "tracking_var_ratio": vr,
                               "phibar_delta_daily": phibar(phi_d, d), "phi_delta_daily": phi_d[d]})
        rows.append(rec)
    df = pd.DataFrame(rows); ch = pd.DataFrame(churn_rows)
    df.to_csv("phi_from_snapshots.csv", index=False, float_format="%.4f")
    ch.to_csv("phi_from_snapshots_churn.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    print("\nDaily truth (intent):")
    print(df[["sleeve", "active"] + [f"phi{h}_daily" for h in H_REPORT] + ["phi45_daily", "phibar45_66_daily"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    for d in DELTAS:
        print(f"\n=== snapshots every {d} active days: fit phi_inf + (1-phi_inf) exp(-h/tau) ===")
        print(df[["sleeve", f"d{d}_npts", f"d{d}_snap", f"d{d}_phi_inf", f"d{d}_tau"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"-- recovered vs daily (fit | naive-frozen), h in {H_REPORT}:")
        hdr = f"{'sleeve':20s}" + "".join(f" | phi{h}: day fit " for h in H_REPORT if h < d or d == 21) \
              + "".join(f" | pbar{h}: day fit naive" for h in H_REPORT)
        print(hdr)
        for _, r in df.iterrows():
            line = f"{r['sleeve']:20s}"
            for h in H_REPORT:
                if h < d or d == 21:
                    line += f" | {r[f'phi{h}_daily']:.2f} {r[f'd{d}_phi{h}_fit']:.2f} "
            for h in H_REPORT:
                line += f" | {r[f'phibar{h}_daily']:.2f} {r[f'd{d}_phibar{h}_fit']:.2f} {r[f'd{d}_phibar{h}_naive']:.2f} "
            print(line)
        print(f"-- churn detector (delta={d}): corr(frozen-snapshot ret, actual ret) vs daily phibar(delta)")
        sub = ch[ch.delta == d].copy()
        sub["err"] = sub.churn_corr - sub.phibar_delta_daily
        print(sub[["sleeve", "churn_corr", "phibar_delta_daily", "err", "tracking_var_ratio"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        rho = sub[["churn_corr", "phibar_delta_daily"]].corr(method="spearman").iloc[0, 1]
        print(f" Spearman(churn_corr, phibar_delta_daily) across sleeves = {rho:.2f} "
              f"max|err| = {sub.err.abs().max():.3f}")
    print("\nwrote phi_from_snapshots.csv, phi_from_snapshots_churn.csv")


if __name__ == "__main__":
    main()
