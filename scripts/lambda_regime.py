"""
lambda_regime.py -- does a regime-conditional long stream contain the stress correlation uplift
Lambda that neither PIC nor a trailing EWMA stream contains (voltarget_prescription.py: residual +0.03 in stress)?

Causal stress indicator I_t: trailing 21-day cross-sleeve mean realized vol >= expanding 80th pct.
Stress-conditional stream: EWMA of sleeve-return outer products updated only on I_t days
(halflife HL_STRESS stress-days); calm-conditional likewise (control). Trailing: ordinary EWMA L.
Lambda_ij(t) = R_stress_ij(t) - R_trailing_ij(t).

  Residual: in ex-post stress forward windows the stress-conditional stream's residual
      correlation is ~0 (|mean| <= 0.02) vs +0.03 for PIC and trailing; calm-conditional worse.
  Sign: Lambda_ij > 0 for the majority of pairs, both positive-rho and hedge (negative-rho).
  Stress-window forecast: when I_t is on, D_stream.(R_PIC + Lambda) has lower covariance RMSE
      than D_stream.R_PIC by a positive median amount.

usage: python research/lambda_regime.py <run_dir> <stacked_intent_weights.csv> [L]
writes lambda_regime_pairs.csv
"""
import sys
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 252
HL_STRESS = 252 # in stress-days
HORIZONS = [21, 63]
STRIDE = 5
RHO_REAL = 0.15
STRESS_Q = 0.80
VOL_WIN = 21
MIN_OBS = 200


def corr_of(M):
    d = np.sqrt(np.einsum("tii->ti", M))
    return M / (d[:, :, None] * d[:, None, :])


def main():
    lam = 0.5 ** (1.0 / L); lam_s = 0.5 ** (1.0 / HL_STRESS)
    W = pd.read_csv(WFILE, index_col=0); W.index = pd.to_datetime(W.index)
    R = pd.read_csv(f"{RUN}/asset_returns_by_asset.csv", index_col=0); R.index = pd.to_datetime(R.index)
    R = R.clip(-0.5, 0.5)
    sleeves = sorted({c.split("||")[0] for c in W.columns})
    assets = sorted({c.split("||")[1] for c in W.columns} & set(R.columns))
    R = R[assets].reindex(W.index).fillna(0.0)
    n, k, T = len(sleeves), len(assets), len(W)
    Wt = np.zeros((T, n, k))
    for i, s in enumerate(sleeves):
        for j, a in enumerate(assets):
            c = f"{s}||{a}"
            if c in W.columns:
                Wt[:, i, j] = W[c].fillna(0.0).values
    active = (np.abs(Wt).sum(axis=2) > 0)
    Rv = R.values
    rs = np.einsum("tij,tj->ti", Wt, Rv)
    print(f"run: {RUN}\nweights: {WFILE}\nsleeves={n} assets={k} days={T} L={L} HL_stress={HL_STRESS}")

    # causal stress indicator from trailing cross-sleeve realized vol (active sleeves only)
    rs_df = pd.DataFrame(rs).where(pd.DataFrame(active))
    trail_vol = rs_df.rolling(VOL_WIN, min_periods=10).std().mean(axis=1).values
    thr = pd.Series(trail_vol).expanding(min_periods=252).quantile(STRESS_Q).values
    I = np.where(np.isfinite(trail_vol) & np.isfinite(thr), trail_vol >= thr, False)
    print(f"causal stress indicator on {I.mean()*100:.1f}% of days")

    outer = np.einsum("ti,tj->tij", rs, rs)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)], axis=0)
    Sig = np.zeros((k, k)); S = np.zeros((n, n)); Ss = np.zeros((n, n)); Sc_ = np.zeros((n, n))
    ns_ = 0; nc_ = 0
    G_all = np.full((T, n, n), np.nan); S_all = np.full((T, n, n), np.nan)
    SS_all = np.full((T, n, n), np.nan); SC_all = np.full((T, n, n), np.nan)
    warm = L
    for t in range(T):
        rt = rs[t]
        Sig = lam * Sig + (1 - lam) * np.outer(Rv[t], Rv[t])
        S = lam * S + (1 - lam) * np.outer(rt, rt)
        if I[t]:
            Ss = lam_s * Ss + (1 - lam_s) * np.outer(rt, rt); ns_ += 1
        else:
            Sc_ = lam_s * Sc_ + (1 - lam_s) * np.outer(rt, rt); nc_ += 1
        if t >= warm:
            G_all[t] = Wt[t] @ Sig @ Wt[t].T; S_all[t] = S
            if ns_ >= 126: SS_all[t] = Ss
            if nc_ >= 126: SC_all[t] = Sc_

    rows = []
    for h in HORIZONS:
        F = (cum[h:] - cum[:-h]) / h
        F = np.concatenate([F, np.full((h, n, n), np.nan)], axis=0)
        ok_t = np.arange(warm, T - h, STRIDE)
        Fo = F[ok_t]; So = S_all[ok_t]
        Fc, Gc, Sc = corr_of(Fo), corr_of(G_all[ok_t]), corr_of(So)
        SSc, SCc = corr_of(SS_all[ok_t]), corr_of(SC_all[ok_t])
        act_t = active[ok_t]; I_t = I[ok_t]
        # ex-post regime: forward cross-sleeve mean vol in top quantile (as in voltarget_prescription.py)
        fv = np.sqrt(np.einsum("tii->ti", Fo))
        fv_mean = np.array([fv[m][act_t[m]].mean() if act_t[m].any() else np.nan for m in range(len(ok_t))])
        post = fv_mean >= np.nanquantile(fv_mean, STRESS_Q)
        Dij = np.sqrt(np.einsum("tii->ti", So))
        for i in range(n):
            for j in range(i + 1, n):
                m = act_t[:, i] & act_t[:, j]
                for M in (Fc, Gc, Sc, SSc, SCc):
                    m &= np.isfinite(M[:, i, j])
                if m.sum() < MIN_OBS or post[m].sum() < 30 or I_t[m].sum() < 30:
                    continue
                y = Fc[m, i, j]; pic = Gc[m, i, j]; tr = Sc[m, i, j]; st = SSc[m, i, j]; cl = SCc[m, i, j]
                ps = post[m]; io = I_t[m]
                lam_ij = st - tr
                rec = {"pair": f"{sleeves[i]}|{sleeves[j]}", "h": h, "n": int(m.sum()),
                       "mean_rho_fwd": float(y.mean()),
                       "lambda_mean": float(lam_ij.mean()),
                       "resid_pic_stress": float((y - pic)[ps].mean()),
                       "resid_trail_stress": float((y - tr)[ps].mean()),
                       "resid_stresscond_stress": float((y - st)[ps].mean()),
                       "resid_calmcond_stress": float((y - cl)[ps].mean()),
                       "resid_pic_calm": float((y - pic)[~ps].mean()),
                       "resid_stresscond_calm": float((y - st)[~ps].mean()),
                       "ind_hit_rate": float((ps & io).sum() / max(io.sum(), 1)),
                       "ind_recall": float((ps & io).sum() / max(ps.sum(), 1))}
                # timing: covariance RMSE in indicator-on windows, C2 vs C2 + Lambda
                dd = Dij[m, i] * Dij[m, j]; yc = Fo[m, i, j]
                fC2 = dd * pic; fCL = dd * (pic + lam_ij)
                rec["rmse_C2_on"] = float(np.sqrt(np.mean((yc - fC2)[io] ** 2)))
                rec["rmse_C2L_on"] = float(np.sqrt(np.mean((yc - fCL)[io] ** 2)))
                rec["rmse_C2_all"] = float(np.sqrt(np.mean((yc - fC2) ** 2)))
                fSw = np.where(io, fCL, fC2)
                rec["rmse_switch_all"] = float(np.sqrt(np.mean((yc - fSw) ** 2)))
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv("lambda_regime_pairs.csv", index=False, float_format="%.6g")
    df["real"] = df.mean_rho_fwd.abs() > RHO_REAL
    pd.set_option("display.width", 250)
    for h in HORIZONS:
        sub = df[(df.h == h) & df.real]
        print(f"\n=== h={h}: real-correlation pairs n={len(sub)} (indicator hit-rate {sub.ind_hit_rate.mean():.2f}, recall {sub.ind_recall.mean():.2f}) ===")
        print("Residual -- mean residual correlation (realized - forecast) in ex-post stress windows:")
        for c, nm in [("resid_pic_stress", "PIC"), ("resid_trail_stress", "trailing stream"),
                      ("resid_stresscond_stress", "stress-conditional stream"), ("resid_calmcond_stress", "calm-conditional stream")]:
            print(f" {nm:28s} {sub[c].mean():+.3f} (pairs >0: {int((sub[c] > 0).sum())}/{len(sub)})")
        print(f" in calm windows: PIC {sub.resid_pic_calm.mean():+.3f} stress-conditional {sub.resid_stresscond_calm.mean():+.3f}")
        pos = sub[sub.mean_rho_fwd > 0]; neg = sub[sub.mean_rho_fwd < 0]
        print(f"Sign -- Lambda = R_stress - R_trailing: mean {sub.lambda_mean.mean():+.3f}; >0 on {int((sub.lambda_mean > 0).sum())}/{len(sub)}"
              f" | positive-rho pairs {pos.lambda_mean.mean():+.3f} (>0 {int((pos.lambda_mean > 0).sum())}/{len(pos)})"
              f" | hedge pairs {neg.lambda_mean.mean():+.3f} (>0 {int((neg.lambda_mean > 0).sum())}/{len(neg)})")
        r_on = sub.rmse_C2L_on / sub.rmse_C2_on; r_all = sub.rmse_switch_all / sub.rmse_C2_all
        print(f"Stress-window forecast -- RMSE ratio (C2+Lambda)/C2 in indicator-on windows: median {r_on.median():.3f}, <1 on {int((r_on < 1).sum())}/{len(sub)}"
              f" | switch/C2 over all days: median {r_all.median():.3f}")
        print(" per pair: rho | Lambda | resid stress PIC/trail/stresscond | (C2+L)/C2 on")
        for _, r in sub.sort_values("mean_rho_fwd").iterrows():
            print(f" {r['pair']:34s} {r['mean_rho_fwd']:+.2f} | {r['lambda_mean']:+.3f} | {r['resid_pic_stress']:+.3f}/{r['resid_trail_stress']:+.3f}/{r['resid_stresscond_stress']:+.3f} | {r['rmse_C2L_on']/r['rmse_C2_on']:.3f}")
    print("\nwrote lambda_regime_pairs.csv")


if __name__ == "__main__":
    main()
