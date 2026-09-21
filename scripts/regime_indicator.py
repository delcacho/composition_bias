"""
regime_indicator.py -- can a market-level causal regime indicator time the stress correlation
residual that the sleeve streams cannot (lambda_regime.py: hit-rate 0.27, recall 0.29)?

Indicators (causal, previous day, on when >= expanding 80th percentile):
  vix       : FRED VIXCLS previous close (data/vixcls.csv)
  assetvol  : cross-sectional mean of trailing 21-day realized vol across all assets
Objects per real-correlation pair and horizon h:
  hit-rate / recall vs ex-post stress windows (forward cross-sleeve vol in top 20%)
  residual (realized corr - PIC corr) in indicator-on vs off windows
  Lambda_hist : stress-conditional stream (EWMA on indicator days) minus trailing, as in lambda_regime.py
  delta_hat   : causal expanding mean of the pair's own past residuals in on-windows
                (only windows with s + h <= t)
  covariance RMSE in on-windows: C2 = D_stream.R_PIC vs C2 + Lambda_hist vs C2 + delta_hat

Findings:
  VIX indicator hit-rate and recall > 0.40; asset-level comparable.
  Residual on-minus-off > 0.03 on real-correlation pairs.
  (C2 + delta_hat)/C2 RMSE in on-windows: median < 1; Lambda_hist still fails on bond/crypto.

usage: python research/regime_indicator.py <run_dir> <stacked_intent_weights.csv> [L]
writes regime_indicator_pairs.csv
"""
import os
import sys
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 252
VIXFILE = "data/vixcls.csv"
if not os.path.exists(VIXFILE):   # pipeline stages run with cwd=OUT (public_panel_pipeline.py); the VIX file lives at the repo root.
    VIXFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vixcls.csv")   # fallback: the repository's data/vixcls.csv
HL_STRESS = 252
HORIZONS = [21, 63]
STRIDE = 5
RHO_REAL = 0.15
Q = 0.80
VOL_WIN = 21
MIN_OBS = 200


def corr_of(M):
    d = np.sqrt(np.einsum("tii->ti", M))
    return M / (d[:, :, None] * d[:, None, :])


def expanding_flag(x, q, min_periods=252):
    thr = pd.Series(x).expanding(min_periods=min_periods).quantile(q).values
    return np.where(np.isfinite(x) & np.isfinite(thr), x >= thr, False)


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
    print(f"run: {RUN}\nweights: {WFILE}\nsleeves={n} assets={k} days={T} L={L}")

    # indicators (previous-day information)
    vix = pd.read_csv(VIXFILE, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    vix_prev = vix.reindex(W.index, method="ffill").shift(1).values
    Rmask = R.replace(0.0, np.nan)
    avol = Rmask.rolling(VOL_WIN, min_periods=10).std().mean(axis=1).shift(1).values
    indicators = {"vix": expanding_flag(vix_prev, Q), "assetvol": expanding_flag(avol, Q)}
    for nm, I in indicators.items():
        print(f"indicator {nm}: on {I.mean()*100:.1f}% of days")

    outer = np.einsum("ti,tj->tij", rs, rs)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)], axis=0)
    Sig = np.zeros((k, k)); S = np.zeros((n, n))
    Ss = {nm: np.zeros((n, n)) for nm in indicators}; ns_ = {nm: 0 for nm in indicators}
    G_all = np.full((T, n, n), np.nan); S_all = np.full((T, n, n), np.nan)
    SS_all = {nm: np.full((T, n, n), np.nan) for nm in indicators}
    warm = L
    for t in range(T):
        rt = rs[t]
        Sig = lam * Sig + (1 - lam) * np.outer(Rv[t], Rv[t])
        S = lam * S + (1 - lam) * np.outer(rt, rt)
        for nm, I in indicators.items():
            if I[t]:
                Ss[nm] = lam_s * Ss[nm] + (1 - lam_s) * np.outer(rt, rt); ns_[nm] += 1
        if t >= warm:
            G_all[t] = Wt[t] @ Sig @ Wt[t].T; S_all[t] = S
            for nm in indicators:
                if ns_[nm] >= 126: SS_all[nm][t] = Ss[nm]

    rows = []
    for h in HORIZONS:
        F = (cum[h:] - cum[:-h]) / h
        F = np.concatenate([F, np.full((h, n, n), np.nan)], axis=0)
        ok_t = np.arange(warm, T - h, STRIDE)
        Fo = F[ok_t]; So = S_all[ok_t]
        Fc, Gc, Sc = corr_of(Fo), corr_of(G_all[ok_t]), corr_of(So)
        SSc = {nm: corr_of(SS_all[nm][ok_t]) for nm in indicators}
        act_t = active[ok_t]
        fv = np.sqrt(np.einsum("tii->ti", Fo))
        fv_mean = np.array([fv[m][act_t[m]].mean() if act_t[m].any() else np.nan for m in range(len(ok_t))])
        post = fv_mean >= np.nanquantile(fv_mean, Q)
        Dij = np.sqrt(np.einsum("tii->ti", So))
        nper = h // STRIDE + 1                      # sampled windows not yet realized at t
        for nm, I in indicators.items():
            I_t = I[ok_t]
            hit = (post & I_t).sum() / max(I_t.sum(), 1); rec_ = (post & I_t).sum() / max(post.sum(), 1)
            for i in range(n):
                for j in range(i + 1, n):
                    m = act_t[:, i] & act_t[:, j]
                    for M in (Fc, Gc, Sc, SSc[nm]):
                        m &= np.isfinite(M[:, i, j])
                    if m.sum() < MIN_OBS or I_t[m].sum() < 30:
                        continue
                    y = Fc[m, i, j]; pic = Gc[m, i, j]; tr = Sc[m, i, j]; st = SSc[nm][m, i, j]
                    io = I_t[m]; ps = post[m]
                    resid = y - pic
                    # causal delta_hat: expanding mean of past on-window residuals, realized by t
                    r_on = np.where(io, resid, 0.0); c_on = io.astype(float)
                    cs_r = np.concatenate([[0.0], np.cumsum(r_on)]); cs_c = np.concatenate([[0.0], np.cumsum(c_on)])
                    idx = np.arange(len(y)); lagk = np.clip(idx - nper + 1, 0, None)
                    cnt = cs_c[lagk]; dh = np.where(cnt >= 10, cs_r[lagk] / np.maximum(cnt, 1), 0.0)
                    dd = Dij[m, i] * Dij[m, j]; yc = Fo[m, i, j]
                    fC2 = dd * pic; fL = dd * (pic + (st - tr)); fD = dd * (pic + dh)
                    def rm(f, sel): return float(np.sqrt(np.mean((yc - f)[sel] ** 2)))
                    rows.append({"indicator": nm, "pair": f"{sleeves[i]}|{sleeves[j]}", "h": h, "n": int(m.sum()),
                                 "mean_rho_fwd": float(y.mean()), "hit_rate": float(hit), "recall": float(rec_),
                                 "resid_on": float(resid[io].mean()), "resid_off": float(resid[~io].mean()),
                                 "resid_post_stress": float(resid[ps].mean()),
                                 "lambda_hist": float((st - tr).mean()),
                                 "resid_lambda_on": float((y - st)[io].mean()),
                                 "delta_hat_last": float(dh[-1]),
                                 "rmse_C2_on": rm(fC2, io), "rmse_C2L_on": rm(fL, io), "rmse_C2D_on": rm(fD, io),
                                 "rmse_C2_all": rm(fC2, slice(None)),
                                 "rmse_switchD_all": float(np.sqrt(np.mean((yc - np.where(io, fD, fC2)) ** 2)))})
    df = pd.DataFrame(rows)
    df.to_csv("regime_indicator_pairs.csv", index=False, float_format="%.6g")
    df["real"] = df.mean_rho_fwd.abs() > RHO_REAL
    for nm in indicators:
        for h in HORIZONS:
            sub = df[(df.indicator == nm) & (df.h == h) & df.real]
            if len(sub) == 0: continue
            print(f"\n=== indicator {nm}, h={h}: real-correlation pairs n={len(sub)} ===")
            print(f"quality: hit-rate {sub.hit_rate.iloc[0]:.2f}  recall {sub.recall.iloc[0]:.2f}")
            d = sub.resid_on - sub.resid_off
            print(f"residual (realized - PIC corr): on {sub.resid_on.mean():+.3f}  off {sub.resid_off.mean():+.3f}  diff {d.mean():+.3f} (pairs >0: {int((d > 0).sum())}/{len(sub)})  | ex-post stress {sub.resid_post_stress.mean():+.3f}")
            rL = sub.rmse_C2L_on / sub.rmse_C2_on; rD = sub.rmse_C2D_on / sub.rmse_C2_on; rS = sub.rmse_switchD_all / sub.rmse_C2_all
            print(f"RMSE ratio in on-windows: (C2+Lambda_hist)/C2 median {rL.median():.3f} (<1 on {int((rL < 1).sum())}/{len(sub)})"
                  f"   (C2+delta_hat)/C2 median {rD.median():.3f} (<1 on {int((rD < 1).sum())}/{len(sub)})   switch-delta/C2 all days {rS.median():.3f}")
            print("    per pair: rho | resid on/off | Lambda_hist, resid with it (on) | delta_hat | (C2+L)/C2 (C2+d)/C2")
            for _, r in sub.sort_values("mean_rho_fwd").iterrows():
                print(f"      {r['pair']:34s} {r['mean_rho_fwd']:+.2f} | {r['resid_on']:+.3f}/{r['resid_off']:+.3f} | {r['lambda_hist']:+.3f}, {r['resid_lambda_on']:+.3f} | {r['delta_hat_last']:+.3f} | {r['rmse_C2L_on']/r['rmse_C2_on']:.3f} {r['rmse_C2D_on']/r['rmse_C2_on']:.3f}")
    print("\nwrote regime_indicator_pairs.csv")


if __name__ == "__main__":
    main()
