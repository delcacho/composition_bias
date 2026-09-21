"""
voltarget_prescription.py -- test the vol-targeted-manager prescription
    Sigma_alloc = D_target . R_PIC . D_target (+ Lambda_stream)
on a book of vol-targeted sleeves (intent frame; TARGET_SLEEVE = 5%, delivered 5.2-5.5%).

Forecast target: forward realized covariance of intent sleeve returns over [t, t+h].
Forecasts (daily variance units), all at matched halflife L:
  A full stream : EWMA covariance of sleeve returns
  B full PIC : W_t Sigma_t W_t' with EWMA asset covariance
  C1 prescription : sigma*^2 . corr(B) (level = mandate constant, shape = PIC)
  C2 prescription' : sqrt(A_ii A_jj) . corr(B) (level = stream vol, shape = PIC)

  Level (diagonal): RMSE(target const) / RMSE(stream) in [0.9, 1.1]; PIC-implied level worst.
  Shape (off-diag): C1 beats A on rotating pairs with |rho|>0.15, ties on frozen pairs,
     equals B within 5% RMSE everywhere (scalar theorem).
  Lambda: residual correlation (realized - PIC corr) higher in stress (top-20% cross-sleeve
     forward vol) than calm by > 0.05 on real-correlation pairs.

usage: python research/voltarget_prescription.py <run_dir> <stacked_intent_weights.csv> [L]
writes voltarget_prescription_pairs.csv, voltarget_prescription_diag.csv
"""
import sys
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 252
HORIZONS = [21, 63]
SIGMA_TARGET = 0.05 / np.sqrt(252) # TARGET_SLEEVE, daily
STRIDE = 5
RHO_REAL = 0.15
STRESS_Q = 0.80
MIN_OBS = 300


def main():
    lam = 0.5 ** (1.0 / L)
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
    active = (np.abs(Wt).sum(axis=2) > 0) # T x n
    Rv = R.values
    rs = np.einsum("tij,tj->ti", Wt, Rv) # sleeve intent returns, T x n
    print(f"run: {RUN}\nweights: {WFILE}\nsleeves={n} assets={k} days={T} L={L}")

    # forward realized covariance via cumulative outer products
    outer = np.einsum("ti,tj->tij", rs, rs)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)], axis=0)

    # incremental EWMA: asset covariance -> PIC book covariance; sleeve-return covariance -> stream
    Sig = np.zeros((k, k)); S = np.zeros((n, n))
    G_all = np.full((T, n, n), np.nan); S_all = np.full((T, n, n), np.nan)
    warm = L
    for t in range(T):
        rt = Rv[t]
        Sig = lam * Sig + (1 - lam) * np.outer(rt, rt)
        S = lam * S + (1 - lam) * np.outer(rs[t], rs[t])
        if t >= warm:
            G_all[t] = Wt[t] @ Sig @ Wt[t].T
            S_all[t] = S

    def corr_of(M):
        d = np.sqrt(np.einsum("tii->ti", M))
        return M / (d[:, :, None] * d[:, None, :])

    rows, diag_rows = [], []
    phis = pd.read_csv("phi_from_snapshots.csv").set_index("sleeve") if _exists("phi_from_snapshots.csv") else None
    for h in HORIZONS:
        F = (cum[h:] - cum[:-h]) / h # F[t] = fwd cov over (t, t+h], length T-h
        F = np.concatenate([F, np.full((h, n, n), np.nan)], axis=0)
        ok_t = np.arange(warm, T - h, STRIDE)
        Fc = corr_of(F[ok_t]); Gc = corr_of(G_all[ok_t]); Sc = corr_of(S_all[ok_t])
        # stress state: cross-sleeve mean forward vol (active sleeves) in top quantile
        fv = np.sqrt(np.einsum("tii->ti", F[ok_t]))
        act_t = active[ok_t]
        fv_mean = np.array([fv[m][act_t[m]].mean() if act_t[m].any() else np.nan for m in range(len(ok_t))])
        thr = np.nanquantile(fv_mean, STRESS_Q); stress = fv_mean >= thr
        # diagonal
        for i, s in enumerate(sleeves):
            m = act_t[:, i] & np.isfinite(F[ok_t][:, i, i]) & np.isfinite(G_all[ok_t][:, i, i])
            if m.sum() < MIN_OBS: continue
            y = F[ok_t][m, i, i]
            fc = {"stream": S_all[ok_t][m, i, i], "pic": G_all[ok_t][m, i, i],
                  "target": np.full(m.sum(), SIGMA_TARGET ** 2)}
            rec = {"sleeve": s, "h": h, "n": int(m.sum()), "realized_vol_ann": float(np.sqrt(y.mean() * 252))}
            for nm, f in fc.items():
                rec[f"rmse_{nm}"] = float(np.sqrt(np.mean((y - f) ** 2)))
            rec["ratio_target_stream"] = rec["rmse_target"] / rec["rmse_stream"]
            rec["ratio_pic_stream"] = rec["rmse_pic"] / rec["rmse_stream"]
            diag_rows.append(rec)
        # off-diagonal
        for i in range(n):
            for j in range(i + 1, n):
                m = act_t[:, i] & act_t[:, j] & np.isfinite(Fc[:, i, j]) & np.isfinite(Gc[:, i, j]) & np.isfinite(Sc[:, i, j])
                if m.sum() < MIN_OBS: continue
                y = F[ok_t][m, i, j]
                A = S_all[ok_t][m, i, j]
                B = G_all[ok_t][m, i, j]
                C1 = SIGMA_TARGET ** 2 * Gc[m, i, j]
                C2 = np.sqrt(S_all[ok_t][m, i, i] * S_all[ok_t][m, j, j]) * Gc[m, i, j]
                sst = np.sum((y - y.mean()) ** 2)
                rec = {"pair": f"{sleeves[i]}|{sleeves[j]}", "h": h, "n": int(m.sum()),
                       "mean_rho_fwd": float(np.nanmean(Fc[m, i, j])),
                       "mean_rho_pic": float(np.nanmean(Gc[m, i, j])),
                       "mean_rho_stream": float(np.nanmean(Sc[m, i, j]))}
                if phis is not None:
                    col = f"phibar{h}_daily"
                    rec["pair_phibar"] = float(np.sqrt(phis.loc[sleeves[i], col] * phis.loc[sleeves[j], col])) \
                        if sleeves[i] in phis.index and sleeves[j] in phis.index else np.nan
                for nm, f in {"A_stream": A, "B_pic": B, "C1_target_picR": C1, "C2_streamD_picR": C2}.items():
                    rec[f"rmse_{nm}"] = float(np.sqrt(np.mean((y - f) ** 2)))
                    rec[f"r2_{nm}"] = float(1 - np.sum((y - f) ** 2) / sst) if sst > 0 else np.nan
                # significance of C2 vs A: Diebold-Mariano on squared-error differentials (NW, bandwidth h)
                dA = (y - A) ** 2 - (y - C2) ** 2
                dd_ = dA[np.isfinite(dA)]; mu = dd_.mean(); g = np.mean((dd_ - mu) ** 2)
                for kk in range(1, h):
                    g += 2 * (1 - kk / h) * np.mean((dd_[kk:] - mu) * (dd_[:-kk] - mu))
                rec["dm_t_C2_vs_A"] = float(mu / np.sqrt(max(g, 1e-30) / len(dd_)))
                # Lambda: residual correlation in stress vs calm
                res = Fc[m, i, j] - Gc[m, i, j]; st = stress[m]
                rec["resid_corr_stress"] = float(np.nanmean(res[st])) if st.sum() > 20 else np.nan
                rec["resid_corr_calm"] = float(np.nanmean(res[~st])) if (~st).sum() > 20 else np.nan
                rec["resid_corr_stream_stress"] = float(np.nanmean((Fc[m, i, j] - Sc[m, i, j])[st])) if st.sum() > 20 else np.nan
                rows.append(rec)
    df = pd.DataFrame(rows); dg = pd.DataFrame(diag_rows)
    df.to_csv("voltarget_prescription_pairs.csv", index=False, float_format="%.6g")
    dg.to_csv("voltarget_prescription_diag.csv", index=False, float_format="%.6g")

    pd.set_option("display.width", 250)
    print("\n=== Level (diagonal): RMSE of forward variance forecasts, ratios to the stream ===")
    print(dg[["sleeve", "h", "n", "realized_vol_ann", "ratio_target_stream", "ratio_pic_stream"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for h in HORIZONS:
        sub = dg[dg.h == h]
        print(f" h={h}: median ratio target/stream {sub.ratio_target_stream.median():.3f} pic/stream {sub.ratio_pic_stream.median():.3f}"
              f" target in [0.9,1.1]: {int(((sub.ratio_target_stream >= 0.9) & (sub.ratio_target_stream <= 1.1)).sum())}/{len(sub)}"
              f" pic worst: {int((sub.ratio_pic_stream > sub.ratio_target_stream).sum())}/{len(sub)}")
    print("\n=== Shape (off-diagonal): RMSE relative to full stream A (ratio<1 = beats stream) ===")
    df["real"] = df.mean_rho_fwd.abs() > RHO_REAL
    df["frozen"] = df.pair_phibar >= 0.95 if "pair_phibar" in df else np.nan
    for h in HORIZONS:
        for cls, msk in [("real corr, rotating", (df.h == h) & df.real & ~df.frozen),
                         ("real corr, frozen", (df.h == h) & df.real & df.frozen),
                         ("|rho|<=0.15 (noise)", (df.h == h) & ~df.real)]:
            sub = df[msk]
            if len(sub) == 0: continue
            rB = (sub.rmse_B_pic / sub.rmse_A_stream); rC1 = (sub.rmse_C1_target_picR / sub.rmse_A_stream)
            rC2 = (sub.rmse_C2_streamD_picR / sub.rmse_A_stream); c1b = (sub.rmse_C1_target_picR / sub.rmse_B_pic)
            print(f" h={h:>3} {cls:22s} n={len(sub):>2} median RMSE/A: B {rB.median():.3f} C1 {rC1.median():.3f} C2 {rC2.median():.3f}"
                  f" | C1/B {c1b.median():.3f} (max {c1b.max():.3f}) | C1 beats A: {int((rC1 < 1).sum())}/{len(sub)}"
                  f" | mean R2: A {sub.r2_A_stream.mean():.2f} B {sub.r2_B_pic.mean():.2f} C1 {sub.r2_C1_target_picR.mean():.2f} C2 {sub.r2_C2_streamD_picR.mean():.2f}"
                  f" | DM t(C2 vs A): median {sub.dm_t_C2_vs_A.median():+.2f}, |t|>1.96 on {int((sub.dm_t_C2_vs_A.abs() > 1.96).sum())}/{len(sub)}, min {sub.dm_t_C2_vs_A.min():+.2f} max {sub.dm_t_C2_vs_A.max():+.2f}")
    print("\n per pair (real-correlation pairs), h=21: RMSE ratios to A and residual corr stress/calm")
    sub = df[(df.h == 21) & df.real].sort_values("pair_phibar")
    for _, r in sub.iterrows():
        print(f" {r['pair']:36s} phibar {r['pair_phibar']:.2f} rho fwd/pic/str {r['mean_rho_fwd']:+.2f}/{r['mean_rho_pic']:+.2f}/{r['mean_rho_stream']:+.2f}"
              f" B/A {r['rmse_B_pic']/r['rmse_A_stream']:.3f} C1/A {r['rmse_C1_target_picR']/r['rmse_A_stream']:.3f} C2/A {r['rmse_C2_streamD_picR']/r['rmse_A_stream']:.3f}"
              f" resid stress/calm {r['resid_corr_stress']:+.3f}/{r['resid_corr_calm']:+.3f}")
    print("\n=== Lambda: mean residual correlation (realized - PIC) in stress vs calm, real-correlation pairs ===")
    for h in HORIZONS:
        sub = df[(df.h == h) & df.real]
        d = sub.resid_corr_stress - sub.resid_corr_calm
        print(f" h={h}: n={len(sub)} mean resid stress {sub.resid_corr_stress.mean():+.3f} calm {sub.resid_corr_calm.mean():+.3f}"
              f" diff {d.mean():+.3f} (pairs with diff>0: {int((d > 0).sum())}/{len(sub)})"
              f" | vs stream corr in stress: {sub.resid_corr_stream_stress.mean():+.3f}")
    print("\nwrote voltarget_prescription_pairs.csv, voltarget_prescription_diag.csv")


def _exists(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
