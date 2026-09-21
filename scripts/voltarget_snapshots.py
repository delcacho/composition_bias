"""
voltarget_snapshots.py -- "stream the vols, PIC the correlations" with the allocator's holdings:
R_PIC from periodic snapshots (held until the next filing, optionally lagged), D from the stream.

Forecast target: forward realized covariance of intent sleeve returns over [t, t+h].
Estimators (daily variance units), matched halflife L; D always = stream EWMA vol:
  A full stream D_stream . R_stream . D_stream
  C2(daily) prescription, daily holdings D_stream . R_PIC(daily) . D_stream
  C2(Delta) prescription, snapshot holdings D_stream . R_PIC(last snapshot <= t-lag) . D_stream
  BL(Delta) blend R = phibar . R_PIC(snap) + (1-phibar) . R_stream
Cadences: daily; 21 rows (monthly); 63 rows (quarterly); 63 rows + 45-row lag (13F-like).
Snapshots are on the union calendar (a manager files on calendar dates, not on its active days).

  Rotating pairs: RMSE(C2) increases monotonically daily < 21 < 63 < 63+45; frozen pairs
      within 1% across cadences.
  At 63+45, snapshot-PIC R loses to stream R on pairs with phibar < 0.7; ties/wins on frozen.
  Blend within 1% of the better of {C2(Delta), A} on every class.

usage: python research/voltarget_snapshots.py <run_dir> <stacked_intent_weights.csv> [L]
writes voltarget_snapshots_pairs.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_from_snapshots import phi_series # noqa: E402
import pair_weight as pw # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 252
PHI_NEUTRAL = "--phi=neutral" in sys.argv # market-neutral persistence feeds the blend weight (alternative)
OUT_CSV = "voltarget_snapshots_pairs" + ("_phineutral" if PHI_NEUTRAL else "") + ".csv"
HORIZONS = [21, 63]
CADENCES = [("daily", 1, 0), ("m21", 21, 0), ("q63", 63, 0), ("q63lag45", 63, 45)]
STRIDE = 5
RHO_REAL = 0.15
MIN_OBS = 300
HMAX = 252
PERSIST_WINDOW = 1008
PERSIST_REFRESH = 21
MIN_PERSIST_ACTIVE = 252


def corr_of(M):
    d = np.sqrt(np.einsum("tii->ti", M))
    return M / (d[:, :, None] * d[:, None, :])


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
    active = (np.abs(Wt).sum(axis=2) > 0)
    Rv = R.values
    rs = np.einsum("tij,tj->ti", Wt, Rv)
    print(f"run: {RUN}\nweights: {WFILE}\nsleeves={n} assets={k} days={T} L={L}")

    persist_cache = {}

    def trailing_persistence(t):
        """Persistence available at t, fit on a trailing window ending no later than t."""
        key = (t // PERSIST_REFRESH) * PERSIST_REFRESH
        if key in persist_cache:
            return persist_cache[key]
        s0 = max(0, key - PERSIST_WINDOW + 1)
        phibar_t, aret_t = {}, {}
        for i, s in enumerate(sleeves):
            act = active[s0:key + 1, i]
            if act.sum() < MIN_PERSIST_ACTIVE:
                continue
            A = Wt[s0:key + 1, i, :][act]
            Rw = Rv[s0:key + 1][act]
            Sig_s = np.cov(Rw.T)
            if PHI_NEUTRAL:
                ev, U = np.linalg.eigh(Sig_s); Sig_s = Sig_s - ev[-1] * np.outer(U[:, -1], U[:, -1])
            keep = np.abs(A).sum(axis=0) > 0
            if keep.sum() < 1:
                continue
            ph = phi_series(A[:, keep], Sig_s[np.ix_(keep, keep)], HMAX)
            phibar_t[s] = {h: float(np.nanmean(ph[1:h + 1])) for h in range(1, HMAX + 1)}
            aret_t[s] = pw.retention(ph)
        persist_cache[key] = (phibar_t, aret_t)
        return persist_cache[key]

    # snapshot weights per cadence: W_snap[t] = W[last snapshot row <= t - lag]
    def snap_weights(delta, lag):
        if delta == 1 and lag == 0:
            return Wt
        Ws = np.zeros_like(Wt)
        snap_rows = np.arange(0, T, delta)
        for t in range(T):
            r = snap_rows[snap_rows <= t - lag]
            Ws[t] = Wt[r[-1]] if len(r) else 0.0
        return Ws
    Wsnap = {nm: snap_weights(d, lg) for nm, d, lg in CADENCES}

    outer = np.einsum("ti,tj->tij", rs, rs)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)], axis=0)
    Sig = np.zeros((k, k)); S = np.zeros((n, n))
    G = {nm: np.full((T, n, n), np.nan) for nm, _, _ in CADENCES}
    S_all = np.full((T, n, n), np.nan)
    warm = L
    for t in range(T):
        rt = Rv[t]
        Sig = lam * Sig + (1 - lam) * np.outer(rt, rt)
        S = lam * S + (1 - lam) * np.outer(rs[t], rs[t])
        if t >= warm:
            S_all[t] = S
            for nm in G:
                G[nm][t] = Wsnap[nm][t] @ Sig @ Wsnap[nm][t].T

    rows = []
    for h in HORIZONS:
        F = (cum[h:] - cum[:-h]) / h
        F = np.concatenate([F, np.full((h, n, n), np.nan)], axis=0)
        ok_t = np.arange(warm, T - h, STRIDE)
        Fo = F[ok_t]; So = S_all[ok_t]; Sc = corr_of(So); Fc = corr_of(Fo)
        Gc = {nm: corr_of(G[nm][ok_t]) for nm in G}
        act_t = active[ok_t]
        Dij = np.sqrt(np.einsum("tii->ti", So))
        for i in range(n):
            for j in range(i + 1, n):
                m = act_t[:, i] & act_t[:, j] & np.isfinite(Fc[:, i, j]) & np.isfinite(Sc[:, i, j])
                for nm in G:
                    m &= np.isfinite(Gc[nm][:, i, j])
                if m.sum() < MIN_OBS:
                    continue
                y = Fo[m, i, j]; dd = Dij[m, i] * Dij[m, j]
                sst = np.sum((y - y.mean()) ** 2)
                pair_pb_h = np.full(len(ok_t), np.nan)
                pair_w_h = np.full(len(ok_t), np.nan)
                stale_pb = {nm: np.full(len(ok_t), np.nan) for nm, _, _ in CADENCES}
                stale_w = {nm: np.full(len(ok_t), np.nan) for nm, _, _ in CADENCES}
                for u, tt in enumerate(ok_t):
                    phibar_t, aret_t = trailing_persistence(int(tt))
                    si, sj = sleeves[i], sleeves[j]
                    if si not in phibar_t or sj not in phibar_t or si not in aret_t or sj not in aret_t:
                        continue
                    pair_pb_h[u] = float(np.sqrt(phibar_t[si][h] * phibar_t[sj][h]))
                    pair_w_h[u] = pw.pair_weight(aret_t[si], aret_t[sj], h)
                    for nm, d, lg in CADENCES:
                        stale = max(1, d + lg)
                        stale_pb[nm][u] = float(np.sqrt(phibar_t[si][stale] * phibar_t[sj][stale]))
                        stale_w[nm][u] = pw.pair_weight_stale(aret_t[si], aret_t[sj], h, stale)
                rec = {"pair": f"{sleeves[i]}|{sleeves[j]}", "h": h, "n": int(m.sum()),
                       "mean_rho_fwd": float(np.nanmean(Fc[m, i, j])),
                       "pair_phibar_h": float(np.nanmedian(pair_pb_h[m])),
                       "w_model_h": float(np.nanmedian(pair_w_h[m]))}
                fA = dd * Sc[m, i, j]
                rec["rmse_A"] = float(np.sqrt(np.mean((y - fA) ** 2)))
                for nm, d, lg in CADENCES:
                    Rsnap = Gc[nm][m, i, j]
                    fC = dd * Rsnap
                    rec[f"rmse_C2_{nm}"] = float(np.sqrt(np.mean((y - fC) ** 2)))
                    pb = stale_pb[nm][m]; pm = stale_w[nm][m]
                    okb = np.isfinite(pb)
                    okm = np.isfinite(pm)
                    rec[f"phibar_{nm}"] = float(np.nanmedian(pb)); rec[f"w_model_{nm}"] = float(np.nanmedian(pm))
                    rec[f"n_blend_{nm}"] = int(okb.sum())
                    fB = dd[okb] * (pb[okb] * Rsnap[okb] + (1 - pb[okb]) * Sc[m, i, j][okb])
                    fM = dd[okm] * (pm[okm] * Rsnap[okm] + (1 - pm[okm]) * Sc[m, i, j][okm])
                    rec[f"rmse_BL_{nm}"] = float(np.sqrt(np.mean((y[okb] - fB) ** 2))) if okb.sum() >= MIN_OBS else np.nan
                    rec[f"rmse_BLM_{nm}"] = float(np.sqrt(np.mean((y[okm] - fM) ** 2))) if okm.sum() >= MIN_OBS else np.nan
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")

    df["real"] = df.mean_rho_fwd.abs() > RHO_REAL
    df["frozen"] = df.pair_phibar_h >= 0.95
    names = [nm for nm, _, _ in CADENCES]
    print("\n=== RMSE ratio to full stream A (median over pairs); C2 = D_stream.R_PIC(snapshot); BL = blend ===")
    for h in HORIZONS:
        for cls, msk in [("rotating |rho|>0.15", (df.h == h) & df.real & ~df.frozen),
                         ("frozen |rho|>0.15", (df.h == h) & df.real & df.frozen),
                         ("|rho|<=0.15", (df.h == h) & ~df.real)]:
            sub = df[msk]
            if len(sub) == 0:
                continue
            c2 = " ".join(f"{nm}:{(sub[f'rmse_C2_{nm}'] / sub.rmse_A).median():.3f}" for nm in names)
            bl = " ".join(f"{nm}:{(sub[f'rmse_BL_{nm}'] / sub.rmse_A).median():.3f}" for nm in names)
            blm = " ".join(f"{nm}:{(sub[f'rmse_BLM_{nm}'] / sub.rmse_A).median():.3f}" for nm in names)
            print(f" h={h:>3} {cls:20s} n={len(sub):>2}\n C2 {c2}\n BL {bl} (shortcut sqrt(phibar phibar))\n BLM {blm} (model weight, eq:pairweight)")
    print("\n=== model weight vs shortcut: RMSE(BLM)/RMSE(BL), median over pairs (below 1 = model better) ===")
    for h in HORIZONS:
        for cls, msk in [("rotating |rho|>0.15", (df.h == h) & df.real & ~df.frozen), ("frozen |rho|>0.15", (df.h == h) & df.real & df.frozen), ("|rho|<=0.15", (df.h == h) & ~df.real)]:
            sub = df[msk]
            if len(sub):
                print(f" h={h:>3} {cls:20s} n={len(sub):>2} " + " ".join(f"{nm}:{(sub[f'rmse_BLM_{nm}'] / sub[f'rmse_BL_{nm}']).median():.3f}" for nm in names))
    print("\n=== per pair, real-correlation pairs, h=21: RMSE/A for C2 and BL by cadence (phibar at staleness in brackets) ===")
    sub = df[(df.h == 21) & df.real].sort_values("pair_phibar_h")
    for _, r in sub.iterrows():
        c2 = " ".join(f"{(r[f'rmse_C2_{nm}'] / r['rmse_A']):.3f}[{r[f'phibar_{nm}']:.2f}]" for nm in names)
        bl = " ".join(f"{(r[f'rmse_BL_{nm}'] / r['rmse_A']):.3f}" for nm in names)
        print(f" {r['pair']:34s} rho {r['mean_rho_fwd']:+.2f} C2 {c2} BL {bl}")
    print("\n=== P10: at q63lag45, pairs with phibar(108) < 0.7 -- does snapshot-PIC R lose to stream R? ===")
    for h in HORIZONS:
        sub = df[(df.h == h) & df.real & (df.phibar_q63lag45 < 0.7)]
        if len(sub):
            r = sub.rmse_C2_q63lag45 / sub.rmse_A; b = sub.rmse_BL_q63lag45 / sub.rmse_A
            print(f" h={h}: n={len(sub)} C2/A median {r.median():.3f} (C2 loses on {int((r > 1).sum())}/{len(sub)}) BL/A median {b.median():.3f} (BL loses on {int((b > 1).sum())}/{len(sub)})")
    print("\n=== Blend vs better of {C2, A}, ratio (1.00 = matches the better one) ===")
    for h in HORIZONS:
        for nm in names:
            sub = df[(df.h == h) & df.real]
            best = np.minimum(sub[f"rmse_C2_{nm}"], sub.rmse_A)
            r = sub[f"rmse_BL_{nm}"] / best
            print(f" h={h:>3} {nm:9s} median {r.median():.3f} p90 {r.quantile(0.9):.3f} max {r.max():.3f} within 1%: {int((r <= 1.01).sum())}/{len(sub)}")
    print("\nwrote voltarget_snapshots_pairs.csv")


if __name__ == "__main__":
    main()
