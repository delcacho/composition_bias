"""
phi_persistence.py -- the composition-persistence statistic phi(h) that scopes
PIC/PIV vs return-stream covariance (the composition-bias article).

  phi(h) = E_t[ w_t' S w_{t+h} / sqrt(w_t'S w_t * w_{t+h}'S w_{t+h}) ] (mix persistence)

the risk-space cosine between a sleeve's book now and h days ahead. Scale-
invariant, so it measures the mix (rotation), not the exposure (gross). The
companion exposure persistence is the same computation without normalization:

  expphi(h) = E_t[ w_t' S w_{t+h} ] / E_t[ w_t' S w_t ]

Reads a run's effective_weights.csv (sleeve||ticker) + asset_returns_by_asset.csv.
S is a fixed full-sample asset covariance per sleeve (a fixed risk geometry, so
phi measures only how the weights rotate in it; a time-varying S would conflate).

The article's use, all from this one holdings-only statistic:
  - when PIC/PIV beats a stream of lookback L: edge ~ phibar(h) - phibar(L)
  - when the blend is needed: phibar(h) below ~0.8 at the allocation horizon
  - difference-as-information: 1 - phi = composition drift / crowding

Usage: python research/phi_persistence.py [RUN_DIR]
"""
import sys
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    r"logs/2026-08-22/20030101-20260701/012714775490_epsilon_guard"
HORIZONS = [5, 21, 63, 126]
ACTIVE_EPS = 1e-10 # a day counts only if the sleeve carries risk


def phi_curve(W: np.ndarray, S: np.ndarray, horizons):
    """W: (T,k) sleeve weights; S: (k,k) asset cov. Returns mix-phi and exp-phi per h."""
    SW = W @ S # (T,k): S w_t (S symmetric)
    q = np.einsum('ij,ij->i', W, SW) # w_t' S w_t (T,)
    out = {}
    for h in horizons:
        num = np.einsum('ij,ij->i', W[:-h], SW[h:]) # w_t' S w_{t+h}
        qa, qb = q[:-h], q[h:]
        act = (qa > ACTIVE_EPS) & (qb > ACTIVE_EPS) # both endpoints active
        if act.sum() < 20:
            out[h] = (np.nan, np.nan, 0)
            continue
        a, b = np.sqrt(qa[act]), np.sqrt(qb[act]) # risk-norms now / ahead
        cos = num[act] / (a * b) # c_t
        mix = float(np.nanmean(cos)) # phi_mix = E[c]
        expp = float(num[act].mean() / qa[act].mean()) # phi_exp = E[abc]/E[a^2]
        out[h] = (mix, expp, int(act.sum()))
    return out


def main():
    print(f"run: {RUN}\n")
    # Optional 2nd arg: weights file. Default = post-netting effective weights; pass the stacked
    # pre-QP intended weights for each sleeve's own composition (effective weights are cross-netted).
    WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}/effective_weights.csv"
    OUT_TAG = "_intent" if len(sys.argv) > 2 else ""
    print(f"weights: {WFILE}")
    W = pd.read_csv(WFILE, index_col=0)
    R = pd.read_csv(f"{RUN}/asset_returns_by_asset.csv", index_col=0)
    W.index = pd.to_datetime(W.index)
    R.index = pd.to_datetime(R.index)
    idx = W.index.intersection(R.index)
    W, R = W.loc[idx], R.loc[idx]

    sleeves = {}
    for c in W.columns:
        if "||" in c:
            sleeves.setdefault(c.split("||")[0], []).append(c)

    hh = " ".join(f"phi{h:>3}" for h in HORIZONS)
    print(f"{'sleeve':18} {hh} | {'exp21':>6} | {'exp126':>6} active")
    print("-" * 78)

    rows = []
    for sleeve in sorted(sleeves):
        cols = sleeves[sleeve]
        tks = [c.split("||")[1] for c in cols]
        avail = [t for t in tks if t in R.columns]
        if len(avail) < 1:
            continue
        colmap = {c.split("||")[1]: c for c in cols}
        Wm = W[[colmap[t] for t in avail]].fillna(0.0)
        Rm = R[avail].fillna(0.0)
        # fixed full-sample asset covariance for this sleeve's universe
        S = np.cov(Rm.to_numpy().T) if len(avail) > 1 else \
            np.array([[float(Rm.to_numpy().var())]])
        S = np.atleast_2d(S)
        cur = phi_curve(Wm.to_numpy(), S, HORIZONS)
        mix = [cur[h][0] for h in HORIZONS]
        exp21 = cur[21][1]
        exp126 = cur[126][1]
        n_act = cur[21][2]
        cells = " ".join(f"{m:6.2f}" if np.isfinite(m) else f"{'--':>6}" for m in mix)
        print(f"{sleeve:18} {cells} | {exp21:6.2f} | {exp126:6.2f} {n_act:6d}")
        rows.append(dict(sleeve=sleeve, **{f"phi{h}": cur[h][0] for h in HORIZONS},
                         expphi21=exp21, expphi126=exp126, n_active=n_act,
                         n_assets=len(avail)))

    out = pd.DataFrame(rows)
    out.to_csv(f"phi_persistence{OUT_TAG}.csv", index=False, float_format="%.4f")
    print("\nwrote phi_persistence.csv")
    print("\nRead-out:")
    print(" low phi63 with high phi5 -> bandpass: PIC/PIV beats a long stream (edge ~ phi(h)-phi(L))")
    print(" phi ~ 1 at all h -> estimators coincide; either serves (the noise floor)")
    print(" exposure-phi << mix-phi -> gross exposure rotates faster than the mix")


if __name__ == "__main__":
    main()
