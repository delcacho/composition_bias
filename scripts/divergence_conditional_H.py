"""Identification check on the divergence result: a regression of forward joint risk on the
stream correlation R and the gap H-R spans the same space as one on R and the holdings level H, so
"gap significant, stream not" cannot show that the disagreement is a distinct signal. The clean test
is the gap conditional on H: regress each forward joint-risk outcome on H and the gap together (pair
fixed effects, 63-day block bootstrap) and ask whether the gap survives given H; and a matched test,
the gap's quintile lift within terciles of H. Same construction as divergence_across_pairs.py.

Before the regression the script REPRODUCES Exhibit C (the per-pair top-minus-bottom gap-quintile
statistics) from the supplied panel and asserts they match the article to a tolerance, so a wrong or
drifted commodity book fails loudly instead of feeding stale numbers to the regression.

Writes <RUN>/divergence_conditional_H_results.csv (samples: `material`, the six correlated pairs the
article reports, and `all_pairs`, the ten-pair robustness). reproduce.py reads the `material` rows.

usage: python -u research/divergence_conditional_H.py <run> <stacked_weights.csv>
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from allocator_multiasset import load_commodity, RUN  # RUN = sys.argv[1]

LOOKBACK, FWD, TAIL_PCT = 252, 63, 90
SEED, NBOOT, BLOCK = 7, 1000, 63
rng = np.random.default_rng(SEED)

# Exhibit C of the article (ex:divpairs): per pair, top-minus-bottom gap-quintile frequency of a large
# 21-day joint move and forward 63-day pair volatility. The book must reproduce these before it is used.
EXHIBIT_C = {
    "LongOnlyEW x TSMomentum": (0.149, 0.032), "CSMomentum x CSValue": (0.137, 0.018),
    "CSCarry x TSMomentum": (0.106, 0.015), "CSValue x TSMomentum": (0.094, 0.014),
    "CSMomentum x LongOnlyEW": (0.067, 0.013), "CSCarry x CSValue": (0.028, 0.004),
    "CSCarry x CSMomentum": (0.023, 0.008), "CSCarry x LongOnlyEW": (-0.012, 0.002),
    "CSValue x LongOnlyEW": (-0.018, 0.005), "CSMomentum x TSMomentum": (-0.065, 0.008),
}
TOL = 0.003


def build_panel():
    cdates, sleeves, assets, Wt, Rv = load_commodity()
    T, K = len(cdates), len(sleeves)
    rs = np.zeros((T, K))
    for i in range(K):
        rs[1:, i] = np.einsum("ta,ta->t", Wt[:-1, i, :], Rv[1:, :])
    Sig = [None] * T
    for t in range(LOOKBACK, T - FWD):
        Sig[t] = np.cov(Rv[t - LOOKBACK:t].T)
    panels = []
    for i in range(K):
        for j in range(i + 1, K):
            rows = []; pair = (rs[:, i] + rs[:, j]) / 2
            for t in range(LOOKBACK, T - FWD):
                wi, wj = Wt[t, i, :], Wt[t, j, :]; S = Sig[t]
                vi, vj = float(wi @ S @ wi), float(wj @ S @ wj)
                if vi <= 0 or vj <= 0:
                    continue
                pic = float(wi @ S @ wj) / np.sqrt(vi * vj)              # H, holdings-implied correlation
                a, b = rs[t - LOOKBACK:t, i], rs[t - LOOKBACK:t, j]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    continue
                roll = float(np.corrcoef(a, b)[0, 1])                    # R, stream correlation
                p21 = float(pair[t + 1:t + 22].sum())
                pv63 = float(pair[t + 1:t + 1 + FWD].std(ddof=0) * np.sqrt(252))
                fi, fj = Rv[t + 1:t + 1 + FWD] @ wi, Rv[t + 1:t + 1 + FWD] @ wj
                rcorr = float(np.corrcoef(fi, fj)[0, 1]) if fi.std() > 0 and fj.std() > 0 else np.nan
                rows.append((t, pic - roll, pic, roll, p21, pv63, rcorr))
            if len(rows) < 500:
                continue
            d = pd.DataFrame(rows, columns=["t", "gap", "H", "R", "p21", "pv63", "rcorr"])
            d["jm21"] = (d.p21.abs() > np.percentile(d.p21.abs(), TAIL_PCT)).astype(float)
            d["pair"] = f"{sleeves[i]} x {sleeves[j]}"; d["mean_rho"] = float(d.rcorr.mean())
            panels.append(d)
    return pd.concat(panels, ignore_index=True)


def check_book(P):
    """Reproduce Exhibit C and assert it matches the article; raise on a wrong or drifted book."""
    bad = []
    for pr, g in P.groupby("pair"):
        g = g.assign(gq=pd.qcut(g.gap, 5, labels=False, duplicates="drop"))
        jm = g[g.gq == g.gq.max()].jm21.mean() - g[g.gq == 0].jm21.mean()
        vol = g[g.gq == g.gq.max()].pv63.mean() - g[g.gq == 0].pv63.mean()
        ejm, evol = EXHIBIT_C[pr]
        ok = abs(jm - ejm) <= TOL and abs(vol - evol) <= TOL
        print(f"  {pr:26s} jm {jm:+.3f} (exp {ejm:+.3f})  vol {vol:+.3f} (exp {evol:+.3f})  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(pr)
    if bad:
        raise SystemExit(f"BOOK CHECK FAILED: Exhibit C not reproduced on {bad}; refusing to run the regression on this book.")
    print("  book reproduces Exhibit C.")


def prep(df, cols):
    dm = {c: (df[c] - df.groupby("pair")[c].transform("mean")).values for c in cols}
    days = np.sort(df.t.unique()); pos = [np.where(df.t.values == tt)[0] for tt in days]
    return dm, days, pos


def fit_boot(dm, days, pos, outcome, regs):
    y = dm[outcome]; X = np.column_stack([dm[c] for c in regs])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    D = len(days); nb = int(np.ceil(D / BLOCK)); boots = []
    for _ in range(NBOOT):
        starts = rng.integers(0, D - BLOCK + 1, nb)
        sel = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:D]
        ridx = np.concatenate([pos[d] for d in sel])
        boots.append(np.linalg.lstsq(X[ridx], y[ridx], rcond=None)[0])
    B = np.array(boots); sd = B.std(0)
    return beta, sd, np.percentile(B, [5, 95], axis=0)


def main():
    P = build_panel()
    mat = P[P.mean_rho.abs() > 0.15].copy()               # the six correlated pairs the article reports
    print(f"pairs {P.pair.nunique()} ({mat.pair.nunique()} material); pair-dates {len(P)} / {len(mat)}")
    print("book check (Exhibit C reproduction):")
    check_book(P)
    out_rows = []
    for sname, sdf in (("material", mat), ("all_pairs", P)):
        dm, days, pos = prep(sdf, ["jm21", "pv63", "gap", "H"])
        for outcome in ["jm21", "pv63"]:
            for regs in (["gap"], ["H"], ["H", "gap"]):
                beta, sd, ci = fit_boot(dm, days, pos, outcome, regs); model = "+".join(regs)
                for k, name in enumerate(regs):
                    t = beta[k] / sd[k] if sd[k] > 0 else np.nan
                    out_rows.append({"sample": sname, "outcome": outcome, "model": model, "regressor": name,
                                     "beta": round(beta[k], 4), "t": round(t, 2), "ci_lo": round(ci[0][k], 4), "ci_hi": round(ci[1][k], 4)})
                    print(f"  {sname:9s} {outcome:5s} ~ {model:6s} {name}: b={beta[k]:+.4f} t={t:+.2f}")
    # matched: gap's top-minus-bottom jm21 lift within terciles of the pair-demeaned holdings level
    m = mat.copy()
    m["Hdm"] = m.H - m.groupby("pair")["H"].transform("mean")
    m["gdm"] = m.gap - m.groupby("pair")["gap"].transform("mean")
    m["Hter"] = m.groupby("pair")["Hdm"].transform(lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    for ht in sorted(m.Hter.dropna().unique()):
        s = m[m.Hter == ht].assign(gq=lambda z: z.groupby("pair")["gdm"].transform(
            lambda v: pd.qcut(v, 5, labels=False, duplicates="drop")))
        diff = s[s.gq == 4].jm21.mean() - s[s.gq == 0].jm21.mean()
        out_rows.append({"sample": "material", "outcome": "jm21_tercile", "model": f"H_tercile_{int(ht)}",
                         "regressor": "gap_top_minus_bottom", "beta": round(diff, 3), "t": "", "ci_lo": "", "ci_hi": ""})
        print(f"  H tercile {int(ht)} (mean H {s.H.mean():+.2f}): gap top-minus-bottom jm21 {diff:+.3f}")
    out = os.path.join(RUN, "divergence_conditional_H_results.csv")
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
