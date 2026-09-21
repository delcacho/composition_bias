"""etf_dyadic_inference.py -- dependence-structure robustness for the active-ETF pair panel.

A fund enters many pairs, in both roles, so clustering by the pair's less persistent fund alone may
understate the dependence. On one pooled chunk directory this script reports:

(a) the two key regressions of the staleness experiment (loss of the holdings correlation against the
    stream; staleness cost of the aged book against the fresh one), age x persistence-quartile dummies,
    with two variance estimates side by side: the experiment's (clustered by the pair's less persistent
    fund and by date) and a DYADIC one that treats every two observations sharing either fund as one
    cluster (Aronow, Samii and Assenova 2015), again combined with date clustering. The dyadic meat is
    sum_f S_f S_f' - sum_p S_p S_p', with S_f the score sum over every observation containing fund f in
    either role and S_p the score sum of pair p: two observations of the same pair share both funds
    and are counted twice by the first sum, once after the correction.
(b) the pooled error-ratio cells of the fastest and frozen bands at 21, 63 and 108 days with three
    bootstraps: funds only (the experiment's, product weights so both funds of a pair are respected),
    calendar quarters only (blocks of 63 trading days), and the two crossed.

  python research/etf_dyadic_inference.py [dir]    default etf_results/experiment/pooled_neutral_all_L63_fullwin
Writes <dir>/dyadic_regressions.csv and <dir>/twoway_bootstrap_cells.csv.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_staleness_experiment as ex

d = sys.argv[1] if len(sys.argv) > 1 else "etf_results/experiment/pooled_neutral_all_L63_fullwin"
AGES = [0, 21, 63, 108]
BLOCK = 63
FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
files = [f for f in sorted(glob.glob(f"{d}/chunks/*.parquet")) if not f.endswith("_diag.parquet")]


def chunks():
    for f in files:
        yield pd.read_parquet(f)


nf = 0
for C in chunks():
    if len(C):
        nf = max(nf, int(max(C.i.max(), C.j.max())) + 1)


def dyadic_ols(y_of, ages):
    cols, design = ex.design_age_quartile(ages)
    k = len(cols)
    yq = lambda C: np.where(C.qmin.values >= 1, y_of(C), np.nan)
    XtX = np.zeros((k, k)); Xty = np.zeros(k); nobs = 0
    for C in chunks():
        Xd, y = design(C), yq(C); ok = np.isfinite(y) & (Xd.sum(axis=1) > 0)
        Xd, y = Xd[ok], y[ok]; XtX += Xd.T @ Xd; Xty += Xd.T @ y; nobs += len(y)
    used = np.diag(XtX) > 0
    beta = np.zeros(k); beta[used] = np.linalg.solve(XtX[np.ix_(used, used)], Xty[used])
    S_f = np.zeros((nf, k)); S_p = np.zeros((nf * nf, k)); S_key = np.zeros((nf, k))
    M_date = np.zeros((k, k)); Mi_f = np.zeros((k, k)); Mi_p = np.zeros((k, k)); Mi_key = np.zeros((k, k))
    for C in chunks():
        Xd, y = design(C), yq(C); ok = np.isfinite(y) & (Xd.sum(axis=1) > 0)
        if not ok.any():
            continue
        Xd, y = Xd[ok], y[ok]; e = y - Xd @ beta; Xe = Xd * e[:, None]
        i, j, kf = C.i.values[ok].astype(int), C.j.values[ok].astype(int), C.keyfund.values[ok].astype(int)
        pid = i * nf + j
        sd = Xe.sum(axis=0); M_date += np.outer(sd, sd)
        Sd_f = np.zeros((nf, k)); Sd_p = {}; Sd_key = np.zeros((nf, k))
        for c in range(k):
            np.add.at(Sd_f[:, c], i, Xe[:, c]); np.add.at(Sd_f[:, c], j, Xe[:, c]); np.add.at(Sd_key[:, c], kf, Xe[:, c])
        P = pd.DataFrame(Xe).groupby(pid).sum()
        S_f += Sd_f; S_key += Sd_key; S_p[P.index.values] += P.values
        Mi_f += Sd_f.T @ Sd_f; Mi_key += Sd_key.T @ Sd_key; Mi_p += P.values.T @ P.values
    M_dyad = S_f.T @ S_f - S_p.T @ S_p; M_key = S_key.T @ S_key
    Vinv = np.zeros((k, k)); Vinv[np.ix_(used, used)] = np.linalg.inv(XtX[np.ix_(used, used)])
    V_key = Vinv @ (M_key + M_date - Mi_key) @ Vinv
    V_dyad = Vinv @ (M_dyad + M_date - (Mi_f - Mi_p)) @ Vinv
    return cols, beta, V_key, V_dyad, nobs, used


rows = []
for name, y_of, ages in [("holdings corr vs stream", lambda C: (C.cH - C.cS).values, AGES),
                         ("staleness cost, holdings corr vs fresh", lambda C: (C.cH - C.cH0).values, AGES[1:])]:
    cols, beta, Vk, Vd, nobs, used = dyadic_ols(y_of, ages)
    for a in ages:
        for lab, c in [("Q1", None), ("Q4", None), ("Q1 - Q4", None)]:
            v = np.zeros(len(cols))
            if lab == "Q1 - Q4":
                v[cols.index((a, 1))] = 1; v[cols.index((a, 4))] = -1
            else:
                v[cols.index((a, int(lab[1])))] = 1
            b = float(v @ beta); sk = float(np.sqrt(max(v @ Vk @ v, 0))); sdy = float(np.sqrt(max(v @ Vd @ v, 0)))
            rows.append({"regression": name, "age": a, "stratum": lab, "coef": b, "se_keyfund_date": sk,
                         "t_keyfund_date": b / sk if sk > 0 else np.nan, "se_dyadic_date": sdy,
                         "t_dyadic_date": b / sdy if sdy > 0 else np.nan, "se_ratio": sdy / sk if sk > 0 else np.nan, "nobs": nobs})
R = pd.DataFrame(rows)
print("clustered regressions: key-fund x date (experiment) against dyadic x date")
print(R.round(4).to_string(index=False))
R.to_csv(f"{d}/dyadic_regressions.csv", index=False, float_format="%.6f")

# ---- (b) ratio cells with fund, block and crossed bootstraps
parts = []
for C in chunks():
    C = C[(C.age > 0) & (C.qmin >= 1)]
    if C.empty:
        continue
    m = np.isfinite(C.cH.values) & np.isfinite(C.cS.values)
    C = C[m]
    band = np.full(len(C), "", dtype=object)
    for lo, hi, lab in FIXED:
        band[(C.phimin.values >= lo) & (C.phimin.values < hi)] = lab
    Z = pd.DataFrame({"band": band, "age": C.age.values, "i": C.i.values.astype(int), "j": C.j.values.astype(int),
                      "q": (C.date_ix.values // BLOCK).astype(int), "sH": C.cH.values, "sS": C.cS.values, "n": 1})
    parts.append(Z.groupby(["band", "age", "i", "j", "q"]).sum())
G = pd.concat(parts).groupby(level=["band", "age", "i", "j", "q"]).sum().reset_index()
G = G[G.band != ""]
# the exhibit's cells keep a pair only where it has at least 12 pair-dates in the cell (analyze.emit); same here
npd = G.groupby(["band", "age", "i", "j"]).n.transform("sum")
G = G[npd >= 12]
rng = np.random.default_rng(7); NB = 500
out = []
for (bd, age), g in G.groupby(["band", "age"]):
    if bd not in ("<0.75", ">0.95", "0.75-0.85", "0.85-0.95"):
        continue
    i, j, q = g.i.values, g.j.values, g.q.values
    funds = np.unique(np.concatenate([i, j])); blocks = np.unique(q); nq = int(q.max()) + 1
    sH, sS = g.sH.values, g.sS.values
    point = np.sqrt(sH.sum() / sS.sum())
    res = {"band": bd, "age": age, "pairs": int(len(g[["i", "j"]].drop_duplicates())), "blocks": int(len(blocks)), "ratio": point}
    for scheme in ("funds", "blocks", "crossed"):
        bs = []
        for _ in range(NB):
            w = np.ones(len(g))
            if scheme in ("funds", "crossed"):
                cnt = np.bincount(rng.choice(funds, len(funds), replace=True), minlength=nf); w = w * cnt[i] * cnt[j]
            if scheme in ("blocks", "crossed"):
                bc = np.bincount(rng.choice(blocks, len(blocks), replace=True), minlength=nq); w = w * bc[q]
            num, den = (w * sH).sum(), (w * sS).sum()
            if den > 0 and num > 0:
                bs.append(0.5 * np.log(num / den))
        res[f"{scheme} lo"] = np.exp(np.percentile(bs, 2.5)); res[f"{scheme} hi"] = np.exp(np.percentile(bs, 97.5))
    out.append(res)
B = pd.DataFrame(out)
order = {b[2]: k for k, b in enumerate(FIXED)}
B = B.sort_values(["age", "band"], key=lambda s: s.map(order) if s.name == "band" else s)
print("\nratio cells, holdings corr / stream, 95% bootstrap intervals by resampling scheme")
print(B.round(3).to_string(index=False))
B.to_csv(f"{d}/twoway_bootstrap_cells.csv", index=False, float_format="%.6f")
print("wrote", f"{d}/dyadic_regressions.csv", "and", f"{d}/twoway_bootstrap_cells.csv")
