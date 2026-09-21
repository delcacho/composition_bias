"""Robustness check: is the fresh holdings edge on the frozen band a clone-history artifact (young funds' streams
see fewer days than the asset covariance)? Split the pooled pair-dates by the shorter clone history of the pair,
in trading days before the forecast date, and read the fresh holdings-corr ratio per fixed band and bucket.
Reads existing chunks only. Usage: python edge_by_history.py [252|63]"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex

L = int(sys.argv[1]) if len(sys.argv) > 1 else 252
BACK = 6 * L
root = f"etf_results/experiment"
sfx = "" if L == 252 else f"_L{L}"
classes = ["allocation", "mf", "longshort", "longonly"]
dirs = [f"{root}/{c}_neutral_all{sfx}" for c in classes if os.path.isdir(f"{root}/{c}_neutral_all{sfx}/chunks")]
sample = pd.read_csv(er.U); cls_of = ex.fund_classes()
admit = sample.in_sample_staleness if "in_sample_staleness" in sample.columns else sample.in_sample
active = [t for t in sample[admit].ticker.astype(str)]
first = pd.to_datetime(sample.set_index(sample.ticker.astype(str))["first"])
# reconstruct the pooled fund index -> ticker: keep = sorted(by_class) per class, offset in pooling order
tick = []
for d in dirs:
    c = os.path.basename(d).split("_")[0]
    inclass = set(t for t in active if cls_of.get(t, "longonly") == c)
    seen = sorted(pd.read_csv(f"{d}/persistence_trailing.csv").ticker.astype(str).unique())
    n_dir = 0
    for f in glob.glob(f"{d}/chunks/*.parquet"):
        if not f.endswith("_diag.parquet"):
            Cc = pd.read_parquet(f, columns=["i", "j"])
            if len(Cc):
                n_dir = max(n_dir, int(max(Cc.i.max(), Cc.j.max())) + 1)
    # keep = sorted(inclass & priced); every kept fund prints in persistence_trailing, so seen == keep when the
    # index count matches (funds without a computable clone were dropped before indexing)
    assert set(seen) <= inclass and len(seen) == n_dir, (c, len(seen), n_dir, len(inclass))
    tick += seen
tick = np.array(tick)
Rs_index = pd.read_parquet(f"{er.OUT}/sizing_dates.parquet").date if False else None
# date_ix -> date: the experiment's Rs index; rebuild cheaply from the clones' calendar in the class dir
dates = pd.to_datetime(sorted(pd.read_csv(f"{dirs[0]}/persistence_trailing.csv").date.unique()))
# date_ix is a row of the full daily index; persistence_trailing has only forecast dates. Map via the daily index:
Rs = er.load_returns(); idx = Rs.index
hist_days = {} # ticker -> position of first holdings day in idx
for t in tick:
    f = first.get(t, pd.NaT)
    hist_days[t] = int(idx.searchsorted(f)) if pd.notna(f) else -1
pooled = f"{root}/pooled_neutral_all{sfx}"
FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
BUCK = [(0, 504, "<2y"), (504, 1008, "2-4y"), (1008, BACK, f"4y-{BACK}d"), (BACK, 10**9, f">={BACK}d (full window)")]
ESTS = [("cH", "cS"), ("cF5", "cS"), ("cF10", "cS"), ("cP13", "cS"), ("cP22", "cS")]   # holdings correlation; statistical (K 5, 10) and named proxy (13, 22) factor benchmarks
acc = None
for C in ex.chunks(pooled):
    C = C[C.age == 0]
    if C.empty:
        continue
    hi = np.array([hist_days[tick[i]] for i in C.i.values]); hj = np.array([hist_days[tick[j]] for j in C.j.values])
    h = C.date_ix.values - np.maximum(hi, hj) # shorter clone history of the pair, trading days
    band = np.full(len(C), "", dtype=object); buck = np.full(len(C), "", dtype=object)
    for lo, hi_, lab in FIXED:
        band[(C.phimin.values >= lo) & (C.phimin.values < hi_)] = lab
    for lo, hi_, lab in BUCK:
        buck[(h >= lo) & (h < hi_)] = lab
    C2 = C.assign(band=band, buck=buck); C2 = C2[(C2.band != "") & (C2.buck != "")]
    per = ex.paired_sums(C2, ["band", "buck", "i", "j"], ESTS)
    acc = per if acc is None else acc.add(per, fill_value=0.0)
rows = []
# one generator per estimator: the holdings column keeps the draw sequence of the original single-estimator script
rngs = {est: np.random.default_rng(7 if est == "cH" else 100 + k) for k, (est, _) in enumerate(ESTS)}
for (bd, bk), per_all in acc.groupby(level=["band", "buck"]):
    row = {"L": L, "band": bd, "history": bk}
    for est, _ in ESTS:
        rng = rngs[est]
        per = per_all[per_all[f"n_{est}"] >= 12]
        sfx = "" if est == "cH" else f"_{est[1:]}"                # cH keeps the original column names; cF5 -> _F5, cF10 -> _F10
        if per.empty:
            row.update({f"pairs{sfx}": 0, f"pair_dates{sfx}": 0, f"ratio{sfx}": np.nan, f"lo{sfx}": np.nan, f"hi{sfx}": np.nan}); continue
        f1 = per.index.get_level_values("i").values.astype(int); f2 = per.index.get_level_values("j").values.astype(int)
        funds = np.unique(np.concatenate([f1, f2])); nf = int(funds.max()) + 1
        se, sr = per[est].values, per[f"ref_{est}"].values
        boots = []
        for _ in range(400):
            cnt = np.bincount(rng.choice(funds, len(funds), replace=True), minlength=nf)
            w = (cnt[f1] * cnt[f2]).astype(float)
            if (w * sr).sum() > 0:
                boots.append(0.5 * np.log((w * se).sum() / (w * sr).sum()))
        row.update({f"pairs{sfx}": len(per), f"pair_dates{sfx}": int(per[f"n_{est}"].sum()), f"ratio{sfx}": np.sqrt(se.sum() / sr.sum()),
                    f"lo{sfx}": np.exp(np.percentile(boots, 2.5)), f"hi{sfx}": np.exp(np.percentile(boots, 97.5))})
    if row.get("pairs", 0) or row.get("pairs_F5", 0):
        rows.append(row)
R = pd.DataFrame(rows)
order = {b[2]: k for k, b in enumerate(BUCK)}; R["o"] = R.history.map(order)
R = R.sort_values(["band", "o"]).drop(columns="o")
out = f"{pooled}/fresh_edge_by_history.csv"; R.to_csv(out, index=False, float_format="%.4f")
print(f"L={L}, BACK={BACK}: fresh holdings-corr ratio to stream by band x shorter clone history of the pair")
print(R.to_string(index=False)); print("wrote", out)
