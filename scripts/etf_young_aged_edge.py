"""Does a short-history manager's holdings edge survive an aged filing? The rule's youth fallback takes
holdings on a fresh filing; this scores whether it should hold on an aged one too. On the under-two-year
pair-dates only, read the holdings-corr / stream ratio by persistence band AND by age of the book (0, 21,
63, 108 trading days). If the fastest band stays below 1.00 at 63 days, a young rotating manager still
beats the stream on a quarter-old filing; if it crosses one, the fresh-filing scope stands. Reads existing
chunks only (etf_edge_by_history.py's construction, split by age instead of by history bucket, restricted
to the under-two-year bucket). Usage: python etf_young_aged_edge.py [252|63]"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex

L = int(sys.argv[1]) if len(sys.argv) > 1 else 63
BACK = 6 * L
root = "etf_results/experiment"
sfx = "" if L == 252 else f"_L{L}"
classes = ["allocation", "mf", "longshort", "longonly"]
dirs = [f"{root}/{c}_neutral_all{sfx}" for c in classes if os.path.isdir(f"{root}/{c}_neutral_all{sfx}/chunks")]
sample = pd.read_csv(er.U); cls_of = ex.fund_classes()
admit = sample.in_sample_staleness if "in_sample_staleness" in sample.columns else sample.in_sample
active = [t for t in sample[admit].ticker.astype(str)]
first = pd.to_datetime(sample.set_index(sample.ticker.astype(str))["first"])
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
    assert set(seen) <= inclass and len(seen) == n_dir, (c, len(seen), n_dir, len(inclass))
    tick += seen
tick = np.array(tick)
Rs = er.load_returns(); idx = Rs.index
hist_days = {t: (int(idx.searchsorted(first.get(t, pd.NaT))) if pd.notna(first.get(t, pd.NaT)) else -1) for t in tick}
pooled = f"{root}/pooled_neutral_all{sfx}"
FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
# edge_by_history's history buckets, assigned in order (later wins); the <2y tag is that script's
# under-two-year column, and at halflife 63 the full-window bucket (BACK, inf) overwrites [BACK, 504),
# so <2y is [0, BACK), a record short of the six-halflife window.
BUCK = [(0, 504, "<2y"), (504, 1008, "2-4y"), (1008, BACK, f"4y-{BACK}d"), (BACK, 10**9, "full")]
AGES = [0, 21, 63, 108]

acc = None
for C in ex.chunks(pooled):
    C = C[C.age.isin(AGES)]
    if C.empty:
        continue
    hi = np.array([hist_days[tick[i]] for i in C.i.values]); hj = np.array([hist_days[tick[j]] for j in C.j.values])
    h = C.date_ix.values - np.maximum(hi, hj)                 # shorter clone history of the pair at the forecast date
    band = np.full(len(C), "", dtype=object); buck = np.full(len(C), "", dtype=object)
    for lo, hi_, lab in FIXED:
        band[(C.phimin.values >= lo) & (C.phimin.values < hi_)] = lab
    for lo, hi_, lab in BUCK:
        buck[(h >= lo) & (h < hi_)] = lab
    C2 = C.assign(band=band, buck=buck)
    C2 = C2[(C2.band != "") & (C2.buck == "<2y")]              # the under-two-year pair-dates, at every age
    if C2.empty:
        continue
    per = ex.paired_sums(C2, ["band", "age", "i", "j"], [("cH", "cS")])
    acc = per if acc is None else acc.add(per, fill_value=0.0)
rows = []; rng = np.random.default_rng(7)
for (bd, age), per_all in acc.groupby(level=["band", "age"]):
    per = per_all[per_all["n_cH"] >= 12]
    if per.empty:
        continue
    f1 = per.index.get_level_values("i").values.astype(int); f2 = per.index.get_level_values("j").values.astype(int)
    funds = np.unique(np.concatenate([f1, f2])); nf = int(funds.max()) + 1
    se, sr = per["cH"].values, per["ref_cH"].values
    boots = []
    for _ in range(400):
        cnt = np.bincount(rng.choice(funds, len(funds), replace=True), minlength=nf)
        w = (cnt[f1] * cnt[f2]).astype(float)
        if (w * sr).sum() > 0:
            boots.append(0.5 * np.log((w * se).sum() / (w * sr).sum()))
    rows.append({"band": bd, "age": int(age), "pairs": len(per), "pair_dates": int(per["n_cH"].sum()),
                 "ratio": np.sqrt(se.sum() / sr.sum()), "lo": np.exp(np.percentile(boots, 2.5)), "hi": np.exp(np.percentile(boots, 97.5))})
R = pd.DataFrame(rows)
ordr = {b[2]: k for k, b in enumerate(FIXED)}
R = R.sort_values(["band", "age"], key=lambda s: s.map(ordr) if s.name == "band" else s)
print(f"L={L}: under-two-year pair-dates, holdings-corr / stream ratio by band and age (below 1.00 beats the stream)")
print(R.round(3).to_string(index=False))
out = f"{pooled}/young_aged_edge.csv"; R.to_csv(out, index=False, float_format="%.4f")
print("wrote", out)
