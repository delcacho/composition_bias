"""The pooled staleness comparison restricted to pairs whose shorter clone history covers the full estimation window
(>= BACK = 6L trading days of holdings before the forecast date), so the stream and the asset covariance see the
same number of days. Filters the existing pooled chunks into <pooled>_fullwin/chunks and runs the unchanged
analyze(). Also prints the frozen-band fresh ratio by calendar period x history (is it history or the period?).
Usage: python fullwindow_race.py [252|63] [class]
       a class name (allocation, longonly, longshort, mf) restricts the run to that class's own chunks and writes
       <pooled>_fullwin_<class> (the article's allocation-only row)."""
import os, sys, glob, shutil
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex

L = int(sys.argv[1]) if len(sys.argv) > 1 else 63
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
BACK = 6 * L
root = "etf_results/experiment"; sfx = "" if L == 252 else f"_L{L}"
classes = ["allocation", "mf", "longshort", "longonly"] if ONLY is None else [ONLY]
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
idx = er.load_returns().index
hist = np.array([int(idx.searchsorted(first.get(t, pd.NaT))) if pd.notna(first.get(t, pd.NaT)) else 10**9 for t in tick])
pooled = f"{root}/pooled_neutral_all{sfx}" if ONLY is None else f"{root}/{ONLY}_neutral_all{sfx}"
out = f"{root}/pooled_neutral_all{sfx}_fullwin" + ("" if ONLY is None else f"_{ONLY}")
if os.path.isdir(f"{out}/chunks"):
    shutil.rmtree(f"{out}/chunks")
os.makedirs(f"{out}/chunks", exist_ok=True)
kept = total = 0; diag_rows = []
for f in sorted(glob.glob(f"{pooled}/chunks/*.parquet")):
    C = pd.read_parquet(f)
    if f.endswith("_diag.parquet"):
        ok = (C.date_ix.values - hist[C.i.values]) >= BACK if len(C) else np.zeros(0, bool)
    else:
        ok = (C.date_ix.values - np.maximum(hist[C.i.values], hist[C.j.values])) >= BACK if len(C) else np.zeros(0, bool)
        if len(C):
            band = np.where(C.phimin.values >= 0.95, ">0.95", np.where(C.phimin.values < 0.75, "<0.75", "mid"))
            yr = pd.DatetimeIndex(idx[C.date_ix.values]).year
            A0 = C[(C.age == 0)]
            diag_rows.append(pd.DataFrame({"year": yr[C.age.values == 0], "band": band[C.age.values == 0],
                                           "full": ok[C.age.values == 0], "cH": A0.cH.values, "cS": A0.cS.values}))
    total += len(C); kept += int(ok.sum())
    C[ok].to_parquet(f"{out}/chunks/{os.path.basename(f)}", index=False)
print(f"L={L}, BACK={BACK}: kept {kept:,} of {total:,} rows with the shorter clone history >= BACK")
D = pd.concat(diag_rows)
D["period"] = pd.cut(D.year, [2017, 2019, 2022, 2026], labels=["2018-19", "2020-22", "2023-26"])
m = np.isfinite(D.cH) & np.isfinite(D.cS)
G = D[m].groupby(["band", "period", "full"], observed=True).agg(n=("cH", "size"), sH=("cH", "sum"), sS=("cS", "sum"))
G["ratio"] = np.sqrt(G.sH / G.sS)
print("fresh holdings-corr ratio by band x period x full-window (pooled, no intervals; the analysis below has them)")
print(G[["n", "ratio"]].round(3).to_string())
ex.analyze(out, "neutral", 21, cls="pooled")
print("wrote", out)
