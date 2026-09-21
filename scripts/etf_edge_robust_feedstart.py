"""Robustness: is the sub-two-year fresh-holdings edge driven by funds whose feed entry
coincides with the absolute feed start (the ones most likely to predate the daily-holdings
feed, so their short clone history is truncation rather than youth)? Recompute the band-by-history
table on all funds and on the subset that drops any pair containing a feed-start-pinned fund (first
daily holdings within BUF trading days of the panel's earliest feed entry), in one pass over the
existing chunks. Usage: python etf_edge_robust_feedstart.py [252|63] [BUF]"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex

L = int(sys.argv[1]) if len(sys.argv) > 1 else 252
BUF = int(sys.argv[2]) if len(sys.argv) > 2 else 63     # trading days from the earliest feed entry
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
hist_days = {}
for t in tick:
    f = first.get(t, pd.NaT)
    hist_days[t] = int(idx.searchsorted(f)) if pd.notna(f) else -1

# feed-start-pinned funds: first within BUF trading days of the earliest feed entry among panel funds
valid = np.array([hist_days[t] for t in tick if hist_days[t] >= 0])
base = int(valid.min())
pinned = set(int(k) for k, t in enumerate(tick) if hist_days[t] >= 0 and hist_days[t] <= base + BUF)
print(f"L={L} BUF={BUF}: earliest feed entry {idx[base].date()}; feed-start-pinned funds dropped: "
      f"{len(pinned)} of {len(tick)} (first <= {idx[min(base+BUF, len(idx)-1)].date()})")

pooled = f"{root}/pooled_neutral_all{sfx}"
FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
BUCK = [(0, 504, "<2y"), (504, 1008, "2-4y"), (1008, BACK, f"4y-{BACK}d"), (BACK, 10**9, f">={BACK}d (full window)")]

def accumulate(drop_pinned):
    acc = None
    for C in ex.chunks(pooled):
        C = C[C.age == 0]
        if C.empty:
            continue
        ii, jj = C.i.values, C.j.values
        if drop_pinned:
            keep = np.array([(i not in pinned) and (j not in pinned) for i, j in zip(ii, jj)])
            C = C[keep]; ii, jj = C.i.values, C.j.values
            if C.empty:
                continue
        hi = np.array([hist_days[tick[i]] for i in ii]); hj = np.array([hist_days[tick[j]] for j in jj])
        h = C.date_ix.values - np.maximum(hi, hj)
        band = np.full(len(C), "", dtype=object); buck = np.full(len(C), "", dtype=object)
        for lo, hi_, lab in FIXED:
            band[(C.phimin.values >= lo) & (C.phimin.values < hi_)] = lab
        for lo, hi_, lab in BUCK:
            buck[(h >= lo) & (h < hi_)] = lab
        C2 = C.assign(band=band, buck=buck); C2 = C2[(C2.band != "") & (C2.buck != "")]
        per = ex.paired_sums(C2, ["band", "buck", "i", "j"], [("cH", "cS")])
        acc = per if acc is None else acc.add(per, fill_value=0.0)
    return acc

def table(acc, tag):
    rows = []
    rng = np.random.default_rng(7)
    for (bd, bk), per_all in acc.groupby(level=["band", "buck"]):
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
        rows.append({"set": tag, "band": bd, "history": bk, "pairs": len(per), "pair_dates": int(per["n_cH"].sum()),
                     "ratio": np.sqrt(se.sum() / sr.sum()),
                     "lo": np.exp(np.percentile(boots, 2.5)), "hi": np.exp(np.percentile(boots, 97.5))})
    R = pd.DataFrame(rows)
    order = {b[2]: k for k, b in enumerate(BUCK)}; R["o"] = R.history.map(order)
    return R.sort_values(["band", "o"]).drop(columns="o")

full = table(accumulate(False), "all")
excl = table(accumulate(True), f"drop_pinned({len(pinned)})")
both = pd.concat([full, excl], ignore_index=True)
out = f"{pooled}/edge_robust_feedstart.csv"
both.to_csv(out, index=False, float_format="%.4f")
piv = both[both.history == "<2y"].pivot(index="band", columns="set", values="ratio")
print("\n<2y fresh holdings-corr ratio to stream (lower = holdings win):")
print(piv.to_string())
print(both.to_string(index=False)); print("wrote", out)
