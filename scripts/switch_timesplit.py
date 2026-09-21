"""Time-split scoring of the switch (Online Appendix G). Calibrate crossing ages per fixed
band on 2018-2022, score the switch on 2023-2026, full-window halflife-63 chunks. Reads existing chunks only.
Two rules are scored side by side on 2023-2026: the CALIBRATED one (ages read from 2018-2022 per band) and the
ARTICLE's rule as written (a month below 0.75, a quarter at or above it), so the reader can see whether the rule
recommended is the rule that was tested and what a wrong age costs."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research")
import etf_race as er
import etf_staleness_experiment as ex
import sys
d = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "etf_results/experiment/pooled_neutral_all_L63_fullwin"
idx = er.load_returns().index
FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
AGES = [0, 21, 63, 108]
def band_of(ph):
    b = np.full(len(ph), "", dtype=object)
    for lo, hi, lab in FIXED:
        b[(ph >= lo) & (ph < hi)] = lab
    return b
# ---- pass 1: calibration on 2018-2022, pooled RMSE ratio holdings corr / stream by band x age
cal = None; test_rows = []
for C in ex.chunks(d):
    C = C[C.age >= 0]
    if C.empty:
        continue
    yr = pd.DatetimeIndex(idx[C.date_ix.values]).year
    C = C.assign(band=band_of(C.phimin.values), year=yr); C = C[C.band != ""]
    A = C[C.year <= 2022]
    if len(A):
        per = ex.paired_sums(A, ["band", "age", "i", "j"], [("cH", "cS")])
        cal = per if cal is None else cal.add(per, fill_value=0.0)
g = cal.groupby(level=["band", "age"]).sum()
ratio = np.sqrt(g["cH"] / g["ref_cH"]).unstack("age")[AGES]
print("calibration 2018-2022, holdings corr / stream:"); print(ratio.round(3).to_string())
cross = {}
for lo, hi, lab in FIXED:
    r = ratio.loc[lab]
    cross[lab] = next((a for a in AGES if r[a] >= 1.0), 10**9)
print("crossing ages (calibrated on 2018-2022):", cross)
article = {"<0.75": 21, "0.75-0.85": 63, "0.85-0.95": 63, ">0.95": 63}      # the article's rule as written
print("crossing ages (article's rule as written):", article)
# ---- pass 2: score on 2023-2026: switch = holdings corr if age < crossing age of the band, else stream
acc = None
for C in ex.chunks(d):
    C = C[C.age >= 0]
    if C.empty:
        continue
    yr = pd.DatetimeIndex(idx[C.date_ix.values]).year
    C = C.assign(band=band_of(C.phimin.values), year=yr); C = C[(C.band != "") & (C.year >= 2023)]
    if C.empty:
        continue
    use_h = C.age.values < np.array([cross[b] for b in C.band.values])
    use_a = C.age.values < np.array([article[b] for b in C.band.values])
    both = np.isfinite(C.cH.values) & np.isfinite(C.cS.values)
    sw = np.where(use_h, C.cH.values, C.cS.values); sa = np.where(use_a, C.cH.values, C.cS.values)
    C2 = C.assign(cW=np.where(both, sw, np.nan), cA=np.where(both, sa, np.nan))
    # paired_sums keys the reference by the estimator name, so each estimator gets ONE reference (the stream); the
    # cell-level better record is min(holdings ratio, 1) computed below, not a pointwise best-of-both
    per = ex.paired_sums(C2, ["band", "age", "i", "j"], [("cW", "cS"), ("cA", "cS"), ("cH", "cS")])
    acc = per if acc is None else acc.add(per, fill_value=0.0)
rng = np.random.default_rng(7); rows = []
for (bd, age), per_all in acc.groupby(level=["band", "age"]):
    per = per_all[per_all["n_cW"] >= 12]
    if per.empty:
        continue
    f1 = per.index.get_level_values("i").values.astype(int); f2 = per.index.get_level_values("j").values.astype(int)
    funds = np.unique(np.concatenate([f1, f2])); nf = int(funds.max()) + 1
    draws = [np.bincount(rng.choice(funds, len(funds), replace=True), minlength=nf) for _ in range(500)]
    out = {"band": bd, "age": age, "pairs": len(per), "pair_dates": int(per["n_cW"].sum())}
    for est, ref, name in [("cW", "cS", "switch/stream"), ("cA", "cS", "article/stream"), ("cH", "cS", "holdings/stream")]:
        se, sr = per[est].values, per[f"ref_{est}"].values
        boots = [0.5 * np.log((cnt[f1] * cnt[f2] * se).sum() / (cnt[f1] * cnt[f2] * sr).sum()) for cnt in draws]
        out[name] = np.sqrt(se.sum() / sr.sum()); out[name + " lo"] = np.exp(np.percentile(boots, 2.5)); out[name + " hi"] = np.exp(np.percentile(boots, 97.5))
    out["better record"] = min(out["holdings/stream"], 1.0); out["switch/better"] = out["switch/stream"] / out["better record"]
    out["article/better"] = out["article/stream"] / out["better record"]
    rows.append(out)
R = pd.DataFrame(rows); order = {b[2]: k for k, b in enumerate(FIXED)}
R = R.sort_values(["age", "band"], key=lambda s: s.map(order) if s.name == "band" else s)
print("\nscoring 2023-2026 (full-window pairs, halflife 63):"); print(R.round(3).to_string(index=False))
R.to_csv(f"{d}/switch_timesplit.csv", index=False, float_format="%.6f"); print("wrote", f"{d}/switch_timesplit.csv")
