"""allocator_es_robustness.py -- is the expected-shortfall attribution one crisis, or stable?

The deepest tail has few effective episodes and the monthly phases overlap, so one can ask
whether the pair ranking of Exhibit A10 is carried by a single crisis. Two checks, both re-running the
full pathwise ES decomposition (allocator_es_attribution.py) on a perturbed sample:

  LEAVE-ONE-YEAR-OUT: drop each calendar year's rebalances in turn and recompute. If the total ES gap
    and the top pairs survive dropping 2008 and 2020, the result is not one crisis.
  BLOCK BOOTSTRAP: resample rebalances in contiguous blocks (preserving the overlap structure) and
    recompute, for a confidence interval on the total attributed gap and the leading pairs.

usage: python research/allocator_es_robustness.py [inputs.pkl] [--ngrid=41] [--alpha=0.05]
                                                  [--nboot=0] [--block=12]
"""
import sys
import pickle
import numpy as np
import pandas as pd
from allocator_es_attribution import es_attribution

args = sys.argv[1:]
SRC = next((a for a in args if not a.startswith("--")), "allocator_attribution_inputs_L34_reb21.pkl")
NG = next((int(a.split("=")[1]) for a in args if a.startswith("--ngrid=")), 41)
ALPHA = next((float(a.split("=")[1]) for a in args if a.startswith("--alpha=")), 0.05)
NBOOT = next((int(a.split("=")[1]) for a in args if a.startswith("--nboot=")), 0)
BLOCK = next((int(a.split("=")[1]) for a in args if a.startswith("--block=")), 12)

with open(SRC, "rb") as f:
    blob = pickle.load(f)
names = blob["sleeves"]; recs = blob["records"]
years = np.array([pd.Timestamp(r["date"]).year for r in recs])


def top_pairs(res, k=3):
    t = res[ALPHA]["table"].copy()
    t["abs"] = t["es_attribution_bp"].abs()
    return t.sort_values("abs", ascending=False).head(k)


# baseline
base, _ = es_attribution(blob, n_grid=NG, alphas=(ALPHA,))
base_gap = base[ALPHA]["direct_bp"]
lead = top_pairs(base, 3)
lead_keys = [(r.sleeve_i, r.sleeve_j) for r in lead.itertuples()]
print(f"BASELINE (alpha={ALPHA}, n_grid={NG}): ES gap {base_gap:+.2f} bp")
print("  leading pairs:", ", ".join(f"{a}/{b} {v:+.2f}" for (a, b), v in
                                     zip(lead_keys, lead.es_attribution_bp)))


def pair_val(res, a, b):
    t = res[ALPHA]["table"]
    m = t[((t.sleeve_i == a) & (t.sleeve_j == b)) | ((t.sleeve_i == b) & (t.sleeve_j == a))]
    return float(m.es_attribution_bp.iloc[0]) if len(m) else np.nan


# ---- leave-one-year-out ----
print(f"\nLEAVE-ONE-YEAR-OUT ({len(np.unique(years))} years): total ES gap and the three leading pairs")
rows = []
for y in np.unique(years):
    sub = {"sleeves": names, "records": [r for r, yr in zip(recs, years) if yr != y]}
    r, _ = es_attribution(sub, n_grid=NG, alphas=(ALPHA,))
    rows.append({"dropped": int(y), "gap_bp": r[ALPHA]["direct_bp"],
                 **{f"{a}/{b}": pair_val(r, a, b) for a, b in lead_keys}})
loyo = pd.DataFrame(rows)
gcol = loyo["gap_bp"]
print(f"  gap range [{gcol.min():+.2f}, {gcol.max():+.2f}] bp  (baseline {base_gap:+.2f}); "
      f"min when dropping {int(loyo.loc[gcol.idxmin(),'dropped'])}, max {int(loyo.loc[gcol.idxmax(),'dropped'])}")
for (a, b) in lead_keys:
    c = loyo[f"{a}/{b}"]
    print(f"  {a}/{b:14s} range [{c.min():+.2f}, {c.max():+.2f}]  sign stable: {bool((c > 0).all() or (c < 0).all())}")
loyo.to_csv(SRC.replace("allocator_attribution_inputs", "allocator_es_loyo").replace(".pkl", ".csv"),
            index=False, float_format="%.3f")

# ---- optional block bootstrap ----
if NBOOT > 0:
    rng = np.random.default_rng(7); n = len(recs); nb = int(np.ceil(n / BLOCK))
    gaps = np.empty(NBOOT); pvals = {k: np.empty(NBOOT) for k in lead_keys}
    for bi in range(NBOOT):
        st = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in st]) % n
        sub = {"sleeves": names, "records": [recs[i] for i in idx[:n]]}
        r, _ = es_attribution(sub, n_grid=NG, alphas=(ALPHA,))
        gaps[bi] = r[ALPHA]["direct_bp"]
        for k in lead_keys:
            pvals[k][bi] = pair_val(r, *k)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"\nBLOCK BOOTSTRAP ({NBOOT} draws, block {BLOCK}): total ES gap {gaps.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] bp")
    for k in lead_keys:
        l, h = np.percentile(pvals[k], [2.5, 97.5])
        print(f"  {k[0]}/{k[1]:14s} {pvals[k].mean():+.2f} [{l:+.2f}, {h:+.2f}]")
print("\ndone.")
