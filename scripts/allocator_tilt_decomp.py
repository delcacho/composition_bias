"""allocator_tilt_decomp.py -- is the blend's allocator premium diversification or a directional tilt?

Both stories give the same signature (a gain at matched volatility, concentrated on divergence days, tails
unchanged), so this attributes the daily blend-minus-stream return r_B - r_S to additive, exhaustive pieces
that sum exactly to it. With w = L*u (gross leverage L, unit-gross weights u, |u|=1) and rho = u.r, a
symmetric two-way split leaves no interaction bucket:

  scale       = (L_B - L_S) * (rho_B + rho_S)/2       leverage difference
  allocation  = (L_B + L_S)/2 * (u_B - u_S).r         sleeve-weight difference

and the allocation splits Brinson-style, with c_i = (L_B+L_S)/2 * (u_B,i - u_S,i) the net over/under-weight
of sleeve i:

  static tilt = mean(c_i) . r                          a standing factor bet (which sleeve, and its sign)
  timing      = (c_i - mean(c_i)) . r                  weight deviations covarying with returns

So r_B - r_S = scale + static tilt + timing, each with a paired block-bootstrap interval. Reading: the
premium in scale is diversification outright; a static tilt on trend/momentum with a reversal sign is a
directional bet; timing dominant and spread across episodes is the "earned where the records diverge"
story, weaker if a few months carry it. Reads allocator_multiasset.py's allocator_tilt_decomp*.parquet.

usage: python research/allocator_tilt_decomp.py [parquet] [--nboot=2000] [--block=63]
"""
import sys
import numpy as np
import pandas as pd

args = sys.argv[1:]
SRC = next((a for a in args if not a.startswith("--")), "allocator_tilt_decomp_L34_reb21.parquet")
NBOOT = next((int(a.split("=")[1]) for a in args if a.startswith("--nboot=")), 2000)
BLOCK = next((int(a.split("=")[1]) for a in args if a.startswith("--block=")), 63)

d = pd.read_parquet(SRC)
sleeves = [c[len("wc2_"):] for c in d.columns if c.startswith("wc2_")]
W_B = d[[f"wc2_{s}" for s in sleeves]].to_numpy()
W_S = d[[f"wst_{s}" for s in sleeves]].to_numpy()
R = d[[f"r_{s}" for s in sleeves]].to_numpy()
civ = d["composition_intensity"].to_numpy() if "composition_intensity" in d else np.full(len(d), np.nan)
dates = pd.to_datetime(d.index)

# weights are as of the close; they earn the NEXT day's returns, so lag them one day (run_phase convention)
wB = np.vstack([np.zeros((1, len(sleeves))), W_B[:-1]])
wS = np.vstack([np.zeros((1, len(sleeves))), W_S[:-1]])
live = (np.abs(wB).sum(1) > 1e-9) & (np.abs(wS).sum(1) > 1e-9) & np.isfinite(R).all(1)
wB, wS, r, civ, dates = wB[live], wS[live], R[live], civ[live], dates[live]

LB = np.abs(wB).sum(1); LS = np.abs(wS).sum(1)
uB = wB / LB[:, None]; uS = wS / LS[:, None]
rhoB = np.einsum("ti,ti->t", uB, r); rhoS = np.einsum("ti,ti->t", uS, r)
scale = (LB - LS) * (rhoB + rhoS) / 2.0
c = ((LB + LS) / 2.0)[:, None] * (uB - uS)          # net over/under-weight of each sleeve, average leverage
alloc = np.einsum("ti,ti->t", c, r)
cbar = c.mean(0)
static = r @ cbar
timing = alloc - static
total = np.einsum("ti,ti->t", wB - wS, r)           # = r_B - r_S
assert np.allclose(scale + static + timing, total, atol=1e-12), "attribution does not sum to the total"

bp = 1e4
rng = np.random.default_rng(7)


def boot_mean_ci(x):
    n = len(x); nb = int(np.ceil(n / BLOCK)); out = np.empty(NBOOT)
    for b in range(NBOOT):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        out[b] = x[idx].mean()
    return np.percentile(out, [2.5, 97.5])


print(f"days={len(total)}   mean leverage: blend {LB.mean():.2f}, stream {LS.mean():.2f}   "
      f"(bootstrap block {BLOCK}, {NBOOT} draws)")
print(f"\nADDITIVE ATTRIBUTION of r_B - r_S (bp/day), pieces sum to the total:")
for nm, x in [("total", total), ("scale", scale), ("  allocation", alloc),
              ("    static tilt", static), ("    timing", timing)]:
    lo, hi = boot_mean_ci(x)
    sh = f"{x.mean()/total.mean():+.0%}" if abs(total.mean()) > 1e-12 and nm != "total" else ""
    print(f"  {nm:16s} {x.mean()*bp:+.3f}  [{lo*bp:+.3f}, {hi*bp:+.3f}]  {sh}")

print("\nstatic tilt by sleeve (average net weight blend-minus-stream, and its bp/day), largest first:")
for i in np.argsort(-np.abs(cbar * r.mean(0))):
    print(f"  {sleeves[i]:18s} weight {cbar[i]:+.4f}   static {cbar[i]*r[:, i].mean()*bp:+.3f} bp/day")

# concentration of the timing piece across months
tim = pd.Series(timing, index=dates)
monthly = tim.groupby(tim.index.to_period("M")).sum()
tot = monthly.sum()
order = monthly.reindex(monthly.abs().sort_values(ascending=False).index)
eff = 1.0 / np.sum((monthly.abs() / monthly.abs().sum()) ** 2)
print(f"\ntiming concentration: top 5 months {order.iloc[:5].sum()/tot:.0%}, top 10 {order.iloc[:10].sum()/tot:.0%}"
      f" of the cumulative timing; effective months 1/HHI = {eff:.0f} of {len(monthly)}")
print("top timing months:", ", ".join(f"{str(m)} {v*bp:+.1f}bp" for m, v in order.iloc[:6].items()))

# does the premium concentrate on divergence days (top quintile of composition_intensity, the A10 state)?
if np.isfinite(civ).any():
    thr = np.nanquantile(civ, 0.80); dv = civ >= thr
    print(f"\non divergence days (top quintile composition_intensity), share {dv.mean():.0%} of days:")
    for nm, x in [("total", total), ("scale", scale), ("static tilt", static), ("timing", timing)]:
        print(f"  {nm:14s} divergence {x[dv].mean()*bp:+.2f}  other {x[~dv].mean()*bp:+.2f} bp/day")
print("\nread: premium in scale -> diversification outright; static tilt on trend/momentum with a reversal "
      "sign -> directional bet; timing dominant, spread and divergence-day concentrated -> diversification timing.")
pd.DataFrame([{"component": nm, "mean_bp": x.mean() * bp} for nm, x in
              [("total", total), ("scale", scale), ("static", static), ("timing", timing)]]
             ).to_csv(SRC.replace(".parquet", "_summary.csv"), index=False, float_format="%.4f")
out = SRC.replace(".parquet", "_result.csv")
pd.DataFrame({"month": monthly.index.astype(str), "timing_bp": monthly.values * bp}).to_csv(out, index=False, float_format="%.4f")
print("wrote", out, "and", SRC.replace(".parquet", "_summary.csv"))
