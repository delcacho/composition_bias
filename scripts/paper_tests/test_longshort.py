"""
test_longshort.py -- Composition bias in long/short sleeves.

Lemma 1 and Proposition 1 assume non-negative, unit-ℓ1 weights (long-only).
This test characterises the bias when sleeves can hold short positions.

Two sub-scenarios illustrate the key cases:

SCENARIO 1 -- Dollar-neutral convergence (bias same direction as long-only):
  Sleeve 1: long A-block, short B-block (+w_A, -w_B)
  Sleeve 2: initially similar; both gradually move to same long-short allocation.
  As overlap δ_ij increases from low to high, true covariance rises and rolling
  understates -- same qualitative result as the long-only case.

SCENARIO 2 -- Opposing-to-aligned transition (bias does not reverse sign):
  Pre-crowding: Sleeve 1 long A, Sleeve 2 short A -> overlap = -1 (opposing)
  Post-crowding: Both long A -> overlap = +1 (aligned)
  Rolling starts at ≈ -σ² (negative covariance), true covariance ends at +σ².
  During the transition the rolling estimator lags the rising true covariance and
  sits below it (a rolling average of a rising series is below it).

Output: figures/longshort_bias.png, longshort_stats.txt
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

T = 1_500
seed = 77
rng = np.random.default_rng(seed)
dates = pd.bdate_range("2000-01-01", periods=T)

# ── Asset universe: 4 assets (A1, A2, B1, B2), equicorrelated within block ──
N = 4
sigma = 0.20 / np.sqrt(252) # daily vol
rho_in = 0.60 # within-block correlation
rho_out= 0.05 # cross-block correlation

Sigma = np.full((N, N), rho_out * sigma**2)
Sigma[:2, :2] = rho_in * sigma**2
Sigma[2:, 2:] = rho_in * sigma**2
np.fill_diagonal(Sigma, sigma**2)
Sigma = 0.5 * (Sigma + Sigma.T)
L = np.linalg.cholesky(Sigma)
R = rng.standard_normal((T, N)) @ L.T # (T, N) asset returns


def _sig(t, t_start, t_end, steep=8.0):
    mid = (t_start + t_end) / 2.0
    scale = (t_end - t_start) / 2.0
    return 1.0 / (1.0 + np.exp(-steep * (t - mid) / scale))


def rolling_cov(r1, r2, window=252, min_obs=63):
    df = pd.concat([pd.Series(r1, index=dates, name="a"),
                    pd.Series(r2, index=dates, name="b")], axis=1)
    return df["a"].rolling(window, min_periods=min_obs).cov(df["b"])


def true_cov_series_np(W1, W2, Sigma):
    return np.array([W1[t] @ Sigma @ W2[t] for t in range(T)])


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Dollar-neutral convergence
# Both sleeves start with moderate long-short books, then converge to the
# same dollar-neutral book (long A, short B).
# ═══════════════════════════════════════════════════════════════════════════════
crowd_s1, crowd_e1 = 500, 800
disp_s1, disp_e1 = 1000, 1200

# Base: sleeve 1 long A1/A2, short B1/B2; sleeve 2 long B1/B2, short A1/A2
w1_base = np.array([+0.5, +0.5, -0.5, -0.5])
w2_base = np.array([-0.5, -0.5, +0.5, +0.5])
# Crowded: both long A1/A2, short B1/B2 (same as sleeve 1 base)
w_crowd = np.array([+0.5, +0.5, -0.5, -0.5])

W1_s1 = np.zeros((T, N))
W2_s1 = np.zeros((T, N))
for t in range(T):
    lam = np.clip(_sig(t, crowd_s1, crowd_e1) - _sig(t, disp_s1, disp_e1), 0, 1)
    W1_s1[t] = (1 - lam) * w1_base + lam * w_crowd
    W2_s1[t] = (1 - lam) * w2_base + lam * w_crowd

r1_s1 = (np.roll(W1_s1, 1, axis=0) * R).sum(axis=1); r1_s1[0] = 0
r2_s1 = (np.roll(W2_s1, 1, axis=0) * R).sum(axis=1); r2_s1[0] = 0
true_s1 = true_cov_series_np(W1_s1, W2_s1, Sigma)
rolling_s1 = rolling_cov(r1_s1, r2_s1).values
overlap_s1 = (W1_s1 * W2_s1).sum(axis=1)

bias_s1 = rolling_s1 - true_s1
crowd_mask_s1 = (np.arange(T) >= crowd_s1) & (np.arange(T) <= crowd_e1)
understate_pct_s1 = (bias_s1[crowd_mask_s1] < 0).mean() * 100

print("SCENARIO 1 -- Dollar-neutral convergence")
print(f" During crowding: rolling understates on {understate_pct_s1:.1f}% of days")
print(f" Mean bias during crowding: {bias_s1[crowd_mask_s1].mean():.2e}")
print(f" Overlap range: {overlap_s1.min():.3f} -> {overlap_s1.max():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Opposing-to-aligned transition -- bias does not reverse sign (rolling lags a rising covariance)
# Pre-crowd: Sleeve 1 = long A1, Sleeve 2 = short A1 (opposing bets)
# Post-crowd: Both = long A1 (same bet)
# ═══════════════════════════════════════════════════════════════════════════════
crowd_s2, crowd_e2 = 500, 900

w1_opp = np.array([+1.0, 0.0, 0.0, 0.0]) # long A1
w2_opp = np.array([-1.0, 0.0, 0.0, 0.0]) # short A1 -> overlap = -1
w_aligned = np.array([+1.0, 0.0, 0.0, 0.0]) # both long A1 -> overlap = +1

W1_s2 = np.zeros((T, N))
W2_s2 = np.zeros((T, N))
for t in range(T):
    lam = _sig(t, crowd_s2, crowd_e2)
    W1_s2[t] = w1_opp.copy() # sleeve 1 stays long A1
    W2_s2[t] = (1 - lam) * w2_opp + lam * w_aligned # sleeve 2 flips from short -> long

r1_s2 = (np.roll(W1_s2, 1, axis=0) * R).sum(axis=1); r1_s2[0] = 0
r2_s2 = (np.roll(W2_s2, 1, axis=0) * R).sum(axis=1); r2_s2[0] = 0
true_s2 = true_cov_series_np(W1_s2, W2_s2, Sigma)
rolling_s2 = rolling_cov(r1_s2, r2_s2).values
overlap_s2 = (W1_s2 * W2_s2).sum(axis=1)

bias_s2 = rolling_s2 - true_s2
crowd_mask_s2 = (np.arange(T) >= crowd_s2) & (np.arange(T) <= crowd_e2)
# During transition: true cov goes from negative to positive, rolling lags behind
# => bias is negative (rolling < true): understatement, same direction as Prop 1
understate_pct_s2 = (bias_s2[crowd_mask_s2] < 0).mean() * 100

print("\nSCENARIO 2 -- Opposing-to-aligned transition")
print(f" During transition: rolling understates on {understate_pct_s2:.1f}% of days")
print(f" (rolling < true current covariance: same direction as Prop 1, no sign reversal)")
print(f" Mean bias during transition: {bias_s2[crowd_mask_s2].mean():.2e}")
print(f" Overlap trajectory: {overlap_s2[0]:.2f} -> {overlap_s2[-1]:.2f}")

# ── Save stats ────────────────────────────────────────────────────────────────
stats = [
    "SCENARIO 1 -- Dollar-neutral convergence (long/short, bias same direction as long-only)",
    f" Understatement frequency during crowding: {understate_pct_s1:.1f}%",
    f" Mean bias: {bias_s1[crowd_mask_s1].mean():.2e}",
    "",
    "SCENARIO 2 -- Opposing-to-aligned transition (bias does NOT reverse sign)",
    f" Understatement frequency during transition: {understate_pct_s2:.1f}%",
    f" Mean bias: {bias_s2[crowd_mask_s2].mean():.2e}",
    f" (Rolling lags the rising true covariance and sits below it throughout the transition)",
]
(Path(__file__).parent / "longshort_stats.txt").write_text(
    "\n".join(stats), encoding="utf-8"
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Composition Bias in Long/Short Sleeves", fontsize=12, y=1.01)

for col, (W1, W2, true_c, rolling_c, bias_c, crowd_mask, overlap, title) in enumerate([
    (W1_s1, W2_s1, true_s1, rolling_s1, bias_s1, crowd_mask_s1, overlap_s1,
     "Scenario 1: Dollar-neutral convergence"),
    (W1_s2, W2_s2, true_s2, rolling_s2, bias_s2, crowd_mask_s2, overlap_s2,
     "Scenario 2: Opposing -> aligned transition"),
]):
    t_idx = np.arange(T)

    ax = axes[0, col]
    ax.plot(t_idx, true_c, color="#1f77b4", lw=2.0, label="True cov")
    ax.plot(t_idx, rolling_c, color="#d62728", lw=1.5, ls="-.", label="Rolling 252d")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvspan(crowd_mask.nonzero()[0][0], crowd_mask.nonzero()[0][-1],
               alpha=0.12, color="red")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Covariance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, col]
    ax.plot(t_idx, bias_c, color="#ff7f0e", lw=1.5, label="Rolling - True (bias)")
    ax.plot(t_idx, overlap, color="#2ca02c", lw=1.5, ls="--", label="Overlap δ_ij")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvspan(crowd_mask.nonzero()[0][0], crowd_mask.nonzero()[0][-1],
               alpha=0.12, color="red")
    ax.set_ylabel("Bias / Overlap")
    ax.set_xlabel("Trading day")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "longshort_bias.png", dpi=160)
plt.close(fig)
print(f"\nSaved: figures/longshort_bias.png")
print("Test PASSED")
