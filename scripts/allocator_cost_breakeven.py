"""Net Sharpe versus transaction cost for the multi-asset allocation, with the breakeven against the
return stream marked. Answers the practitioner question the cost grid buries: at what cost does the
blend stop being worth its turnover against the return stream.
Reads the shipped cost grid; writes figures/allocator_cost_breakeven.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

CSV = "public_repo/data/commodity/allocator_multiasset_costs_L34_reb21.csv"
OUT = "figures/allocator_cost_breakeven.png"
BPS = np.array([0, 2, 5, 10, 20])

d = pd.read_csv(CSV).set_index("estimator")
def curve(e): return np.array([d.loc[e, f"sharpe_{b}bp"] for b in BPS])

# fine grid by linear interpolation between the shipped cost points (Sharpe is ~linear in cost)
xf = np.linspace(0, 20, 400)
def fine(e): return np.interp(xf, BPS, curve(e))

def breakeven(a, b):
    diff = a - b
    for i in range(len(BPS) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            return BPS[i] + diff[i] / (diff[i] - diff[i + 1]) * (BPS[i + 1] - BPS[i])
    return None

series = [
    ("holdings correlation blend", "c2_daily", "#1f4e79", 2.4),
    ("holdings covariance, no blend", "pic_daily", "#2a9d8f", 1.8),
    ("return stream", "stream", "#8a8a8a", 1.8),
    ("inverse volatility, no correlations", "invvol", "#e07b39", 1.8),
]
be_st = breakeven(curve("c2_daily"), curve("stream"))
be_pic = breakeven(curve("pic_daily"), curve("stream"))
be_iv = breakeven(curve("c2_daily"), curve("invvol"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "figure.dpi": 200,
})
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for label, key, color, lw in series:
    ax.plot(xf, fine(key), color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.scatter(BPS, curve(key), color=color, s=16, zorder=4)
    yend = curve(key)[-1]
    ax.annotate(label, xy=(20.2, yend), va="center", ha="left", color=color,
                fontsize=9.5, fontweight="bold" if key == "c2_daily" else "normal")

# breakeven markers: the costs at which the blend stops beating the naive book and the stream,
# and at which holdings-used-whole stops beating the stream
for be, key, color, txt in [(be_iv, "c2_daily", "#e07b39", f"blend beats inverse-vol\nto {be_iv:.1f} bp"),
                            (be_st, "c2_daily", "#1f4e79", f"blend beats stream\nto {be_st:.0f} bp"),
                            (be_pic, "pic_daily", "#2a9d8f", f"holdings alone\nto {be_pic:.0f} bp")]:
    y = np.interp(be, BPS, curve(key))
    ax.plot([be, be], [0, y], color=color, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.scatter([be], [y], facecolor="white", edgecolor=color, s=42, lw=1.4, zorder=5)
    ax.annotate(txt, xy=(be, y), xytext=(be + 0.5, y + 0.045), fontsize=8.6,
                color=color, ha="left", va="bottom")

ax.set_xlim(0, 24.5)
ax.set_ylim(0.26, 0.71)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel("transaction cost (basis points per unit notional traded)")
ax.set_ylabel("Sharpe ratio, net of cost")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}; blend vs inverse-vol {be_iv:.2f} bp, blend vs stream {be_st:.2f} bp, holdings-alone vs stream {be_pic:.2f} bp")
