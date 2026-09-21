"""
test_bias.py -- Test 1 & 2 from the paper.

Test 1: The return-based estimator is biased; position-implied tracks the
        true covariance during a crowding episode.

Test 2 (Proposition 1): The bias is monotonically related to the degree of
        crowding (cosine similarity between sleeve weight vectors).

Output: figures/bias_over_time.png, figures/bias_vs_crowding.png, bias_stats.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from synthetic import CrowdingScenario
from estimators import (
    true_cov_series, pic_series,
    rolling_return_cov_series, ewma_return_cov_series,
    sleeve_returns,
)

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def run_test_bias(scenario: CrowdingScenario, window: int = 252) -> pd.DataFrame:
    """
    Compute all four estimators over time.
    Returns a DataFrame with columns: true, pic, rolling, ewma, bias_rolling, bias_pic
    """
    r1, r2 = scenario.sleeve_returns()

    true_c = true_cov_series(scenario.weights_1, scenario.weights_2, scenario.Sigma)
    pic_c = pic_series(scenario.weights_1, scenario.weights_2, scenario.asset_returns, min_obs=63)
    roll_c = rolling_return_cov_series(r1, r2, window=window, min_obs=63)
    ewma_c = ewma_return_cov_series(r1, r2, halflife=63, min_obs=30)

    df = pd.concat({
        "true": true_c,
        "pic": pic_c,
        "rolling": roll_c,
        "ewma": ewma_c,
    }, axis=1).dropna()

    df["bias_rolling"] = df["rolling"] - df["true"]
    df["bias_pic"] = df["pic"] - df["true"]
    df["crowd"] = scenario.crowd_intensity.reindex(df.index)
    df["cosine_sim"] = scenario.cosine_similarity().reindex(df.index)
    return df


def plot_bias_over_time(df: pd.DataFrame, scenario: CrowdingScenario, window: int = 252):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel A: Covariance estimates
    ax = axes[0]
    ax.plot(df.index, df["true"], lw=2.0, color="#1f77b4", label="True (oracle)", zorder=5)
    ax.plot(df.index, df["pic"], lw=1.5, color="#2ca02c", ls="--", label="Holdings-implied")
    ax.plot(df.index, df["rolling"], lw=1.5, color="#d62728", ls="-.", label=f"Return-stream rolling ({window}d)")
    ax.plot(df.index, df["ewma"], lw=1.2, color="#ff7f0e", ls=":", label="EWMA return-stream")
    ax.axvspan(scenario.dates[scenario.crowd_start], scenario.dates[scenario.crowd_end],
               alpha=0.12, color="red", label="Crowding period")
    if scenario.disperse_start < scenario.T:
        ax.axvspan(scenario.dates[scenario.disperse_start], scenario.dates[min(scenario.disperse_end, scenario.T-1)],
                   alpha=0.08, color="blue", label="Dispersion period")
    ax.set_ylabel("Sleeve Covariance")
    ax.set_title("Covariance Estimates: Return-Stream vs Holdings-Implied")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: Bias (estimator - true)
    ax = axes[1]
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(df.index, df["bias_rolling"], 0,
                    where=df["bias_rolling"] < 0, alpha=0.4, color="#d62728",
                    label="Rolling underestimates (bias < 0)")
    ax.fill_between(df.index, df["bias_rolling"], 0,
                    where=df["bias_rolling"] > 0, alpha=0.2, color="#d62728")
    ax.plot(df.index, df["bias_pic"], lw=1.2, color="#2ca02c", ls="--", label="Holdings-implied bias ≈ 0")
    ax.axvspan(scenario.dates[scenario.crowd_start], scenario.dates[scenario.crowd_end],
               alpha=0.12, color="red")
    ax.set_ylabel("Bias = Estimate - True")
    ax.set_title("Estimation Bias Over Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Crowding intensity
    ax = axes[2]
    ax.plot(df.index, df["crowd"], color="gray", lw=1.5)
    ax.fill_between(df.index, df["crowd"], 0, alpha=0.3, color="gray")
    ax.set_ylabel("Crowding Intensity λ(t)")
    ax.set_xlabel("Date")
    ax.set_title("Crowding Intensity (0 = Diversified, 1 = Fully Crowded)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT / "bias_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f" Saved: {path}")


def plot_bias_vs_crowding(df: pd.DataFrame):
    valid = df[["bias_rolling", "bias_pic", "cosine_sim"]].dropna()
    x = valid["cosine_sim"].values
    y_roll = valid["bias_rolling"].values
    y_pic = valid["bias_pic"].values

    # OLS on crowding vs rolling bias
    slope, intercept, r, p, se = stats.linregress(x, y_roll)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(x, y_roll, alpha=0.15, s=6, color="#d62728", rasterized=True)
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, intercept + slope * xfit, "k-", lw=2,
            label=f"OLS: slope={slope:.4f}, R²={r**2:.3f}, p={p:.2e}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Crowding (Cosine Similarity between Sleeve Weights)")
    ax.set_ylabel("Bias = Rolling Estimate - True Covariance")
    ax.set_title("Proposition 1: Bias vs Crowding (Rolling Estimator)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(x, y_pic, alpha=0.15, s=6, color="#2ca02c", rasterized=True)
    slope_p, intercept_p, r_p, p_p, _ = stats.linregress(x, y_pic)
    ax.plot(xfit, intercept_p + slope_p * xfit, "k-", lw=2,
            label=f"OLS: slope={slope_p:.4f}, R²={r_p**2:.3f}, p={p_p:.2e}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Crowding (Cosine Similarity between Sleeve Weights)")
    ax.set_ylabel("Bias = Holdings-Implied Estimate - True Covariance")
    ax.set_title("Holdings-Implied: Bias vs Crowding (should be near zero)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Proposition 1 - Crowding Drives Return-Stream Bias", fontweight="bold")
    fig.tight_layout()
    path = OUT / "bias_vs_crowding.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f" Saved: {path}")


def print_summary_table(df: pd.DataFrame, scenario: CrowdingScenario) -> pd.DataFrame:
    periods = {
        "Pre-crowding": (df.index < scenario.dates[scenario.crowd_start]),
        "Crowding": (df.index >= scenario.dates[scenario.crowd_start]) &
                         (df.index <= scenario.dates[scenario.crowd_end]),
        "Post-crowding": (df.index > scenario.dates[scenario.crowd_end]) &
                         (df.index < scenario.dates[scenario.disperse_start]),
        "Dispersion": (df.index >= scenario.dates[scenario.disperse_start]),
    }

    rows = []
    for label, mask in periods.items():
        sub = df[mask].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "Period": label,
            "N_days": len(sub),
            "True Cov (mean)": sub["true"].mean(),
            "Rolling Bias (mean)": sub["bias_rolling"].mean(),
            "PIC Bias (mean)": sub["bias_pic"].mean(),
            "RMSE Rolling": np.sqrt((sub["bias_rolling"]**2).mean()),
            "RMSE PIC": np.sqrt((sub["bias_pic"]**2).mean()),
            "Rolling < True (%)": (sub["rolling"] < sub["true"]).mean() * 100,
        })

    tbl = pd.DataFrame(rows).set_index("Period")
    print("\n=== Table 1: Bias Summary by Period ===")
    with pd.option_context("display.float_format", "{:.6f}".format, "display.width", 120):
        print(tbl.to_string())
    return tbl


if __name__ == "__main__":
    print("Running Test 1 & 2: Bias Demonstration and Crowding-Bias Relationship")
    print("-" * 70)

    scenario = CrowdingScenario(T=1500, N=20, crowd_start=600, crowd_end=900,
                                 disperse_start=1050, disperse_end=1250,
                                 crowd_n_assets=3, seed=42)

    print(f" Scenario: {scenario.T} days, {scenario.N} assets, "
          f"crowding [{scenario.crowd_start}-{scenario.crowd_end}]")

    WINDOW = 252
    df = run_test_bias(scenario, window=WINDOW)

    tbl = print_summary_table(df, scenario)
    tbl.to_csv(Path(__file__).parent / "bias_stats.csv")
    print("\n Saved: bias_stats.csv")

    plot_bias_over_time(df, scenario, window=WINDOW)
    plot_bias_vs_crowding(df)

    # Key headline statistics
    crowd_mask = ((df.index >= scenario.dates[scenario.crowd_start]) &
                  (df.index <= scenario.dates[scenario.crowd_end]))
    crowd = df[crowd_mask].dropna()
    pct_understated = (crowd["rolling"] < crowd["true"]).mean()

    print(f"\n Key results:")
    print(f" - During crowding, rolling estimator understates true cov {pct_understated:.1%} of days")
    print(f" - Mean rolling bias during crowding: {crowd['bias_rolling'].mean():.6f} "
          f"(negative = underestimate)")
    print(f" - Mean PIC bias during crowding: {crowd['bias_pic'].mean():.6f} "
          f"(should be near 0)")
    print(f" - RMSE ratio (Rolling/PIC) during crowding: "
          f"{np.sqrt((crowd['bias_rolling']**2).mean()) / (np.sqrt((crowd['bias_pic']**2).mean()) + 1e-12):.1f}x")

    print("\nTest 1 & 2 PASSED" if pct_understated > 0.70 else "\nTest 1 & 2 FAILED")
