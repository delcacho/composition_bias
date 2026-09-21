"""
etf_sizing_admitted.py -- the sizing comparison (Appendix G, "Sizing the diagonal") restricted to the admitted
funds: the 369-fund figures the appendix quotes. Replicates etf_sizing_race.main()'s fund bootstrap on the
median log ratio of the holdings rule's error to the stream's (2000 draws, seed 7), h=21, both stream
halflives, by persistence band, on the forecast RMSE and on the sized-series RMSE, plus the median
leverage turnover per refresh. Reads etf_results/sizing_funds.csv (the per-fund output of
etf_sizing_race.py) and research/etf_universe/universe_sample.csv (the in_sample admission flag).
Writes etf_results/sizing_admitted_summary.csv. The all-scored set (578 funds) is kept alongside.
"""
import numpy as np, pandas as pd
F = pd.read_csv("etf_results/sizing_funds.csv")
u = pd.read_csv("research/etf_universe/universe_sample.csv", dtype=str)
adm = set(u[u.in_sample.astype(str).str.lower().isin(("true", "1", "yes", "y"))].ticker.str.upper())
F["tk"] = F.ticker.str.upper()
BANDS = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, ">0.95")]
rng = np.random.default_rng(7)
rows = []
for label, Fx in (("all_scored", F), ("admitted", F[F.tk.isin(adm)])):
    Fh = Fx[Fx.h == 21]
    for hl in (63, 252):
        Hd = Fh[Fh.rule == f"holdings{hl}"].set_index("ticker"); Sd = Fh[Fh.rule == f"stream{hl}"].set_index("ticker")
        common = Hd.index.intersection(Sd.index)
        for metric in ("rmse_forecast", "rmse_sizing"):
            lr = np.log(Hd.loc[common, metric] / Sd.loc[common, metric])
            band_of = pd.cut(Hd.loc[common, "phibar_neutral_63"], [b[0] for b in BANDS] + [1.01], labels=[b[2] for b in BANDS], right=False)
            for band in [b[2] for b in BANDS] + ["all"]:
                x = (lr if band == "all" else lr[band_of == band]).dropna().values
                if len(x) < 3:
                    continue
                meds = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
                rows.append({"set": label, "stream_halflife": hl, "metric": metric, "band": band, "n": len(x),
                             "median_log_ratio": np.median(x), "ci_lo": np.percentile(meds, 2.5), "ci_hi": np.percentile(meds, 97.5)})
    for rule, t in Fh.groupby("rule").turnover.median().items():
        rows.append({"set": label, "stream_halflife": np.nan, "metric": "turnover_per_refresh", "band": rule,
                     "n": int((Fh.rule == rule).sum()), "median_log_ratio": t, "ci_lo": np.nan, "ci_hi": np.nan})
out = pd.DataFrame(rows)
out.to_csv("etf_results/sizing_admitted_summary.csv", index=False, float_format="%.4f")
a = out[out.set == "admitted"]
print(a[a.metric != "turnover_per_refresh"].to_string(index=False))
print(a[a.metric == "turnover_per_refresh"].to_string(index=False))
