"""
validate_french.py -- correctness check for the equity factor sleeves. Correlates the
Momentum / Value / Quality daily long-short returns with Kenneth French's daily factors
(Mom / HML / RMW), downloaded via pandas_datareader (public Dartmouth library, no WRDS needed).

A high correlation (about 0.85 or more) indicates that the Compustat-built factors track the standard ones.

Usage: python research/equity_factors/validate_french.py
"""
import os
import sys
import numpy as np
import pandas as pd

OUT = os.path.join("research", "equity_factors", "_out")
FNAME = sys.argv[1] if len(sys.argv) > 1 else "ff_annual_returns.csv" # the French-validated build (A13); sleeve_returns.csv is the monthly build

PAIRS = [("momentum", "Mom", "F-F_Momentum_Factor_daily"),
         ("value", "HML", "F-F_Research_Data_5_Factors_2x3_daily"),
         ("quality", "RMW", "F-F_Research_Data_5_Factors_2x3_daily")]


ROWS = [] # Exhibit A13 rows -> _out/validate_french.csv


def main():
    ours = pd.read_csv(os.path.join(OUT, FNAME), index_col=0, parse_dates=True)
    start, end = ours.index.min(), ours.index.max()

    import pandas_datareader.data as web
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0] / 100.0
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0] / 100.0
    mom.columns = [c.strip() for c in mom.columns]
    french = pd.concat([ff5, mom], axis=1)

    print(f"sleeves: {start.date()}..{end.date()} French cols: {list(french.columns)}\n")
    print(f"{'sleeve':9s} {'French':6s} {'corr':>7s} {'sleeve Sharpe':>13s} {'French Sharpe':>14s}")
    for sleeve, fcol, _ in PAIRS:
        a = ours[sleeve].dropna()
        b = french[fcol].dropna()
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        j.columns = ["ours", "french"]
        corr = j["ours"].corr(j["french"])
        sh_o = j["ours"].mean() / j["ours"].std() * np.sqrt(252)
        sh_f = j["french"].mean() / j["french"].std() * np.sqrt(252)
        print(f"{sleeve:9s} {fcol:6s} {corr:>7.2f} {sh_o:>13.2f} {sh_f:>14.2f}")
        ROWS.append({"kind": "factor", "ours": sleeve, "french": fcol, "corr": corr, "sharpe_ours": sh_o, "sharpe_french": sh_f, "n_days": len(j)})

    # bucket-level diagnostic: the four RMW cells vs French's 6 ME/OP portfolios
    CELLS = [("q_SH", "SMALL HiOP"), ("q_SL", "SMALL LoOP"), ("q_BH", "BIG HiOP"), ("q_BL", "BIG LoOP")]
    if any(c in ours.columns for c, _ in CELLS):
        port = web.DataReader("6_Portfolios_ME_OP_2x3_daily", "famafrench", start, end)[0] / 100.0
        port.columns = [c.strip() for c in port.columns]
        print(f"\nFrench 6 ME/OP cols: {list(port.columns)}")
        print(f"{'cell':9s} {'French':12s} {'corr':>7s}")
        for oc, fc in CELLS:
            if oc in ours.columns and fc in port.columns:
                j = pd.concat([ours[oc], port[fc]], axis=1, join="inner").dropna()
                print(f"{oc:9s} {fc:12s} {j.iloc[:, 0].corr(j.iloc[:, 1]):>7.2f}")
                ROWS.append({"kind": "cell", "ours": oc, "french": fc, "corr": j.iloc[:, 0].corr(j.iloc[:, 1]), "n_days": len(j)})
        # Does the FF5 RMW factor equal the RMW built from its own six portfolios?
        need = ["SMALL HiOP", "BIG HiOP", "SMALL LoOP", "BIG LoOP"]
        if all(c in port.columns for c in need):
            rmw6 = 0.5 * (port["SMALL HiOP"] + port["BIG HiOP"]) - 0.5 * (port["SMALL LoOP"] + port["BIG LoOP"])
            jf = pd.concat([french["RMW"], rmw6], axis=1, join="inner").dropna()
            jo = pd.concat([ours["quality"], rmw6], axis=1, join="inner").dropna()
            print(f"\nFrench RMW factor vs RMW from its own 6 portfolios : corr {jf.iloc[:, 0].corr(jf.iloc[:, 1]):.3f}")
            print(f"quality sleeve vs RMW from the 6 portfolios : corr {jo.iloc[:, 0].corr(jo.iloc[:, 1]):.3f}")
            ROWS.append({"kind": "rmw6", "ours": "French RMW", "french": "RMW from 6", "corr": jf.iloc[:, 0].corr(jf.iloc[:, 1]), "n_days": len(jf)})
            ROWS.append({"kind": "rmw6", "ours": "quality", "french": "RMW from 6", "corr": jo.iloc[:, 0].corr(jo.iloc[:, 1]), "n_days": len(jo)})
    pd.DataFrame(ROWS).to_csv(os.path.join(OUT, "validate_french.csv"), index=False, float_format="%.4f") # Exhibit A13, persisted


if __name__ == "__main__":
    main()
