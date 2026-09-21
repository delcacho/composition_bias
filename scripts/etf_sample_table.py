"""
etf_sample_table.py -- the sample exhibit: funds scored and admitted by persistence band and fund class.

Reads research/etf_universe/universe_sample.csv (the rule's admission, in_sample and book_incomplete),
etf_results/persistence_daily.csv (market-neutral phibar at 63 days, median over each fund's days) and the census
download lists (the class a fund was drawn from). Bands are the article's cuts (0.75 / 0.85 / 0.95), frozen.
Writes etf_results/sample_by_band.csv and etf_results/sample_by_band.tex (a tabular body for the appendix) and prints it.
  python research/etf_sample_table.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from etf_staleness_experiment import fund_classes # noqa: E402 the brochure-based classes the staleness experiment uses (override applied last)

U = "research/etf_universe"
OUT = "etf_results"
BANDS = [(-1.0, 0.75, "below 0.75"), (0.75, 0.85, "0.75 to 0.85"), (0.85, 0.95, "0.85 to 0.95"), (0.95, 1.01, "above 0.95")]
CLASSES = [("allocation", "Allocation"), ("longonly", "Long-only"), ("longshort", "Long/short"), ("mf", "Managed futures"),
           ("excluded", "Excluded by strategy")]


def main():
    u = pd.read_csv(f"{U}/universe_sample.csv")
    pers = pd.read_csv(f"{OUT}/persistence_daily.csv")
    col = "phibar_neutral_63" # the horizon-average phibar(63) the article's bands and the staleness experiment use
    ph = pers.groupby("ticker")[col].median().rename("phibar")
    cls = fund_classes() # brochure (Form ADV Part 2A) classes, the same map the staleness experiment uses
    d = u.merge(ph, left_on="ticker", right_index=True, how="inner")
    d["cls"] = [("excluded" if cls.get(str(t), "longonly").startswith("excluded") else cls.get(str(t), "longonly")) for t in d.ticker]
    d["band"] = pd.cut(d.phibar, [b[0] for b in BANDS] + [BANDS[-1][1]], labels=[b[2] for b in BANDS], right=False)
    inc = d.get("book_incomplete", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    rows = []
    for _, _, band in BANDS:
        g = d[d.band == band]
        row = {"band": band, "scored": len(g), "admitted": int(g.in_sample.sum())}
        for c, _ in CLASSES:
            row[c] = int((g.in_sample & (g.cls == c)).sum())
        row["incomplete_books"] = int((g.in_sample & inc.loc[g.index]).sum())
        rows.append(row)
    T = pd.DataFrame(rows)
    tot = {"band": "all", **{k: int(T[k].sum()) for k in T.columns if k != "band"}}
    T = pd.concat([T, pd.DataFrame([tot])], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(f"{OUT}/sample_by_band.csv", index=False)
    head = ["Persistence band", "Scored", "Admitted"] + [lab for _, lab in CLASSES] + ["Incomplete books"]
    lines = [" & ".join(head) + r" \\", r"\midrule"]
    for r in T.itertuples(index=False):
        cells = [r.band.capitalize() if r.band != "all" else "All", r.scored, r.admitted] + [getattr(r, c) for c, _ in CLASSES] + [r.incomplete_books]
        if r.band == "all":
            lines.append(r"\midrule")
        lines.append(" & ".join(str(x) for x in cells) + r" \\")
    open(f"{OUT}/sample_by_band.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(T.to_string(index=False))
    print(f"wrote {OUT}/sample_by_band.csv and {OUT}/sample_by_band.tex")


if __name__ == "__main__":
    main()
