"""etf_type_merge.py -- consolidate the language-model classifier's results into one per-fund table and
the class override the staleness experiment reads.

Reads every etf_results/classification/results/result_*.csv (columns ticker,type,rotates,transparent,
confidence,agrees_name,source,note) plus the crosswalk (name_type, phi_63). Writes:
  research/etf_universe/etf_type_classification.csv -- the full per-fund classification (the committed record)
  research/etf_universe/etf_class_overrides.csv -- ticker,class,... where the brochure class differs from the
                                                         census-list default (etf_staleness_experiment.fund_classes
                                                         reads it last). A class starting 'excluded_' drops the fund.
Class = the brochure type mapped to {allocation, mf, longshort, longonly}; a transparent=N fund
(proxy-basket / ActiveShares) is excluded_semitransparent; options/fixed-income/crypto/other are excluded.

Safety: the override is regenerated from the classification, but existing override rows for tickers not
present in the current results are preserved (so a partial results set cannot silently drop corrections;
a warning lists them). The definitive override comes from a complete classification; regenerate any
missing batches first.

  python research/etf_type_merge.py
"""
import os, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(ROOT, "research", "etf_universe")
RES = os.path.join(ROOT, "etf_results", "classification", "results")
TYPE2CLASS = {"equity_longonly": "longonly", "allocation_fof": "allocation", "commodity_futures": "mf",
              "equity_longshort": "longshort", "equity_options": "excluded_options",
              "fixed_income": "excluded_fixedincome", "crypto_digital": "excluded_crypto", "other": "excluded_other"}
CENSUS = [("census_download_list_allocation.csv", "allocation"),
          ("census_download_list_futures.csv", "mf"),
          ("census_download_list_longshort.csv", "longshort")]
INDIV = {"AESR", "FCTE", "KEAT", "MODL", "SNAV"} # name-typed allocation_fof but hold individual stocks -> longonly


def read_results(res_dir):
    """Robust read of the classifier result CSVs: the note field can carry unquoted commas, so split each
    line into the 8 fixed columns (extra commas fold back into the note)."""
    cols = ["ticker", "type", "rotates", "transparent", "confidence", "agrees_name", "source", "note"]
    rows = []
    for f in sorted(glob.glob(os.path.join(res_dir, "result_*.csv"))):
        with open(f, encoding="utf-8") as fh:
            next(fh, None)
            for ln in fh:
                if not ln.strip():
                    continue
                p = ln.rstrip("\n").split(",", 7)
                if len(p) >= 7:
                    rows.append((p + [""] * 8)[:8])
    return pd.DataFrame(rows, columns=cols).drop_duplicates("ticker", keep="last")


def default_class():
    """fund_classes()' default: census-list membership, else longonly."""
    d = {}
    for fn, c in CENSUS:
        p = os.path.join(U, fn)
        if os.path.exists(p):
            for t in pd.read_csv(p).ticker.astype(str):
                d[t] = c
    return d


def panel_of(row):
    # a daily-holdings test needs the real book: exclude non-transparent (N) and unconfirmed (U)
    # funds, whose published "holdings" may be a proxy basket rather than the portfolio held.
    if str(row.get("transparent", "")).upper() in ("N", "U"):
        return "excluded_semitransparent"
    return TYPE2CLASS.get(str(row.get("type", "")), "longonly")


def main():
    files = sorted(glob.glob(os.path.join(RES, "result_*.csv")))
    if not files:
        raise SystemExit(f"no result_*.csv in {RES}")
    res = read_results(RES)      # the note field carries unquoted commas, which pd.read_csv cannot parse
    X = pd.read_csv(os.path.join(U, "etf_brochure_crosswalk.csv"), dtype=str)
    M = res.merge(X[["ticker", "name_type", "phi_63", "phibar_63"]], on="ticker", how="left")
    M.to_csv(os.path.join(U, "etf_type_classification.csv"), index=False)
    print(f"classification: {len(M)} funds from {len(files)} result files")

    dflt = default_class()
    M["panel"] = M.apply(panel_of, axis=1)
    M["dflt"] = M.ticker.map(lambda t: dflt.get(str(t), "longonly"))
    corr = M[M.panel != M.dflt][["ticker", "panel", "type", "name_type", "source", "note"]].rename(
        columns={"panel": "class", "type": "brochure_type"})

    ovp = os.path.join(U, "etf_class_overrides.csv")
    if os.path.exists(ovp): # preserve corrections for funds not in this results set
        old = pd.read_csv(ovp, dtype=str)
        keep = old[~old.ticker.isin(M.ticker)].copy()
        if len(keep):
            print(f" preserving {len(keep)} existing override rows for funds not in current results: {sorted(keep.ticker)}")
        for c in corr.columns:
            if c not in keep.columns:
                keep[c] = ""
        corr = pd.concat([keep[corr.columns], corr], ignore_index=True).drop_duplicates("ticker", keep="last")
    corr.to_csv(ovp, index=False)
    print(f"override: {len(corr)} class corrections -> {os.path.relpath(ovp, ROOT)}")
    print(corr.groupby(["name_type", "class"]).size().to_string())

    print("\ncorrections vs name_type (agrees_name=N):", (M.agrees_name.str.upper() == "N").sum(), "of", len(M))
    sus = M[(M.name_type == "equity_longonly") & (pd.to_numeric(M.phi_63, errors="coerce") <= 0.85)]
    print(f"long-only class integrity: {len(sus)} longonly funds with phi_63<=0.85; brochure rotates:")
    print(" ", sus.rotates.value_counts().to_dict())


if __name__ == "__main__":
    main()
