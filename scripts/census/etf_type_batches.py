"""etf_type_batches.py -- split funds into batches for the language-model strategy classifier.

Reads research/etf_universe/etf_brochure_crosswalk.csv (etf_brochure_crosswalk.py), re-checks each
fund's brochure text on disk, and writes fixed-size batch CSVs the classifier reads under
research/etf_brochure_type_prompt.txt. Funds with no brochure text are listed for the web/prospectus
fallback. Output dir: etf_results/classification/batches/.

  python research/etf_type_batches.py               # all funds in the crosswalk
  python research/etf_type_batches.py --priority     # priority classes (allocation/mf/longshort/options) + longonly with phi_63<=0.85
  python research/etf_type_batches.py --rest         # everything not in --priority
Options: --per N (funds per batch, default 40).
"""
import os, sys, math
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROCH = os.path.join(ROOT, "adv_data", "brochures")
OUT = os.path.join(ROOT, "etf_results", "classification", "batches")
COLS = ["ticker", "fund_name", "name_type", "phi_63", "crds", "text_paths"]
TREATMENT = ["allocation_fof", "commodity_futures", "equity_longshort", "equity_options"]


def flo(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 99.0


def disk_paths(crds):
    return ";".join(f"adv_data/brochures/{c}/text.txt" for c in str(crds).split(";")
                    if c and os.path.exists(os.path.join(BROCH, c, "text.txt"))
                    and os.path.getsize(os.path.join(BROCH, c, "text.txt")) > 200)


def main():
    per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 40
    X = pd.read_csv(os.path.join(ROOT, "research", "etf_universe", "etf_brochure_crosswalk.csv"), dtype=str).fillna("")
    X["text_paths"] = X.crds.map(disk_paths)                 # re-derive from disk (picks up fresh fetches)
    X["p63"] = X.phi_63.map(flo)
    pri = X.name_type.isin(TREATMENT) | ((X.name_type == "equity_longonly") & (X.p63 <= 0.85))
    if "--priority" in sys.argv:
        X = X[pri]
    elif "--rest" in sys.argv:
        X = X[~pri]

    have = X[X.text_paths != ""].copy()
    none = X[X.text_paths == ""]
    print(f"{len(X)} funds; {len(have)} with brochure text, {len(none)} without (web/prospectus fallback)")
    if len(none):
        print("  no text:", none.ticker.tolist())

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.startswith("batch_"):                           # clear only batch inputs; never delete result/ files
            os.remove(os.path.join(OUT, f))
    have["fund_name"] = ""
    have = have.sort_values(["name_type", "ticker"]).reset_index(drop=True)
    nb = math.ceil(len(have) / per) if len(have) else 0
    for i in range(nb):
        have.iloc[i * per:(i + 1) * per][COLS].to_csv(os.path.join(OUT, f"batch_{i:02d}.csv"), index=False)
    print(f"wrote {nb} batches of <= {per} to {os.path.relpath(OUT, ROOT)}")
    print("classify each with research/etf_brochure_type_prompt.txt; write results to "
          "etf_results/classification/results/result_NN.csv")


if __name__ == "__main__":
    main()
