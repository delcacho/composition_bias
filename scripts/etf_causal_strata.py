"""
etf_causal_strata.py -- re-cut the active-ETF full-window comparison on EX-ANTE persistence
(Online Appendix G). Reuses the experiment's analysis unchanged.

The forecast chunks do not depend on the strata, so nothing is recomputed: the pooled full-window
halflife-63 chunks (etf_fullwindow_race.py's output, pooled ids = allocation 0.., long/short, long-only
in that order) are copied with `phimin`, the pair's less persistent fund, replaced by the causal
pair-minimum at the chunk's date from etf_results/persistence_causal.csv (nan where either fund has
no causal value yet, which drops the pair-date from the bands); the diagonal frames get the causal
value as `phibar`. The copy is passed through the experiment's build_pooled_chunks (re-cuts the
quartiles on the new phimin, ids unchanged) and analyze, so Exhibit 9's fixed bands, the quartile
regressions and the allocation row come out on causal persistence; switch_timesplit.py and
etf_dyadic_inference.py are then run on the pooled copy.

Measures: "phibar" (expanding-window phibar(63), the primary) and "twosnap" (the two-snapshot phi(63)).
Output: etf_results/experiment_causal/<measure>/pooled_neutral_all_L63_fullwin[_allocation]/...,
plus etf_results/experiment_causal/coverage_<measure>.csv.

usage: python -u scripts/etf_causal_strata.py [--measure=phibar|twosnap|both]
"""
import glob
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_staleness_experiment as ex  # noqa: E402
import etf_race as er  # noqa: E402

POOLED_ORDER = ["allocation", "longshort", "longonly"]      # build_pooled_chunks' dir order in the experiment (mf has no chunks)
SRC = f"{er.OUT}/experiment"
DST = f"{er.OUT}/experiment_causal"
COL = {"phibar": "phibar_causal_63", "twosnap": "phi63_two_snapshot"}


def pooled_tickers():
    """Pooled id -> ticker: the classes' sorted priced funds concatenated in build_pooled_chunks' order."""
    tk = []
    for c in POOLED_ORDER:
        tk += pd.read_csv(f"{er.OUT}/persistence_causal_keep_{c}.csv").ticker.astype(str).tolist()
    return np.array(tk, dtype=object)


def rebin(src, raw, tickers, measure, P):
    """Copy src/chunks into raw/chunks with phimin (pairs) and phibar (diagonal) := the causal value at the chunk's date.
    Returns (pair-dates, pair-dates with a causal phimin)."""
    if os.path.isdir(raw):
        shutil.rmtree(raw)
    os.makedirs(f"{raw}/chunks", exist_ok=True)
    val = P.set_index(["date", "ticker"])[COL[measure]]
    dates_with = set(val.index.get_level_values(0))
    n_all = n_cov = 0
    for f in sorted(glob.glob(f"{src}/chunks/*.parquet")):
        C = pd.read_parquet(f); name = os.path.basename(f)
        date = pd.Timestamp(name.replace("_diag", "").split(".")[0][-8:])
        if len(C):
            def look(ids):
                if date not in dates_with:
                    return np.full(len(ids), np.nan)
                return val.reindex(pd.MultiIndex.from_arrays([[date] * len(ids), tickers[ids]])).values
            if name.endswith("_diag.parquet"):
                C["phibar"] = look(C.i.values).astype(np.float32)
            else:
                C["phimin"] = np.minimum(look(C.i.values), look(C.j.values)).astype(np.float32)
                n_all += len(C); n_cov += int(np.isfinite(C.phimin.values).sum())
        C.to_parquet(f"{raw}/chunks/{name}", index=False)
    return n_all, n_cov


def run(measure, P):
    print(f"\n===== measure: {measure} =====", flush=True)
    tk = pooled_tickers(); cov = []
    for src_name, tickers, label in [("pooled_neutral_all_L63_fullwin", tk, "pooled"),
                                     ("pooled_neutral_all_L63_fullwin_allocation", tk[:tk_len("allocation")], "allocation")]:
        src = f"{SRC}/{src_name}"; raw = f"{DST}/{measure}/{src_name}_raw"; dst = f"{DST}/{measure}/{src_name}"
        n_all, n_cov = rebin(src, raw, tickers, measure, P)
        cov.append({"set": label, "pair_dates": n_all, "with_causal_phimin": n_cov})
        print(f"  {label:10s} pair-dates {n_all:,}  with causal phimin {n_cov:,} ({n_cov / max(n_all, 1):.0%})", flush=True)
        if ex.build_pooled_chunks([raw], dst):
            ex.analyze(dst, "neutral", 21, cls="pooled")
        shutil.rmtree(raw)
    pd.DataFrame(cov).to_csv(f"{DST}/coverage_{measure}.csv", index=False)
    pooled = f"{DST}/{measure}/pooled_neutral_all_L63_fullwin"
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ["switch_timesplit.py", "etf_dyadic_inference.py"]:
        print(f"\n---- {script} on {pooled} ----", flush=True)
        subprocess.run([sys.executable, "-u", os.path.join(here, script), pooled], check=False)


def tk_len(cls):
    return len(pd.read_csv(f"{er.OUT}/persistence_causal_keep_{cls}.csv"))


def main():
    which = ex.arg("measure", "both"); measures = ["phibar", "twosnap"] if which == "both" else [which]
    P = pd.read_csv(f"{er.OUT}/persistence_causal.csv", parse_dates=["date"]); P["ticker"] = P.ticker.astype(str)
    tk = pooled_tickers()
    assert len(tk) == 446 and len(set(tk)) == 446, len(tk)
    ok = P[np.isfinite(P.phibar_causal_63)]
    print(f"causal persistence: {len(P):,} fund-dates, {ok.ticker.nunique()} funds measured; "
          f"funds ever below 0.85: {int((ok.groupby('ticker').phibar_causal_63.min() < 0.85).sum())}; "
          f"two-snapshot below 0.60 at some date: {int((P.groupby('ticker').phi63_two_snapshot.min() < 0.60).sum())}", flush=True)
    for m in measures:
        run(m, P)
    print("\nwrote", DST)


if __name__ == "__main__":
    main()
