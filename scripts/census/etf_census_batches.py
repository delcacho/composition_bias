"""
etf_census_batches.py -- split the N-CEN active-ETF census into batch files for the language-model classifier
(prompt in research/etf_census_review_prompt.txt) and, after the classifier has run, merge its outputs.

  python research/etf_census_batches.py split [--size=300]   -> research/etf_universe/census_batches/batch_NN.csv
  python research/etf_census_batches.py merge                -> research/etf_universe/ncen_active_types.csv

The merged file joins the classifier's type/transparent labels back to the census (series_id) and prints the
counts; funds typed equity_longonly or equity_longshort with transparent != N are the download list for
the constituents pull (research/massive.py takes any CSV with a 'ticker' column).
"""
import glob
import os
import sys

import pandas as pd

U = "research/etf_universe"
BATCH_DIR = f"{U}/census_batches"


def split(size):
    A = pd.read_csv(f"{U}/ncen_active_etfs.csv", dtype=str)
    A = A[A.tickers.notna()].sort_values("tickers").reset_index(drop=True)
    os.makedirs(BATCH_DIR, exist_ok=True)
    cols = ["SERIES_ID", "tickers", "fund_name", "registrant", "adviser", "first_year", "last_year", "fund_of_funds", "net_assets_last", "terminated"]
    n = 0
    for i in range(0, len(A), size):
        b = A.iloc[i:i + size][cols].rename(columns={"SERIES_ID": "series_id"})
        b.to_csv(f"{BATCH_DIR}/batch_{n:02d}.csv", index=False); n += 1
    print(f"{len(A)} active ETF series with tickers -> {n} batches of up to {size} in {BATCH_DIR}/")


def merge():
    outs = sorted(glob.glob(f"{BATCH_DIR}/batch_*_types.csv"))
    if not outs:
        sys.exit("no batch_*_types.csv outputs yet")
    T = pd.concat([pd.read_csv(f, dtype=str) for f in outs], ignore_index=True).drop_duplicates("series_id")
    A = pd.read_csv(f"{U}/ncen_active_etfs.csv", dtype=str).rename(columns={"SERIES_ID": "series_id"})
    M = A.merge(T[["series_id", "type", "transparent", "confidence", "note"]], on="series_id", how="left")
    M.to_csv(f"{U}/ncen_active_types.csv", index=False)
    print(f"{len(M)} active series, {M.type.notna().sum()} typed")
    print(M.type.fillna("untyped").value_counts().to_string())
    print("transparent:", M.transparent.fillna("?").value_counts().to_dict())
    dl = M[M.type.isin(["equity_longonly", "equity_longshort"]) & M.transparent.ne("N")]
    dl = dl.assign(ticker=dl.tickers.str.split("|").str[0])[["ticker", "series_id", "fund_name", "type", "first_year", "last_year"]]
    dl.to_csv(f"{U}/census_download_list.csv", index=False)
    print(f"download list: {len(dl)} equity funds -> {U}/census_download_list.csv")
    # crypto books are priceable on the crypto sleeve's own data: kept as a separate list, not excluded
    cr = M[M.type.eq("crypto_digital") & M.transparent.ne("N")]
    cr = cr.assign(ticker=cr.tickers.str.split("|").str[0])[["ticker", "series_id", "fund_name", "type", "first_year", "last_year"]]
    cr.to_csv(f"{U}/census_download_list_crypto.csv", index=False)
    print(f"crypto list: {len(cr)} funds -> {U}/census_download_list_crypto.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "split"
    if cmd == "split":
        size = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--size=")), 300)
        split(size)
    else:
        merge()
