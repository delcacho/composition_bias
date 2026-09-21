"""
ncen_census.py -- the population of US ETFs, year by year, from the funds' own N-CEN annual reports
(SEC structured data sets, quarterly zips, September 2018 onward). This is the census the sample is
compared against so survivorship and selection are numbers in a table rather than an objection.

Per FUND_REPORTED_INFO row (one per series per annual report): IS_ETF / IS_ETMF (exchange-traded),
IS_INDEX (the fund's own declaration that it tracks an index), IS_FUND_OF_FUND, IS_MONEY_MARKET,
MONTHLY_AVG_NET_ASSETS, NAV_PER_SHARE; SHARES_OUTSTANDING gives the class tickers, ADVISER the
adviser names, SUBMISSION the report period and CIK, REGISTRANT the trust name, and
TERMINATED_ORGANIZATION the series that died in the year with their termination month.

  python -u research/ncen_census.py                # download every quarterly zip (cached in ncen_data/) and build the census
  python -u research/ncen_census.py --no-download  # rebuild from the cached zips

Writes research/etf_universe/ncen_etf_census.csv  one row per (series_id, report year): tickers, name,
       registrant, adviser, is_index, is_fund_of_funds, is_money_market, net_assets, nav, terminated
       research/etf_universe/ncen_active_etfs.csv  the active (non-index, non-money-market) ETF series
       with first/last report year and tickers: the population for the article's appendix table.
SEC asks for a descriptive User-Agent; the script sends one.
"""
import io
import os
import re
import sys
import zipfile

import pandas as pd
import requests

PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-n-cen-data-sets"
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "research-reproduction contact@example.com"),   # set SEC_USER_AGENT to your name and email
      "Accept-Encoding": "gzip, deflate"}
CACHE = "ncen_data"
OUT_DIR = "research/etf_universe"


def zip_links():
    html = requests.get(PAGE, headers=UA, timeout=60).text
    links = sorted(set(re.findall(r'href="([^"]*ncen[^"]*\.zip)"', html, re.I)))
    return [l if l.startswith("http") else "https://www.sec.gov" + l for l in links]


def download_all():
    os.makedirs(CACHE, exist_ok=True)
    for url in zip_links():
        name = url.rsplit("/", 1)[-1]
        p = f"{CACHE}/{name}"
        if os.path.exists(p):
            continue
        r = requests.get(url, headers=UA, timeout=300); r.raise_for_status()
        open(p, "wb").write(r.content)
        print(f"  {name}: {len(r.content)/1e6:.1f} MB", flush=True)


def read_tsv(z, name, usecols=None):
    cand = [n for n in z.namelist() if n.upper().endswith(name.upper() + ".TSV")]
    if not cand:
        return None
    with z.open(cand[0]) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), sep="\t", dtype=str,
                           usecols=lambda c: (usecols is None) or (c in usecols), on_bad_lines="skip")


def main():
    if "--no-download" not in sys.argv:
        download_all()
    frames = []
    for zp in sorted(os.listdir(CACHE)):
        if not zp.lower().endswith(".zip"):
            continue
        z = zipfile.ZipFile(f"{CACHE}/{zp}")
        sub = read_tsv(z, "SUBMISSION", ["ACCESSION_NUMBER", "SUBMISSION_TYPE", "CIK", "FILING_DATE", "REPORT_ENDING_PERIOD"])
        reg = read_tsv(z, "REGISTRANT", ["ACCESSION_NUMBER", "REGISTRANT_NAME"])
        fund = read_tsv(z, "FUND_REPORTED_INFO", ["FUND_ID", "ACCESSION_NUMBER", "FUND_NAME", "SERIES_ID", "IS_ETF", "IS_ETMF", "IS_INDEX",
                                                   "IS_FUND_OF_FUND", "IS_MONEY_MARKET", "IS_TARGET_DATE", "MONTHLY_AVG_NET_ASSETS", "NAV_PER_SHARE"])
        sh = read_tsv(z, "SHARES_OUTSTANDING", ["FUND_ID", "CLASS_ID", "TICKER"])
        adv = read_tsv(z, "ADVISER", ["FUND_ID", "ADVISER_TYPE", "ADVISER_NAME"])
        term = read_tsv(z, "TERMINATED_ORGANIZATION", ["ACCESSION_NUMBER", "SERIES_ID", "TERMINATION_DATE"])
        if fund is None or sub is None:
            print(f"  {zp}: missing tables, skipped"); continue
        f = fund[(fund.IS_ETF == "Y") | (fund.IS_ETMF == "Y")].copy()
        f = f.merge(sub, on="ACCESSION_NUMBER", how="left").merge(reg, on="ACCESSION_NUMBER", how="left")
        if sh is not None:
            t = sh.dropna(subset=["TICKER"]).groupby("FUND_ID").TICKER.apply(lambda s: "|".join(sorted(set(s)))).rename("tickers")
            f = f.merge(t, on="FUND_ID", how="left")
        if adv is not None:
            a = adv[adv.ADVISER_TYPE.fillna("").str.startswith("Adviser")].groupby("FUND_ID").ADVISER_NAME.first().rename("adviser")
            f = f.merge(a, on="FUND_ID", how="left")
        if term is not None:
            tt = term.dropna(subset=["SERIES_ID"]).groupby("SERIES_ID").TERMINATION_DATE.first().rename("terminated")
            f = f.merge(tt, on="SERIES_ID", how="left")
        f["zip"] = zp
        frames.append(f)
        print(f"  {zp}: {len(f)} ETF series rows", flush=True)
    C = pd.concat(frames, ignore_index=True)
    C["report_year"] = pd.to_datetime(C.REPORT_ENDING_PERIOD, errors="coerce").dt.year
    C = C[C.SUBMISSION_TYPE.isin(["N-CEN", "N-CEN/A"])]
    # amendments supersede originals for the same series and period
    C = C.sort_values(["SERIES_ID", "REPORT_ENDING_PERIOD", "FILING_DATE"]).drop_duplicates(["SERIES_ID", "REPORT_ENDING_PERIOD"], keep="last")
    os.makedirs(OUT_DIR, exist_ok=True)
    C.to_csv(f"{OUT_DIR}/ncen_etf_census.csv", index=False)
    act = C[(C.IS_INDEX != "Y") & (C.IS_MONEY_MARKET != "Y")]
    pop = (act.groupby("SERIES_ID").agg(fund_name=("FUND_NAME", "last"), registrant=("REGISTRANT_NAME", "last"), adviser=("adviser", "last"),
                                        tickers=("tickers", "last"), first_year=("report_year", "min"), last_year=("report_year", "max"),
                                        years=("report_year", "nunique"), fund_of_funds=("IS_FUND_OF_FUND", "last"),
                                        net_assets_last=("MONTHLY_AVG_NET_ASSETS", "last"), terminated=("terminated", "last")).reset_index())
    pop.to_csv(f"{OUT_DIR}/ncen_active_etfs.csv", index=False)
    print(f"\ncensus: {C.SERIES_ID.nunique():,} ETF series, {len(C):,} series-years, {C.report_year.min()}-{C.report_year.max()}")
    print(f"active (not index, not money market): {len(pop):,} series; by first report year:")
    print(pop.first_year.value_counts().sort_index().to_string())
    print(f"terminated: {int(pop.terminated.notna().sum())}")
    print(f"wrote {OUT_DIR}/ncen_etf_census.csv and ncen_active_etfs.csv")


if __name__ == "__main__":
    main()
