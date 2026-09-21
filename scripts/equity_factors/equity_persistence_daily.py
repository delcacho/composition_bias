"""
equity_persistence_daily.py -- equity sleeve persistence phi(h) at daily resolution (h=5,21,63,126),
to slot into the article's commodity persistence exhibit on the same footing.

phi(h) is the Sigma-norm cosine between a sleeve's book on day t and its book h days later, averaged
over t (equation phi of the article). With a fixed full-sample stock covariance Sigma, this equals
the sample covariance of the two frozen books' virtual return series, so it never forms the N x N
Sigma: it builds the daily virtual-return panel V[s,t] = B_t . r(s) (sparse daily books times dense
returns, exploiting sparsity) and reads phi off the T x T Gram. This is the virtual-return
projection, run for the equity sleeves.

Inputs (research/equity_factors/_data/, via build_ff_annual + build_equity_holdings).
Output: appends to research/equity_factors/_out/equity_persistence.csv the daily phi columns.
Usage: python research/equity_factors/equity_persistence_daily.py
"""
import os
import numpy as np
import pandas as pd

from panel_pivot import pivot_mean
from scipy import sparse

from build_ff_annual import load_daily, monthly, annual_acct
from build_equity_holdings import build_membership

OUT = os.path.join("research", "equity_factors", "_out")
SLEEVES = ["momentum", "value", "quality"]
HS = [5, 21, 63, 126]


def daily_books(px, membership, sleeve):
    """Sparse daily net-weight book matrix (T_active x n_stock) and the aligned active dates.
    Net weight of stock k on day d = sign(leg) * mcap_k(d) / sum(mcap in its size x leg cell)."""
    mk = membership[membership["sleeve"] == sleeve]
    dm = px[["gvkey", "ym", "date", "mktcap"]]
    dm = dm[dm["mktcap"] > 0]
    gvks = np.sort(mk["gvkey"].unique())
    gix = {g: i for i, g in enumerate(gvks)}
    day_rows = {} # date -> list of (col, w)
    for aym, g in mk.groupby("apply_ym"):
        d = dm[dm["ym"] == aym].merge(g[["gvkey", "leg"]], on="gvkey")
        if d.empty:
            continue
        d["sign"] = np.where(d["leg"].str[1] == "H", 0.5, -0.5)
        den = d.groupby(["date", "leg"])["mktcap"].transform("sum")
        d["w"] = d["sign"] * d["mktcap"] / den
        for dt, sub in d.groupby("date"):
            day_rows.setdefault(dt, []).extend(
                zip((gix[gv] for gv in sub["gvkey"]), sub["w"].to_numpy()))
    dates = sorted(day_rows)
    rows, cols, vals = [], [], []
    for r, dt in enumerate(dates):
        for c, w in day_rows[dt]:
            rows.append(r); cols.append(c); vals.append(w)
    Wd = sparse.csr_matrix((vals, (rows, cols)), shape=(len(dates), len(gvks)))
    return Wd, pd.to_datetime(dates), gvks


def phi_from_books(Wd, dates, gvks, px):
    """phi(h) via the daily virtual-return Gram. Sigma is the fixed full-sample stock covariance,
    implicit in V.T V; Rmat holds each name's return on the sleeve's active dates."""
    ret = pivot_mean(px, dates, gvks) # filled directly (see panel_pivot)
    Rmat = np.nan_to_num(ret.to_numpy(dtype="float64"), nan=0.0) # screened+winsorized at source
    V = np.asarray(Wd @ Rmat.T) # T(book) x T(sample), sparse@dense
    T = V.shape[1]
    G = (V @ V.T) / T # T x T Gram of frozen-book virtual returns
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = G / np.outer(d, d)
    out = {}
    for h in HS:
        vals = [C[t, t + h] for t in range(len(dates) - h)]
        out[h] = float(np.nanmean(vals)) if vals else np.nan
    return out


def main():
    px = load_daily()
    m = monthly(px)
    fa = annual_acct()
    june = m[m["month"] == 6][["gvkey", "year", "mktcap", "nyse", "gsector"]]
    dec = m[m["month"] == 12][["gvkey", "year", "mktcap"]].rename(columns={"mktcap": "me_dec"})
    membership = build_membership(m, fa, june, dec)
    px = px[["gvkey", "ym", "date", "mktcap", "ret"]]

    rows = []
    for s in SLEEVES:
        Wd, dates, gvks = daily_books(px, membership, s)
        phi = phi_from_books(Wd, dates, gvks, px[px["gvkey"].isin(gvks)])
        rows.append({"sleeve": s, **{f"phi{h}_daily": phi[h] for h in HS}})
        print(f"{s:9s} " + " ".join(f"phi({h})={phi[h]:.2f}" for h in HS), flush=True)
    df = pd.DataFrame(rows).set_index("sleeve")
    path = os.path.join(OUT, "equity_persistence_daily.csv")
    df.to_csv(path, float_format="%.3f")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
