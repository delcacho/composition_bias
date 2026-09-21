"""
etf_sizing_race.py -- sizing one fund: is its volatility better forecast by the book it holds
now (holdings-implied volatility) or by the volatility its past books generated (stream volatility)?

On every fund's clone at unit gross (book of day t on asset returns of day t+1, so an unscaled book: the
internal sizing problem), at the month-end trading dates of the common calendar (the same dates for every fund):
  sigma_H(t) = sqrt(w_t' Sigma_t w_t) Sigma_t = EWMA asset covariance restricted to the fund's names, at halflife 63 and 252.
               This is the definition; it is computed by re-pricing the in-force book on the asset returns (sigma_H_virtual,
               virtual returns) with the identical missing-return rule as the clone/stream -- parameter-free, O(T*n), and
               equal to the quadratic form when the held names are fully observed. --cov forces the O(n^2) covariance form.
  sigma_R(t) = EWMA volatility of the clone's own returns, halflife 63 (the article's stream) and 252
  holdings and stream are always compared at the same halflife (on a frozen book they are then the same number), and by the
  same missing-return rule: the only difference is holdings applies today's book to history, the stream each day's own book
  (a) forecast comparison: e = log(sigma_hat) - log(sigma_realized over the next h days), h in {21, 63}; RMSE per fund
  (b) sizing comparison: l_t = sigma* / sigma_hat(t) held for the next REBAL days, sized return l_t r_{t+1};
      realized volatility of the sized series over each forward h-day window vs sigma*; RMSE of log(realized/sigma*);
      turnover = mean |l_t - l_{t-1}| / l_{t-1} per rebalance
Both scores per fund, then medians by persistence band (market-neutral phibar(63) from etf_results/persistence_daily.csv)
with a fund bootstrap on the median log ratio holdings / stream.

Outputs etf_results/sizing_funds.csv (one row per fund x h x rule) and etf_results/sizing_summary.csv (bands).
  python -u research/etf_sizing_race.py [--all] [--seed=7] [--cov (covariance quadratic form instead of virtual returns)]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import etf_coverage as ec # noqa: E402
import etf_race as er # noqa: E402
import return_availability as ra # noqa: E402 missing returns as missing, not zeros

L_ASSET = 252
L_STREAM = [63, 252]
HS = [21, 63]
REBAL = 5 # leverage refresh inside a scored window, trading days (the evaluation dates are month-ends)
MIN_EVAL_DATES = 12 # month-end dates a fund needs to be scored (a year of monthly dates)
START_CLONE_DAYS = 300 # clone days before a fund's first evaluation date
MIN_FWD_SHARE = 0.9 # a forward window needs 90% of its days defined
SIGMA_STAR = 0.10 / np.sqrt(252)
OUT = "etf_results"
BANDS = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, ">0.95")]
N_BOOT = 2000
MIN_L = 0.05; MAX_L = 20.0
VIRTUAL = True # sigma_H by virtual returns (the practical estimate); --cov forces the O(n^2) quadratic form
DATE_ROWS = [] # per fund x date forecasts, written to sizing_dates.parquet for the divergence check


def ewma_var(x, hl):
    lam = 0.5 ** (1.0 / hl); v = np.zeros(len(x)); s = 0.0
    for i, r in enumerate(x):
        s = lam * s + (1 - lam) * r * r; v[i] = s
    return v


def sigma_H_cov(Xraw, Wd, last, need, hl):
    """holdings-implied sigma as the pairwise-masked EWMA covariance quadratic form sqrt(w' Sigma w): the mathematical
    definition. Retained only for the equivalence check against sigma_H_virtual; the run uses the virtual path."""
    T, n = Xraw.shape
    E = ra.MaskedEWMA(n, hl); s_h = np.full(T, np.nan); bad = 0
    for d in range(int(np.max(np.where(need)[0])) + 1):
        E.update(Xraw[d])
        if not need[d] or last[d] < 0:
            continue
        wk = ra.renorm_books(Wd[last[d]][None, :], E.priced(), min_share=ra.MIN_PRICED_SHARE)[0]
        if not np.isfinite(wk).all():
            continue
        nz = np.nonzero(wk)[0]; wz = wk[nz].astype(np.float64)
        v = float(wz @ E.cov(nz) @ wz) # NaN if two held names were never observed together
        if np.isfinite(v) and v > 0:
            s_h[d] = np.sqrt(v)
        elif np.isfinite(v):
            bad += 1 # the pairwise-masked S is not PSD: counted, not sized on
    return s_h, bad


def sigma_H_virtual(Xraw, Av, Wd, last, need, hl):
    """holdings-implied sigma by virtual returns (the practical estimate): re-price the in-force book on the asset returns
    with the same missing-return rule as the clone and the stream -- ra.masked_book_returns (each day, drop the unpriced
    names, rescale each leg to its gross, undefined below MIN_PRICED_SHARE) -- then take the missing-aware EWMA vol
    (ra.masked_ewma_var). One series per in-force book. Parameter-free beyond the clone's existing floors, and symmetric
    with the stream: the only difference from sigma_R is that holdings applies today's book to history, the stream each
    day's own book. Equals the covariance quadratic form sqrt(w' Sigma w) exactly when the held names are fully observed;
    under gaps it treats a missing return exactly as the stream does. O(T*n) per in-force book, no n x n state."""
    T, n = Xraw.shape
    s_h = np.full(T, np.nan); cache = {}; dmax = int(np.max(np.where(need)[0]))
    for d in range(dmax + 1):
        if not need[d] or last[d] < 0:
            continue
        b = int(last[d]); sig = cache.get(b)
        if sig is None: # project the fixed in-force book over history, once per book
            v = ra.masked_book_returns(np.broadcast_to(Wd[b], (dmax + 1, n)), Xraw[:dmax + 1], Av[:dmax + 1])
            sig = np.sqrt(ra.masked_ewma_var(v, hl)); cache[b] = sig
        s_h[d] = sig[d]
    return s_h, 0


def month_ends(index):
    """positions of the last trading day of each calendar month: the common evaluation grid of every fund, so the divergence
    check's date clusters line up across funds."""
    s = pd.Series(np.arange(len(index)), index=index)
    return np.sort(s.groupby([index.year, index.month]).max().values)


def fund_scores(t, dates, W, Rs, X=None, RAW=None):
    """per fund: forecast errors and sized-series realized vols at every month-end evaluation date. X is unused (kept for
    callers); RAW is Rs.values, passed once by the caller to avoid a copy per fund.

    Timing, on the calendar of Rs (etf_race.clone_returns): r[d+1] is the book of day d on the returns of day d+1. At an
    evaluation day d both forecasts read information through day d: the stream the clone returns r[..d], the holdings the
    asset covariance through d applied to the book in force at d (the last book at or before d); the target is r[d+1..d+h].
    Both sides are indexed on the calendar of Rs, so the stream and the holdings EWMA decay over the same days."""
    idx = Rs.index; T = len(idx)
    RAW = Rs.values if RAW is None else RAW
    pos = idx.get_indexer(pd.DatetimeIndex(dates)); rows_ok = np.where(pos >= 0)[0]
    ii = pos[rows_ok]
    if len(ii) < 400:
        return None
    cols = np.unique(W.indices)
    Wd = W[rows_ok][:, cols].toarray() # fund book days x support
    Xraw = RAW[:, cols] # NaN where a name has no return (not a zero fill)
    Av = np.isfinite(Xraw)
    # the clone on the calendar, missing returns as missing (NaN on days under 95% of the gross is priced)
    r = np.full(T, np.nan); nx = ii + 1 < T
    r[ii[nx] + 1] = ra.masked_book_returns(Wd[nx], Xraw[ii[nx] + 1], Av[ii[nx] + 1])
    last = np.searchsorted(ii, np.arange(T), side="right") - 1 # book row in force at day d; -1 before the first book
    ncl = np.cumsum(np.isfinite(r))
    ev = [d for d in month_ends(idx) if last[d] >= 0 and ncl[d] >= START_CLONE_DAYS and d + max(HS) < T]
    if not ev:
        return None
    need = np.zeros(T, bool) # evaluation days and the leverage refreshes inside windows
    for d in ev:
        need[d:d + max(HS):REBAL] = True
    # EWMA asset covariance on the support at each stream halflife (holdings and stream are compared at the same
    # halflife), a recursion over the calendar: the exact pairwise-masked EWMA (ra.MaskedEWMA), names under the coverage floor
    # unpriced on the day, each leg of the book back to its gross, under the clone's 95% priced share undefined
    sigH = {}; nonpsd = {}
    for hl in L_STREAM:
        if VIRTUAL:
            s_h, bad = sigma_H_virtual(Xraw, Av, Wd, last, need, hl)
        else:
            s_h, bad = sigma_H_cov(Xraw, Wd, last, need, hl)
        sigH[hl] = s_h; nonpsd[hl] = bad
    sigR = {hl: np.sqrt(ra.masked_ewma_var(r, hl)) for hl in L_STREAM} # same missing-aware, normalized EWMA, through day d
    # per-date record for the divergence check: both forecasts at d, at each halflife, and the realized vol ahead
    for d in ev:
        rec = {"ticker": t, "date": idx[d]}
        for hl in L_STREAM:
            rec[f"sigma_H{hl}"] = float(sigH[hl][d]); rec[f"sigma_R{hl}"] = float(sigR[hl][d])
        for h in HS:
            win = r[d + 1:d + 1 + h]
            rec[f"realized_{h}"] = float(np.sqrt(np.nanmean(win ** 2))) if np.isfinite(win).mean() >= MIN_FWD_SHARE else np.nan
        DATE_ROWS.append(rec)
    rows = []
    for hl in L_STREAM:
        # holdings and stream at this halflife are scored on the same dates: a date (or a leverage refresh) where either
        # forecast is undefined is skipped (refresh: the previous leverage is kept) for both rules
        valid = np.isfinite(sigH[hl]) & np.isfinite(sigR[hl]) & (np.nan_to_num(sigH[hl]) > 0) & (np.nan_to_num(sigR[hl]) > 0)
        for name, sig in ((f"holdings{hl}", sigH[hl]), (f"stream{hl}", sigR[hl])):
            with np.errstate(divide="ignore", invalid="ignore"):
                lev = np.clip(SIGMA_STAR / sig, MIN_L, MAX_L)
            for h in HS:
                fe, se, lv = [], [], []
                for d in ev:
                    if not valid[d]:
                        continue
                    win = r[d + 1:d + 1 + h]
                    if np.isfinite(win).mean() < MIN_FWD_SHARE:
                        continue # a forward window with more than 10% undefined clone days is not scored
                    real = np.sqrt(np.nanmean(win ** 2))
                    if real <= 0:
                        continue
                    fe.append(np.log(sig[d]) - np.log(real))
                    # sized series: the leverage set at day d + j (information through d + j) applies to r[d + j + 1] = win[j],
                    # refreshed every REBAL days inside the window
                    l = lev[d]; sized = np.empty(h)
                    for j in range(h):
                        if j and j % REBAL == 0 and valid[d + j]:
                            l = lev[d + j]
                        sized[j] = l * win[j]
                    se.append(np.log(np.sqrt(np.nanmean(sized ** 2)) / SIGMA_STAR))
                    lv.append(lev[d])
                if len(fe) < MIN_EVAL_DATES:
                    continue
                lv = np.array(lv)
                rows.append({"ticker": t, "rule": name, "h": h, "n": len(fe), "rmse_forecast": float(np.sqrt(np.mean(np.square(fe)))),
                             "bias_forecast": float(np.mean(fe)), "rmse_sizing": float(np.sqrt(np.mean(np.square(se)))),
                             "turnover": float(np.mean(np.abs(np.diff(lv)) / lv[:-1])) if len(lv) > 1 else np.nan,
                             "lev_med": float(np.median(lv)), "holdings_nonpsd_days": nonpsd[hl]})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    global VIRTUAL
    VIRTUAL = "--cov" not in sys.argv
    print(f"sigma_H = {'virtual returns (stream-symmetric missing-return rule)' if VIRTUAL else 'covariance quadratic form (--cov)'}", flush=True)
    seed = er.arg("seed", 7)
    sample = pd.read_csv(er.U)
    tickers = sorted(sample.ticker if "--all" in sys.argv else sample[sample.in_sample].ticker)
    Rs = er.load_returns(); names = list(Rs.columns); name_ix = {c: i for i, c in enumerate(names)}
    books = er.load_books(tickers, names, name_ix)
    RAW = Rs.values # once, not a copy per fund
    pers = pd.read_csv(f"{OUT}/persistence_daily.csv", index_col=0) if os.path.exists(f"{OUT}/persistence_daily.csv") else None
    rows = []; t0 = time.time()
    for i, (t, (dates, W, _)) in enumerate(books.items(), 1):
        r = fund_scores(t, dates, W, Rs, RAW=RAW)
        if r:
            rows += r
        print(ec.progress(i, len(books), t0), t, "" if r else "(too short)", flush=True)
    F = pd.DataFrame(rows)
    if pers is not None:
        F = F.merge(pers[["phibar_neutral_63", "phibar_risk_63"]], left_on="ticker", right_index=True, how="left")
    F.to_csv(f"{OUT}/sizing_funds.csv", index=False, float_format="%.5f")
    if DATE_ROWS:
        D = pd.DataFrame(DATE_ROWS); D["date"] = pd.to_datetime(D.date)
        D.to_parquet(f"{OUT}/sizing_dates.parquet", index=False)
        print(f"wrote {OUT}/sizing_dates.parquet: {len(D)} fund-dates (divergence-check input)")
    # summary: holdings vs each stream rule, per h and band, fund bootstrap on the median log ratio
    rng = np.random.default_rng(seed); out = []
    for h in HS:
        Fh = F[F.h == h]
        for hl in L_STREAM:
            sr = f"stream{hl}"
            Hd = Fh[Fh.rule == f"holdings{hl}"].set_index("ticker") # holdings at the same halflife as the stream
            Sd = Fh[Fh.rule == sr].set_index("ticker")
            common = Hd.index.intersection(Sd.index)
            for metric in ("rmse_forecast", "rmse_sizing"):
                lr = np.log(Hd.loc[common, metric] / Sd.loc[common, metric])
                band_of = pd.cut(Hd.loc[common, "phibar_neutral_63"], [b[0] for b in BANDS] + [1.01], labels=[b[2] for b in BANDS], right=False) \
                    if "phibar_neutral_63" in Hd.columns else pd.Series("all", index=common)
                for band in list(dict.fromkeys(band_of.dropna())) + ["all"]:
                    x = lr if band == "all" else lr[band_of == band]
                    x = x.dropna().values
                    if len(x) < 3:
                        continue
                    meds = [np.median(rng.choice(x, len(x))) for _ in range(N_BOOT)]
                    out.append({"h": h, "stream_rule": sr, "metric": metric, "band": band, "funds": len(x),
                                "median_log_ratio_H_over_R": float(np.median(x)), "ci_lo": float(np.percentile(meds, 2.5)),
                                "ci_hi": float(np.percentile(meds, 97.5)), "share_H_better": float(np.mean(x < 0))})
    S = pd.DataFrame(out); S.to_csv(f"{OUT}/sizing_summary.csv", index=False, float_format="%.4f")
    print(S.round(3).to_string(index=False))
    turn = F.groupby(["rule", "h"]).turnover.median().unstack(); print("\nmedian leverage turnover per rebalance:\n", turn.round(3).to_string())


if __name__ == "__main__":
    main()
