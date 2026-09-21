"""
etf_causal_persistence.py -- ex-ante persistence per fund and forecast date, for the causal re-cut of the
active-ETF strata (Online Appendix G).

At each monthly forecast date of the halflife-63 experiment, for every fund in a pair-test class with a
computable clone, two statistics an allocator could have computed on that date:
  phibar_causal_63   the article's phibar(63) from the fund's books up to the date only (expanding window,
                     gate 64 books, one quarter), in the market-neutral geometry of the trailing asset
                     covariance at that date: the experiment's trailing_persistence with the window opened
                     and the book gate lowered, nothing else changed.
  phi63_two_snapshot the cosine of the book at the date against the book 63 trading days earlier, in the
                     same geometry: the statistic Exhibit 10 prescribes from two quarterly filings.
Nothing after the forecast date is read: the book window ends at the date and the covariance is the
experiment's own trailing replay at that date.

Writes etf_results/persistence_causal.csv (date, ticker, class, n_books, phibar_causal_63, phi63_two_snapshot)
and etf_results/persistence_causal_keep_<class>.csv (each class's fund order, the id the chunks index by,
reproduced exactly as the experiment builds it). Same universe and dates as the experiment: split all,
L=63, neutral geometry. Loads the books; run it alone.

usage: python -u scripts/etf_causal_persistence.py [--L=63] [--h=21]
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_staleness_experiment as ex  # noqa: E402
import etf_race as er  # noqa: E402
import return_availability as ra  # noqa: E402

ex.PW = 10 ** 9              # expanding window: every book of the fund up to the forecast date
ex.MIN_PHI_BOOKS = 64        # one quarter of books suffices for phi(63)
CLASSES = ["allocation", "longshort", "longonly"]    # the pair-test classes (managed futures cannot form one)
OUT = er.OUT


def phibar_from_gram(G, lags=(1, 5, 21, 63)):
    """phibar(63) from a Gram matrix of consecutive books: mean cosine at each lag, interpolated over 1..63 and
    averaged, as trailing_persistence does. nan when a lag has fewer than 20 pairs."""
    dg = np.diag(G); nrm = np.sqrt(np.where(dg > 0, dg, np.nan))
    curve = {}
    for u in lags:
        if G.shape[0] > u + 20:
            v = np.diag(G, u) / (nrm[:-u] * nrm[u:])
            curve[u] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    if len(curve) < len(lags) or not np.isfinite(list(curve.values())).all():
        return np.nan
    grid = np.interp(np.arange(1, 64), list(lags), [curve[u] for u in lags])
    return float(grid.mean())


def two_snapshot_phi63(books, tk, t, t63, Sig_U, U):
    """Cosine of the book at (or just before) t against the book at (or just before) t63, under Sig_U."""
    dates, W = books[tk][0], books[tk][1]
    hi = dates.searchsorted(t, side="right") - 1
    lo = dates.searchsorted(t63, side="right") - 1
    if hi < 0 or lo < 0 or hi == lo:
        return np.nan
    Wf = W[[lo, hi]]
    cols = np.unique(Wf.indices)
    if len(cols) == 0:
        return np.nan
    p = np.searchsorted(U, cols)
    if (p >= len(U)).any() or (U[p] != cols).any():
        return np.nan
    Wd = Wf[:, cols].toarray().astype(np.float64)
    g = Wd @ np.asarray(Sig_U[np.ix_(p, p)], np.float64) @ Wd.T
    d = np.sqrt(np.diag(g))
    return float(g[0, 1] / (d[0] * d[1])) if (d > 0).all() else np.nan


def main():
    L = int(ex.arg("L", 63)); h = int(ex.arg("h", 21)); BACK = 6 * L
    sample = pd.read_csv(er.U); cls_of = ex.fund_classes()
    admit = sample.in_sample_staleness if "in_sample_staleness" in sample.columns else sample.in_sample
    active_all = [t for t in sample[admit].ticker.astype(str)]
    by_class = {c: [t for t in active_all if cls_of.get(t, "longonly") == c] for c in CLASSES}
    need = sorted(set().union(*by_class.values()))
    print(f"funds by class {{ {', '.join(f'{c}: {len(v)}' for c, v in by_class.items())} }}; loading {len(need)} books", flush=True)
    Rs = er.load_returns(); names = list(Rs.columns); name_ix = {c: i for i, c in enumerate(names)}
    books = er.load_books(need, names, name_ix)
    clones = er.clone_returns(books, Rs); priced = set(clones.columns)
    keep_of = {c: sorted(set(by_class[c]) & priced) for c in CLASSES}
    for c, kp in keep_of.items():
        pd.DataFrame({"id": range(len(kp)), "ticker": kp}).to_csv(f"{OUT}/persistence_causal_keep_{c}.csv", index=False)
        print(f"  {c:10s} {len(kp)} priced funds (id order written)", flush=True)
    keep = [t for c in CLASSES for t in keep_of[c]]; cls_by = {t: c for c in CLASSES for t in keep_of[c]}
    idx = Rs.index; T = len(idx); RAW = Rs.values; A = ra.availability(RAW)
    rv = clones[keep].reindex(idx).values.astype(np.float32); have = np.isfinite(rv)
    stock = [j for j, c in enumerate(names) if not (c.startswith("FUT:") or c.startswith("CASH") or c.startswith("TSY:"))]
    f_all = np.nan_to_num(np.asarray(np.nanmean(Rs.values[:, stock], axis=1), np.float32))
    eval_t = [ti for ti in range(L, T - h) if ti % ex.STRIDE == 0]
    print(f"{len(keep)} funds, {len(eval_t)} monthly dates, L={L}, expanding-window persistence (gate {ex.MIN_PHI_BOOKS} books)", flush=True)
    rows = []; t0 = time.time()
    for k, ti in enumerate(eval_t, 1):
        t = idx[ti]; t63 = idx[ti - 63]
        s0, wts = er.ewma_window(ti, L, BACK)
        active = have[s0:ti + 1].any(axis=0)
        wins = {tk: w for i, tk in enumerate(keep) if active[i] and (w := ex.persistence_window(books, tk, t)) is not None}
        if not wins:
            continue
        psup = [books[tk][1][w[0]:w[1]].indices for tk, w in wins.items()]
        U = np.unique(np.concatenate(psup)).astype(np.int64)
        Sig_U, _, _ = er.replay_covariance(RAW, A, s0, ti, wts, U, f_all)
        n_ok = 0
        for tk, w in wins.items():
            pb, _, _ = ex.trailing_persistence(books, tk, t, Sig_U, U)
            ts = two_snapshot_phi63(books, tk, t, t63, Sig_U, U)
            rows.append({"date": t, "ticker": tk, "class": cls_by[tk], "n_books": w[1] - w[0],
                         "phibar_causal_63": pb, "phi63_two_snapshot": ts})
            n_ok += int(np.isfinite(pb))
        el = time.time() - t0
        print(f"  [{k}/{len(eval_t)}] {t.date()}  active {int(active.sum())}  measured {n_ok}  support {len(U)}  "
              f"{el/60:.1f} min, ETA {el/k*(len(eval_t)-k)/60:.1f} min", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(f"{OUT}/persistence_causal.csv", index=False, float_format="%.4f")
    ok = P[np.isfinite(P.phibar_causal_63)]
    print(f"\nwrote {OUT}/persistence_causal.csv: {len(P)} fund-dates, {len(ok)} with phibar_causal_63; "
          f"funds ever below 0.85: {int((ok.groupby('ticker').phibar_causal_63.min() < 0.85).sum())} of {ok.ticker.nunique()}")


if __name__ == "__main__":
    main()
