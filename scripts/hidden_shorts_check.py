"""
hidden_shorts_check.py -- what a 13F-style filing (long side only) does to the allocator's tools on
a long/short manager. On the public commodity book, for each long/short sleeve: build quarterly
"filings" that show only the long positions (negatives dropped, dollar scale kept, as a 13F reports them), then
  (a) snapshot persistence phi(k quarters) from the long-only filings vs the true (full-book) phi;
  (b) churn detector: corr(frozen long-only filing return over the next quarter, actual full-book
      return) vs the same with the full filing, vs true phibar(63).
Long/flat and long-only sleeves are unaffected by construction and serve as controls.

Prediction: the long-only churn reading falls short of the true phibar(63) by more than 0.10 on the
long/short sleeves (carry, CS momentum, value) and is unchanged on TSMomentum / baskets; long-only
snapshot persistence is biased (direction unknown) relative to the full book.

usage: python research/hidden_shorts_check.py <run_dir> <stacked_weights.csv>
"""
import sys
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "public_panel_run"
WFILE = sys.argv[2] if len(sys.argv) > 2 else "preqp_weights_stacked.csv"
D = 63


def long_only_filing(w):
    """The 13F view of a book: short positions dropped, the long positions at their filed scale.
    Renormalizing to unit gross would rescale each filing by that quarter's leverage and move the pooled churn
    reading even on a book with no shorts."""
    return np.clip(w, 0, None)


def churn_readings(Wv, Rv, idx, d=D):
    """corr(frozen-filing return over the next d days, actual full-book return), pooled over filings, for the full
    filing and for its long-only view. Wv, Rv: (T, N) weights and returns; idx: active-day positions."""
    fr_full, fr_lo, ac = [], [], []
    for k in range(0, len(idx) - d, d):
        seg = idx[k:k + d]; w0 = Wv[idx[k]]
        fr_full.append(Rv[seg] @ w0); fr_lo.append(Rv[seg] @ long_only_filing(w0)); ac.append(np.einsum("ij,ij->i", Wv[seg], Rv[seg]))
    fr_full, fr_lo, ac = map(np.concatenate, (fr_full, fr_lo, ac))
    return float(np.corrcoef(fr_full, ac)[0, 1]), float(np.corrcoef(fr_lo, ac)[0, 1])


def main():
    W = pd.read_csv(WFILE, index_col=0); W.index = pd.to_datetime(W.index)
    R = pd.read_csv(f"{RUN}/asset_returns_by_asset.csv", index_col=0); R.index = pd.to_datetime(R.index)
    R = R.clip(-0.5, 0.5)
    sleeves = sorted({c.split("||")[0] for c in W.columns})
    truth = pd.read_csv("phi_from_snapshots.csv").set_index("sleeve")
    rows = []
    for s in sleeves:
        cols = [c for c in W.columns if c.startswith(s + "||")]
        tick = [c.split("||")[1] for c in cols if c.split("||")[1] in R.columns]
        Ws = W[[f"{s}||{t}" for t in tick]].reindex(R.index).fillna(0.0); Ws.columns = tick
        Rs = R[tick].fillna(0.0)
        act = (Ws.abs().sum(axis=1) > 0).values
        Wv, Rv = Ws.values, Rs.values
        idx = np.where(act)[0]
        Sig = np.cov(Rv[act].T)
        short_share = float((-Wv[act].clip(max=0)).sum(axis=1).mean() / np.abs(Wv[act]).sum(axis=1).mean())
        snaps = idx[::D]
        full = Wv[snaps]
        longonly = long_only_filing(full)
        def phi_k(A, k):
            Q = A @ Sig; qa = np.einsum("ij,ij->i", Q, A)
            num = np.einsum("ij,ij->i", Q[:-k], A[k:]); den = np.sqrt(qa[:-k] * qa[k:]); ok = den > 0
            return float(np.mean(num[ok] / den[ok]))
        rec = {"sleeve": s, "short_share": short_share,
               "phi1q_full": phi_k(full, 1), "phi1q_longonly": phi_k(longonly, 1),
               "phi4q_full": phi_k(full, 4), "phi4q_longonly": phi_k(longonly, 4)}
        # churn detector: frozen filing (full vs long-only) against the actual full-book return
        rec["churn_full"], rec["churn_longonly"] = churn_readings(Wv, Rv, idx, D)
        rec["phibar63_true"] = float(truth.loc[s, "phibar63_daily"]) if s in truth.index else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv("hidden_shorts_check.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nread-out: churn_longonly - phibar63_true on long/short sleeves (prediction: below -0.10):")
    for r in df.itertuples():
        print(f" {r.sleeve:12s} short share {r.short_share:.2f} full-filing err {r.churn_full - r.phibar63_true:+.3f} long-only err {r.churn_longonly - r.phibar63_true:+.3f} | phi1q full {r.phi1q_full:.2f} long-only {r.phi1q_longonly:.2f}")


if __name__ == "__main__":
    main()
