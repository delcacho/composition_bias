"""
allocator_multiasset.py -- the risk-parity allocation experiment of allocator_risk_parity.py, extended
to a multi-asset book: the five commodity sleeves plus three single-stock equity factor sleeves
(momentum, value, profitability). Same question, same estimators, same metrics; the only structural
change is how holdings-implied covariance is built.

Motivation. The original forms the full asset covariance Sigma (k x k) and reads holdings-implied
covariance off W Sigma W'. That is fine for fourteen commodity contracts and impossible for ~9,000
stocks. The equivalent, tractable construction is the virtual-return projection: project each sleeve's
frozen book to a single daily P&L series v_i(s) = w_i . r(s), then take the small K x K covariance of
those series. For a book on the sample covariance this equals W Sigma W' exactly; for the equity
sleeves it is the only feasible path. Every sleeve, commodity and equity, is handled this way, so the
holdings-implied covariance is one K x K object regardless of how many underlying names each sleeve holds.

Estimators (halflife L): invvol, stream (EWMA of sleeve returns), pic_daily (holdings second moment),
c2_daily (D_stream . R_holdings . D_stream, the prescription), blend_daily (entrywise sqrt(phi) blend),
c2_/blend_q63lag60 (the N-PORT filing regime), oracle (forward realized covariance). Metrics, cost and
band grid, and the paired bootstrap are identical to the commodity script.

usage: python research/allocator_multiasset.py <commodity_run_dir> <stacked_intent_weights.csv>
       <equity_out_dir> [L] [REB] [--commodity-only] [--exclude-sleeve=NAME]
  --commodity-only drops the equity sleeves and must reproduce allocator_risk_parity.py (the gate).
  --exclude-sleeve=NAME drops one commodity sleeve (e.g. LongOnlyEW) for the tail robustness check.
writes allocator_multiasset_L{L}_reb{REB}.csv and ..._returns_L{L}_reb{REB}.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_factors"))

RUN = sys.argv[1] if len(sys.argv) > 1 else "panel_run"
WFILE = sys.argv[2] if len(sys.argv) > 2 else "panel_run/preqp_weights_stacked.csv"
EQOUT = sys.argv[3] if len(sys.argv) > 3 else "research/equity_factors/_out"
_ints = [a for a in sys.argv[4:] if a.isdigit()]
L = int(_ints[0]) if len(_ints) > 0 else 252
REB = int(_ints[1]) if len(_ints) > 1 else 1
COMMODITY_ONLY = "--commodity-only" in sys.argv
EXCLUDE_SLEEVE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--exclude-sleeve=")), None)  # drop one commodity sleeve (robustness)
BLOCK = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--block=")), 63) # bootstrap block length, days
PHI_NEUTRAL = "--phi=neutral" in sys.argv # market-neutral persistence feeds the blend weight (alternative)
EQ_MKT = None # equal-weight stock market series, set by load_equity
H = 21
ORACLE_H = max(REB, 10)
TARGET = 0.10
HMAX = 252
PERSIST_WINDOW = 1008
PERSIST_REFRESH = 21
MIN_PERSIST_ACTIVE = 252
H_BLEND = int(os.environ["H_BLEND"]) if os.environ.get("H_BLEND", "").isdigit() else min(max(REB, 1), HMAX)
FILING = (63, 45) # quarterly filing, 45-day lag (matches Exhibit ex:allocator)
SEED = 7


def corr_of(M):
    d = np.sqrt(np.maximum(np.diag(M), 1e-30)); return M / np.outer(d, d)


def psd_clip(C):
    C = np.where(np.isfinite(C), C, 0.0); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C); ev = np.maximum(ev, 1e-8); C2 = V @ np.diag(ev) @ V.T
    d = np.sqrt(np.diag(C2)); return C2 / np.outer(d, d)


def psd_clip_cov(C):
    """Nearest PSD covariance by flooring eigenvalues, keeping the variance scale (the variance blend
    carries holdings-implied volatilities in the diagonal, which psd_clip would strip)."""
    C = np.where(np.isfinite(C), C, 0.0); C = 0.5 * (C + C.T)
    ev, V = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12)
    return V @ np.diag(ev) @ V.T


def _risk_parity_builtin(Sig):
    n = len(Sig)
    if n == 1:
        return np.array([1.0])
    d = np.sqrt(np.diag(Sig)); C = Sig / np.outer(d, d)
    def f(w):
        return 0.5 * w @ C @ w - np.sum(np.log(w)) / n
    def g(w):
        return C @ w - 1.0 / (n * w)
    res = minimize(f, np.ones(n) / n, jac=g, method="L-BFGS-B", bounds=[(1e-6, None)] * n)
    w = res.x / d; return w / w.sum()


risk_parity = _risk_parity_builtin
KAPPA_MAX = 1e3
FLOOR_MULT = 2.0                                          # the floor on forecast book volatility: the allocation never sizes the book as if it were more than
#   FLOOR_MULT x as diversified as its n active sleeves would be uncorrelated. Scale cap on the unit-sum book = FLOOR_MULT*sqrt(n)
#   (5.66 at n=8), i.e. forecast book vol >= TARGET/(FLOOR_MULT*sqrt(n)) = 1.77% at n=8, half of 10%/sqrt(8). Online Appendix E.
def scale_cap(n_active):
    return FLOOR_MULT * np.sqrt(n_active)
BOOK_HL = 34      # halflife of the book's own realized-volatility level (the sleeves' own)
BOOK_SEED = 252   # days of the book's unit returns generated before its first rebalance, so the level is warm
BOOK_MIN = 63     # below this many unit-return days, fall back to the book's own covariance read
SLEEVE_CAP = 6.0                                          # per-sleeve vol-target cap


def condition(C):
    d = np.sqrt(np.maximum(np.diag(C), 1e-30)); Rm = C / np.outer(d, d)
    ev, V = np.linalg.eigh(Rm); ev = np.maximum(ev, ev.max() / KAPPA_MAX); Rm = V @ np.diag(ev) @ V.T
    dd = np.sqrt(np.diag(Rm)); Rm = Rm / np.outer(dd, dd)
    return np.outer(d, d) * Rm


def rc_spread(C, w):
    rc = w * (C @ w); rc = rc / rc.sum(); return float(rc.max() - rc.min())


def max_drawdown(r):
    c = np.cumprod(1 + r); peak = np.maximum.accumulate(c); return float((c / peak - 1).min())


def decay(h, phi_inf, tau):
    return phi_inf + (1.0 - phi_inf) * np.exp(-h / tau)


def load_commodity():
    W = pd.read_csv(WFILE, index_col=0); W.index = pd.to_datetime(W.index)
    R = pd.read_csv(f"{RUN}/asset_returns_by_asset.csv", index_col=0); R.index = pd.to_datetime(R.index)
    R = R.clip(-0.5, 0.5)
    sleeves = sorted({c.split("||")[0] for c in W.columns})
    assets = sorted({c.split("||")[1] for c in W.columns} & set(R.columns))
    R = R[assets].reindex(W.index).fillna(0.0)
    Wt = np.zeros((len(W), len(sleeves), len(assets)))
    for i, s in enumerate(sleeves):
        for j, a in enumerate(assets):
            c = f"{s}||{a}"
            if c in W.columns:
                Wt[:, i, j] = W[c].fillna(0.0).values
    return W.index, sleeves, assets, Wt, R.values


def vol_target_leverage(r, hl=34, target=TARGET, cap=SLEEVE_CAP):
    """Daily leverage to scale a raw sleeve to `target` annual vol from its own trailing EWMA vol,
    lagged one day and capped -- the convention the commodity sleeves already carry."""
    lam = 0.5 ** (1.0 / hl)
    r0 = np.nan_to_num(r, nan=0.0)
    v = np.zeros_like(r0); acc = np.nanvar(r0[:hl]) if len(r0) >= hl else np.nanvar(r0)
    for t in range(len(r0)):
        v[t] = acc
        acc = lam * acc + (1 - lam) * r0[t] ** 2
    sig = np.sqrt(np.maximum(v, 1e-12) * 252.0)
    lev = np.clip(target / sig, 0.0, cap)
    lev[:hl] = 0.0 # no leverage until the vol estimate has warmed
    return lev


def load_equity(dates):
    """Equity streams reindexed to `dates`; monthly virtual-return panels; snapshot index per day.
    Each equity sleeve is vol-targeted to 10% like the commodity sleeves; the same daily leverage
    scales its virtual returns, so the holdings-implied covariance is on the vol-targeted book."""
    streams = pd.read_csv(os.path.join(EQOUT, "ff_annual_returns.csv"), index_col=0, parse_dates=True)
    sl = ["momentum", "value", "quality"]
    streams = streams[sl].reindex(dates)
    snaps = pd.read_parquet(os.path.join(EQOUT, "equity_holdings_monthly.parquet"))
    sr = pd.read_parquet(os.path.join(EQOUT, "equity_stock_returns.parquet"))
    # dense T x n_stock pivot filled directly: equivalent to drop_duplicates(date, gvkey) -> unstack -> reindex(dates)
    # (first occurrence wins, gvkeys sorted, dates off the calendar dropped), without the intermediate copies
    codes, cols = pd.factorize(sr["gvkey"], sort=True); cols = pd.Index(cols, name="gvkey")
    di = pd.Index(dates).get_indexer(sr["date"])
    first = ~pd.DataFrame({"d": sr["date"].to_numpy(), "g": codes}).duplicated(keep="first").to_numpy()
    ok = first & (di >= 0)
    Rz = np.full((len(dates), len(cols)), np.nan)
    Rz[di[ok], codes[ok]] = sr["ret"].to_numpy(dtype="float64")[ok]
    del sr, codes, di, first, ok
    col_ix = {g: i for i, g in enumerate(cols)}
    np.nan_to_num(Rz, copy=False, nan=0.0) # T x n_stock; already screened
    global EQ_MKT
    _cnt = np.maximum((Rz != 0).sum(axis=1), 1); EQ_MKT = Rz.sum(axis=1) / _cnt # equal-weight market, the common factor
    # and winsorized at source (build_ff_annual.load_daily, per Ince-Porter)
    panels, snap_idx = {}, {}
    for s in sl:
        g = snaps[snaps["sleeve"] == s]
        sds = sorted(g["snap_date"].unique())
        V = np.zeros((len(dates), len(sds)))
        for m, sd in enumerate(sds):
            book = g[g["snap_date"] == sd]
            idx = [col_ix[gv] for gv in book["gvkey"] if gv in col_ix]
            w = book.loc[book["gvkey"].isin(cols), "w"].to_numpy()
            if len(idx):
                V[:, m] = Rz[:, idx] @ w
        panels[s] = V
        sd_ts = pd.to_datetime(sds)
        si = np.searchsorted(sd_ts.values, dates.values, side="right") - 1 # latest snapshot <= day
        snap_idx[s] = si
    lev = np.column_stack([vol_target_leverage(streams[s].to_numpy()) for s in sl]) # T x 3
    return sl, streams.values, panels, snap_idx, lev


def main():
    cdates, csleeves, assets, Wt, Rv = load_commodity()
    if EXCLUDE_SLEEVE:
        _keep = [i for i, s in enumerate(csleeves) if s != EXCLUDE_SLEEVE]
        assert len(_keep) == len(csleeves) - 1, f"--exclude-sleeve: {EXCLUDE_SLEEVE} not among {csleeves}"
        csleeves = [csleeves[i] for i in _keep]; Wt = Wt[:, _keep, :]
    if COMMODITY_ONLY:
        dates = cdates; esleeves = []; estream = np.zeros((len(dates), 0)); panels = {}; snap_idx = {}; elev = np.zeros((len(dates), 0))
    else:
        esleeves, estream_full, panels, snap_idx, elev_full = load_equity(cdates)
        # common calendar: days with equity data (streams start mid-2001)
        have = np.isfinite(estream_full).any(axis=1)
        keep = np.where(have)[0]
        lo, hi = keep.min(), keep.max() + 1
        dates = cdates[lo:hi]
        Wt = Wt[lo:hi]; Rv = Rv[lo:hi]
        estream = np.nan_to_num(estream_full[lo:hi], nan=0.0) * elev_full[lo:hi] # vol-targeted equity stream
        elev = elev_full[lo:hi]
        panels = {s: panels[s][lo:hi] for s in esleeves}
        snap_idx = {s: snap_idx[s][lo:hi] for s in esleeves}
        global EQ_MKT
        if EQ_MKT is not None:
            EQ_MKT = EQ_MKT[lo:hi] # same calendar as the panels
    sleeves = list(csleeves) + [f"eq_{s}" for s in esleeves]
    nc, ne = len(csleeves), len(esleeves); n = nc + ne; T = len(dates)

    # sleeve returns: commodity = Wt.R (daily book), equity = the drifting-book stream
    rs = np.zeros((T, n))
    rs[:, :nc] = np.einsum("tij,tj->ti", Wt, Rv)
    if ne:
        rs[:, nc:] = estream
    active = np.ones((T, n), dtype=bool)
    active[:, :nc] = np.abs(Wt).sum(axis=2) > 0
    if ne: # an equity sleeve is active once its stream prints
        active[:, nc:] = np.abs(estream) > 0

    print(f"commodity run: {RUN}\nequity: {'(excluded)' if COMMODITY_ONLY else EQOUT}\n"
          f"sleeves={n} ({nc} commodity + {ne} equity) days={T} L={L} REB={REB} "
          f"{dates.min().date()}..{dates.max().date()}")
    print("sleeve ann. vol: " + ", ".join(
        f"{s} {rs[active[:, i], i].std() * np.sqrt(252):.3f}" for i, s in enumerate(sleeves)))

    # Holdings covariance: an EWMA over the full history of every column a book can hold (the commodity assets and every
    # equity snapshot's replayed return), updated daily in the main loop, the same unnormalized recursion from day 0 as
    # allocator_risk_parity.py. A book as of `asof` is a row of loadings A on those columns and its covariance is A C A'.
    eq_off = []; dimZ = Rv.shape[1]
    for s in esleeves:
        eq_off.append(dimZ); dimZ += panels[s].shape[1]
    Cz = np.zeros((dimZ, dimZ))

    def z_row(t):
        return np.concatenate([Rv[t]] + [panels[s][t] for s in esleeves]) if ne else Rv[t]

    def holdings_cov(asof):
        A = np.zeros((n, dimZ)); A[:nc, :Rv.shape[1]] = Wt[asof]
        for j, s in enumerate(esleeves):
            m = snap_idx[s][asof]
            if m >= 0:
                A[nc + j, eq_off[j] + m] = elev[asof, j] # vol-targeted frozen book
        return A @ Cz @ A.T

    # persistence phi(h): estimated causally on trailing windows ending at the forecast date.
    # Commodity sleeves use risk-space cosines on the asset book; equity sleeves use their
    # virtual-return Gram over snapshots available by that date.
    from phi_from_snapshots import phi_series
    import pair_weight as pw
    persist_cache = {}

    def trailing_persistence(t):
        key = (t // PERSIST_REFRESH) * PERSIST_REFRESH
        if key in persist_cache:
            return persist_cache[key]
        s0 = max(0, key - PERSIST_WINDOW + 1)
        phibar_t, aret_t = {}, {}
        for i in range(nc):
            act = active[s0:key + 1, i]
            if act.sum() < MIN_PERSIST_ACTIVE:
                continue
            A = Wt[s0:key + 1, i, :][act]
            Rw = Rv[s0:key + 1][act]
            Sig_s = np.cov(Rw.T); keep = np.abs(A).sum(axis=0) > 0
            if PHI_NEUTRAL:
                ev, U = np.linalg.eigh(Sig_s); Sig_s = Sig_s - ev[-1] * np.outer(U[:, -1], U[:, -1])
            if keep.sum() < 1:
                continue
            ph = phi_series(A[:, keep], Sig_s[np.ix_(keep, keep)], HMAX)
            phibar_t[sleeves[i]] = {h: float(np.nanmean(ph[1:h + 1])) for h in range(1, HMAX + 1)}
            aret_t[sleeves[i]] = pw.retention(ph)
        for j, s in enumerate(esleeves):
            V = panels[s]
            mmax = snap_idx[s][key] if key < len(snap_idx[s]) else -1
            if mmax < 2:
                continue
            cols = np.arange(mmax + 1)
            P = V[s0:key + 1][:, cols]
            if PHI_NEUTRAL and EQ_MKT is not None:
                f = EQ_MKT[s0:key + 1]; P = P - np.outer(f, (f @ P) / max(f @ f, 1e-30))
            if P.shape[0] < MIN_PERSIST_ACTIVE:
                continue
            G = (P.T @ P) / P.shape[0]; d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
            C = G / np.outer(d, d); M = C.shape[0]
            lags = np.array([21 * k for k in range(1, min(M, 12))])
            vals = np.array([np.nanmean([C[a, a + k] for a in range(M - k)]) for k in range(1, len(lags) + 1)])
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            lags = lags[:len(vals)]
            try:
                p, _ = curve_fit(decay, lags.astype(float), vals, p0=[max(vals.min(), 0.0), 60.0],
                                 bounds=([-1, 1], [1, 5000]), maxfev=20000)
                phi_h = np.array([np.nan] + [decay(h, *p) for h in range(1, HMAX + 1)])
            except Exception:
                continue
            phibar_t[f"eq_{s}"] = {h: float(np.nanmean(phi_h[1:h + 1])) for h in range(1, HMAX + 1)}
            aret_t[f"eq_{s}"] = pw.retention(np.concatenate([[1.0], phi_h[1:]]))
        persist_cache[key] = (phibar_t, aret_t)
        return persist_cache[key]

    def pair_w(t, ix, h, s=0):
        """Model pair weights over horizon h from persistence known at date t."""
        _, aret_t = trailing_persistence(t)
        a = np.nan_to_num(np.array([aret_t.get(sleeves[i], 0.0) for i in ix], dtype=float), nan=0.0) # an undefined retention gets weight 0
        return pw.weight_matrix(a, h, s) # (the stream); a NaN weight would zero the variance blend's covariance

    # Holdings covariance is computed at every day (not every REB): the covariance at a date does not
    # depend on the rebalance phase, so all phases reuse it. Stream is a K-dim EWMA (exact).
    lam = 0.5 ** (1.0 / L)
    outer = np.einsum("ti,tj->tij", rs, rs)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)], axis=0)
    S = np.zeros((n, n)); warm = L; fwd = max(H, ORACLE_H)
    est_names = ["invvol", "stream", "pic_daily", "c2_daily", "blend_daily", "varblend_daily",
                 "c2_q63lag45", "blend_q63lag45", "varblend_q63lag45", "oracle"]
    covs = {nm: {} for nm in est_names}
    fdelta, flag = FILING; stale = fdelta + flag
    for t in range(T):
        S = lam * S + (1 - lam) * np.outer(rs[t], rs[t])
        zt = z_row(t); Cz *= lam; Cz += (1 - lam) * np.outer(zt, zt)
        if t < warm or t >= T - fwd:
            continue
        act = active[t]
        if act.sum() < 2:
            continue
        ix = np.where(act)[0]
        Ss = S[np.ix_(ix, ix)]; Ds = np.sqrt(np.diag(Ss)); Rst = corr_of(Ss)
        G = holdings_cov(t)[np.ix_(ix, ix)]
        Pd = pair_w(t, ix, H_BLEND) # model pair weight at the rebalance horizon
        covs["invvol"][t] = np.diag(np.diag(Ss)); covs["stream"][t] = Ss; covs["pic_daily"][t] = G
        covs["c2_daily"][t] = np.outer(Ds, Ds) * corr_of(G)
        covs["blend_daily"][t] = np.outer(Ds, Ds) * psd_clip(Pd * corr_of(G) + (1 - Pd) * Rst)
        covs["varblend_daily"][t] = psd_clip_cov(Pd * G + (1 - Pd) * Ss)
        if t < flag:
            # no filing has cleared its lag yet: an allocator reading filings has nothing, so the filing rows are the stream
            Gs = Ss
        else:
            asof = fdelta * ((t - flag) // fdelta) # most recent quarterly filing, lagged: steps every 63 days
            Gs = holdings_cov(asof)[np.ix_(ix, ix)]
            if not np.all(np.diag(Gs) > 0):
                Gs = G
        Rsnap = corr_of(Gs)
        P = pair_w(t, ix, H_BLEND, stale) # model weight: filing `stale` days old, horizon H_BLEND
        covs["c2_q63lag45"][t] = np.outer(Ds, Ds) * Rsnap
        covs["blend_q63lag45"][t] = np.outer(Ds, Ds) * psd_clip(P * Rsnap + (1 - P) * Rst)
        covs["varblend_q63lag45"][t] = psd_clip_cov(P * Gs + (1 - P) * Ss)
        Fo = (cum[t + ORACLE_H] - cum[t]) / ORACLE_H
        Fo = Fo[np.ix_(ix, ix)]
        if np.all(np.diag(Fo) > 0) and np.all(np.isfinite(Fo)):
            covs["oracle"][t] = Fo

    # ---- phase-averaged allocation: run every rebalance offset and take the median metric, as
    # Exhibit ex:allocator does (a single phase is a calendar lottery). Covariances above are per-day
    # so all phases reuse them; the allocation given covs is cheap.
    def phases_for(reb):
        if reb <= 1:
            return [0]
        return list(range(reb))

    def run_phase(reb_dates):
        res = {}
        for nm in est_names:
            F = np.full((T, n), np.nan); pr = np.zeros(T); prev = None; turn = []
            # book-level sizing: each book (oracle excepted) targets the EWMA volatility of its own unit
            # daily returns (halflife BOOK_HL), seeded from a pre-grid so the level is warm at the first
            # rebalance; the estimator under test then enters only through its correlations (Online Appendix E).
            grid = list(reb_dates)
            if nm != "oracle":
                grid = [t for t in range(reb_dates[0] - REB, reb_dates[0] - BOOK_SEED - 1, -REB) if t >= 0][::-1] + grid
            lam_b = 0.5 ** (1.0 / BOOK_HL); var_b = 0.0; wsum_b = 0.0; n_b = 0; w_prev = ix_prev = t_prev = None
            for t in grid:
                if t not in covs[nm]:
                    continue
                ix = np.where(active[t])[0]
                try:
                    C = condition(covs[nm][t]); w = risk_parity(C)
                except np.linalg.LinAlgError:
                    continue
                if nm != "oracle":
                    if w_prev is not None:
                        for u in rs[t_prev + 1:t + 1][:, ix_prev] @ w_prev:   # unit returns of the book held since the last rebalance
                            var_b = lam_b * var_b + (1 - lam_b) * u * u; wsum_b = lam_b * wsum_b + (1 - lam_b); n_b += 1
                    w_prev, ix_prev, t_prev = w, ix, t
                    if t not in reb_dates:
                        continue                                              # seed grid: build history only, no position
                ex_ante = np.sqrt(252 * w @ C @ w)
                if nm != "oracle" and n_b >= BOOK_MIN:
                    ex_ante = np.sqrt(252 * var_b / wsum_b)                   # the book's own realized volatility sets the level
                sc = min(TARGET / ex_ante, scale_cap(len(ix)))
                full = np.zeros(n); full[ix] = w * sc; F[t] = full
                if prev is not None:
                    turn.append(np.abs(full - prev).sum())
                prev = full
                seg = rs[t + 1:t + 1 + REB]; pr[t + 1:t + 1 + REB] = seg @ full
            r = pr[reb_dates[0] + 1:reb_dates[-1] + 1 + REB]
            sd = r.std() * np.sqrt(252); sh = r.mean() * 252 / sd if sd > 0 else np.nan
            # delivered risk of the portfolio as run: realized vol on non-overlapping 21-day windows
            nwin = len(r) // H; dv = np.sqrt(252) * r[:nwin * H].reshape(nwin, H).std(axis=1, ddof=0) / TARGET
            ldv = np.log(dv[dv > 0])
            Ff = F.copy(); ok = np.isfinite(Ff).all(axis=1); first = int(np.argmax(ok))
            for t in range(first + 1, T):
                if not ok[t]:
                    Ff[t] = Ff[t - 1]
            prc = np.zeros(T); prc[1:] = np.einsum("ti,ti->t", Ff[:-1], rs[1:])
            prevw = np.zeros(n); tv = np.zeros(T)
            for t in range(first, T - H):
                tv[t] = np.abs(Ff[t] - prevw).sum(); prevw = Ff[t]
            rr = prc[first + 1:T - H]; tvv = tv[first:T - H - 1]; rn = rr - 5e-4 * tvv
            res[nm] = {"pr": pr, "F": F, "delivered_median": float(np.exp(np.median(ldv))) if len(ldv) else np.nan,
                       "dispersion": float(ldv.std()) if len(ldv) else np.nan, "max_dd": max_drawdown(r),
                       "ann_turnover": float(np.mean(turn) * 252 / REB) if turn else 0.0, "sharpe": sh,
                       "sharpe_5bp": float(rn.mean() / rn.std() * np.sqrt(252)) if rn.std() > 0 else np.nan}
        return res

    phases = phases_for(REB)
    per_phase = [run_phase(list(range(warm + p, T - fwd, REB))) for p in phases]
    out0 = {nm: pd.Series(per_phase[0][nm]["pr"], index=dates) for nm in est_names}
    METR = ["delivered_median", "dispersion", "max_dd", "ann_turnover", "sharpe", "sharpe_5bp"]
    metrics = []
    for nm in est_names:
        rec = {"estimator": nm}
        for k in METR:
            rec[k] = float(np.nanmedian([per_phase[pi][nm][k] for pi in range(len(phases))]))
        metrics.append(rec)
    M = pd.DataFrame(metrics).set_index("estimator")

    # bootstrap the Sharpe difference vs the stream on phase 0
    rng = np.random.default_rng(SEED); base = out0["stream"].values
    reb0 = list(range(warm + phases[0], T - fwd, REB))
    lo2, hi2 = reb0[0] + 1, reb0[-1] + 1 + REB; nb = max(1, (hi2 - lo2) // BLOCK) # the phase-0 Sharpe window itself,
    # as allocator_risk_parity.py resamples its own Sharpe window
    for nm in est_names:
        x = out0[nm].values[lo2:hi2]; y = base[lo2:hi2]; diffs = []
        for _ in range(1000):
            starts = rng.integers(0, len(x) - BLOCK, nb); idx = np.concatenate([np.arange(s0, s0 + BLOCK) for s0 in starts])
            diffs.append(x[idx].mean() / x[idx].std() * np.sqrt(252) - y[idx].mean() / y[idx].std() * np.sqrt(252))
        M.loc[nm, "d_sharpe_vs_stream"] = M.loc[nm, "sharpe"] - M.loc["stream", "sharpe"]
        M.loc[nm, "d_sharpe_ci_lo"], M.loc[nm, "d_sharpe_ci_hi"] = np.quantile(diffs, [0.025, 0.975])

    # ---- cost and rebalance-band sensitivity (Exhibit ex:costs): re-execute each estimator's stored
    # target weights, holding a sleeve until its target moves by more than BAND of NAV, charging COST
    # per unit of notional traded. Daily runs sweep bands; the monthly clock reports band 0 only, the
    # median over the REB phases. Sharpe convention matches run_phase (net excess mean / net vol).
    BANDS = [0.0, 0.05, 0.10] if REB == 1 else [0.0]
    COSTS_BP = [0, 2, 5, 10, 20]
    rows_c = []
    for nm in est_names:
        for band in BANDS:
            phase_recs = []
            for pi in range(len(phases)):
                F = per_phase[pi][nm]["F"].copy()
                ok = np.isfinite(F).all(axis=1)
                if not ok.any():
                    continue
                first = int(np.argmax(ok))
                for t in range(first + 1, T): # hold last solved weights between rebalances
                    if not ok[t]:
                        F[t] = F[t - 1]
                held = np.zeros_like(F); prev = np.zeros(n); turn = np.zeros(T)
                for t in range(first, T - fwd):
                    tgt = F[t]
                    new = np.where(np.abs(tgt - prev) > band, tgt, prev) if band > 0 else tgt
                    turn[t] = np.abs(new - prev).sum(); held[t] = new; prev = new
                pr_c = np.zeros(T); pr_c[1:] = np.einsum("ti,ti->t", held[:-1], rs[1:])
                r = pr_c[first + 1:T - fwd]; tv = turn[first:T - fwd - 1]
                nwin = len(r) // H; dv = np.sqrt(252) * r[:nwin * H].reshape(nwin, H).std(axis=1, ddof=0) / TARGET
                ldv = np.log(dv[dv > 0])
                rec = {"ann_turnover": float(tv.mean() * 252), "dispersion": float(ldv.std()) if len(ldv) else np.nan}
                for c in COSTS_BP:
                    rn = r - c / 1e4 * tv
                    rec[f"sharpe_{c}bp"] = float(rn.mean() / rn.std() * np.sqrt(252)) if rn.std() > 0 else np.nan
                phase_recs.append(rec)
            if not phase_recs:
                continue
            agg = {"estimator": nm, "band": band}
            for k in phase_recs[0]:
                agg[k] = float(np.nanmedian([p[k] for p in phase_recs]))
            rows_c.append(agg)
    tag = f"_L{L}_reb{REB}" + ("_comm" if COMMODITY_ONLY else "") + ("_phineutral" if PHI_NEUTRAL else "") + (f"_block{BLOCK}" if BLOCK != 63 else "")
    C = pd.DataFrame(rows_c)
    C.to_csv(f"allocator_multiasset_costs{tag}.csv", index=False, float_format="%.4g")
    M.to_csv(f"allocator_multiasset{tag}.csv", float_format="%.5g")

    # ---- per-offset daily returns for the drawdown attribution (allocator_mdd_attribution.py).
    # Max drawdown is reported as the median over rebalance offsets, so the attribution needs every
    # offset's daily path, not one. composition_intensity is the divergence signal: the mean absolute
    # gap between the holdings correlation and the stream correlation over the active sleeve pairs,
    # computed from the covariances built above and independent of the offset. Writes one parquet;
    # the metric and cost tables are unchanged.
    if not COMMODITY_ONLY:
        ci = pd.Series(np.nan, index=dates)
        for t in covs["stream"]:
            if t in covs["pic_daily"]:
                Rs_ = corr_of(covs["stream"][t]); Rh_ = corr_of(covs["pic_daily"][t])
                m = Rs_.shape[0]
                if m >= 2:
                    iu = np.triu_indices(m, 1)
                    ci.iloc[t] = float(np.mean(np.abs(Rh_[iu] - Rs_[iu])))
        att_est = ["stream", "pic_daily", "c2_daily", "blend_daily", "invvol"]
        frames = []
        for pi, ph in enumerate(phases):
            fr = pd.DataFrame({nm: per_phase[pi][nm]["pr"] for nm in att_est}, index=dates)
            fr.insert(0, "phase", ph); frames.append(fr)
        allp = pd.concat(frames)
        allp["composition_intensity"] = ci.reindex(allp.index)
        allp.to_parquet(f"allocator_multiasset_phasedaily{tag}.parquet")

    pd.set_option("display.width", 250)
    print(f"\n=== net of costs: Sharpe by cost per unit traded (bp) and rebalance band ===")
    print(C.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n=== multi-asset risk parity, every {REB} day(s), {len(phases)} phase(s), {TARGET:.0%} target ===")
    cols = ["delivered_median", "dispersion", "max_dd", "ann_turnover", "sharpe", "sharpe_5bp",
            "d_sharpe_vs_stream", "d_sharpe_ci_lo", "d_sharpe_ci_hi"]
    print(M[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    phibar_last, _ = trailing_persistence(T - fwd - 1)
    print("\ntrailing phibar(63) by sleeve at last evaluation date: " + ", ".join(
        f"{s} {phibar_last[s][63]:.2f}" for s in sleeves if s in phibar_last))
    print(f"\nwrote allocator_multiasset{tag}.csv")


if __name__ == "__main__":
    main()
