"""
etf_staleness_experiment.py -- the staleness experiment on the active-ETF daily-holdings panel: two covariance
forecasters compared on every fund pair at every age of the book the allocator is allowed to see, pooled across
strategy classes and stratified by persistence.

Unit and estimand. A fund pair at a monthly forecast date t. The target is the realized covariance of the two
clones over the next H days, priced from the books the funds actually held on each of those days (rotating,
never frozen), so no estimator shares the target's construction. The clone is the priced book of day s held
over day s+1, renormalized to the priced gross (etf_race.clone_returns).

Filing age, within pair-date. Everything is held fixed (pair, date, horizon, target, asset covariance) and only
the age of the book the allocator is allowed to see varies: today's, 21 days old, 63 days old, 63 plus a 45-day
lag. The return stream (EWMA of the clones' own returns, halflife L) is the fixed reference. The estimators at
each age: holdings covariance H (book replayed through the trailing asset returns, W Sigma_t W'), holdings
correlation with stream volatilities C (omega = 1), the correlation blend B at the ex-ante pair weight of
eq:pairweight shifted by the age, and the stream S. Positive control: the future book at the middle of the
horizon (t + h/2, the article's oracle), which should beat today's book if the machinery detects rotation at all.

No look-ahead. The pricing covariance is the trailing EWMA replay (as in the article). The retention a_i behind
the pair weight is fitted through the origin on the fund's trailing persistence curve at t (phi(u) from its own
books over the last PW trading days, in the geometry of the trailing replay: Gram of the replayed books under
the trailing weights). The market-neutral geometry removes an equal-weight market factor of the priced stocks:
each clone's beta is fitted on the trailing window only (days the clone exists) and the same hedge is applied
to the stream, to the replayed book of every age and forward to the target (a hedge ratio known at t). The
estimators are etf_race.stream_and_target and etf_race.holdings_covs, shared with the covariance comparison.

Strata. Pairs are stratified by the persistence of the pair's less persistent fund (phimin), measured as the
full-sample market-neutral phibar(63) of etf_results/persistence_daily.csv; it orders the strata and enters no
forecast. Two cuts are reported: the article's fixed bands (0.75 / 0.85 / 0.95) and quartiles of the pooled
phimin distribution (Q1 = most rotating, Q4 = most frozen).

Controls. The frozen stratum (phimin above 0.95, Q4): no age effect expected. The future book: positive control.
Clone-vs-NAV fidelity: reported from etf_results/fidelity.csv where it exists, never used to select.

Inference. Pair-dates are dependent (a fund sits in every pair it belongs to, windows overlap, dates share
shocks). Effects are estimated by within-pair-date differences of squared error, scaled by the sampling noise of
the target, regressed on stratum x age dummies with standard errors two-way clustered by the pair's less
persistent fund and by date. Pooled RMSE ratios per stratum x age with fund-clustered bootstrap intervals, and
the persistence at which the stale book's ratio to the stream crosses one, are reported alongside.

Design. One pooled comparison across the strategy classes (allocation, mf = managed futures, longshort,
longonly; fund_classes). Pairs are formed within a class only (a cross-class pair carries almost no covariance
signal); the per-class runs are the computation stage, the pooled analysis the reported one
(build_pooled_chunks, analyze). Incomplete books (a subsidiary sleeve hidden from the daily book) are admitted
and flagged; --complete-only drops them.

Sample split. Funds are split A/B by the parity of the SHA-1 of the ticker: --split=A, --split=B, or
--split=all for the full sample (the reported results).

Outputs etf_results/experiment/<class>_<geom>_<split>[_h<h>][_L<L>]/chunks/*.parquet pair-date-age
         observations per class, one file per date, plus persistence_trailing.csv;
         etf_results/experiment/pooled_<geom>_<split>[_h<h>][_L<L>]/ with rmse_by_fixedband_age.csv,
         rmse_by_quartile_age.csv, rmse_by_completeness_age.csv, panel_regressions.csv, crossover.csv,
         sizing_by_quartile_age.csv, and a printed report
Usage python -u research/etf_staleness_experiment.py [--split=A|B|all] [--geom=risk|neutral|both] [--h=21] [--L=252]
         [--class=allocation|longonly|mf|longshort|all] [--complete-only] [--analyze-only] [--out=dir] [--longonly-cap=N]
         (valued flags are --name=value; a space form like "--split A" is ignored and the default is used)
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etf_coverage as ec # noqa: E402
import etf_factor_benchmark as fb  # noqa: E402  return-based factor benchmarks (statistical and named proxy sets)
import etf_race as er # noqa: E402
import pair_weight as pw # noqa: E402
import return_availability as ra # noqa: E402 missing returns as missing, not zeros

H = 21 # primary horizon (63 as robustness via --h)
AGES = [0, 21, 63, 108] # filing ages: today, month, quarter, quarter + 45-day lag
FUTURE = H // 2 # positive control: the mid-horizon book, FUTURE days ahead of t (set from --h in main)
L = 252 # EWMA halflife: stream covariance and the pricing replay (article convention)
BACK = 6 * L # replay window
PW = 504 # trailing window for persistence (two years of the fund's own books)
PHI_LAGS = [1, 5, 21, 63] # lags of the trailing persistence curve
MIN_PHI_BOOKS = 63 + 40 # books needed inside the trailing window to measure phi(63)
STRIDE = 21
N_BOOT = 1000
OUT = "etf_results/experiment"


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def split_of(ticker):
    return "A" if int(hashlib.sha1(ticker.encode()).hexdigest(), 16) % 2 == 0 else "B"


def corr_of(M):
    d = np.sqrt(np.clip(np.einsum("ii->i", M), 1e-30, None))
    return M / np.outer(d, d)


def book_at(books, t, date, age):
    """The fund's book `age` trading days before `date` (age < 0: after; the positive control)."""
    return er.book_at(books, t, date, age)


def persistence_window(books, t, date):
    """(lo, hi) rows of the fund's books inside the last PW trading days ending at `date`, or None when too few."""
    dates = books[t][0]
    hi = dates.searchsorted(date, side="right"); lo = max(0, hi - PW)
    return (lo, hi) if hi - lo >= MIN_PHI_BOOKS else None


def trailing_persistence(books, t, date, Sig_p, psup):
    """phi(u) at the lags PHI_LAGS from the fund's books inside the last PW trading days ending at `date`, in the
    geometry of the trailing replay: cosine of the books under Sig_p, the pairwise-normalized trailing covariance on the
    sorted support psup (the same estimate as the holdings covariance; the coverage-scaled replay shrank partly covered
    names). Returns (phibar63, a, curve) or (nan, nan, None) when the window holds too few books; a is the retention
    through the origin (er.retention_through_origin)."""
    win = persistence_window(books, t, date)
    if win is None:
        return np.nan, np.nan, None
    Wf = books[t][1][win[0]:win[1]]
    cols = np.unique(Wf.indices); p = np.searchsorted(psup, cols)
    assert len(cols) == 0 or (p < len(psup)).all() and (psup[np.minimum(p, len(psup) - 1)] == cols).all(), "support mismatch"
    Wd = Wf[:, cols].toarray().astype(np.float64)
    G = Wd @ np.asarray(Sig_p[np.ix_(p, p)], np.float64) @ Wd.T # Gram of the books under Sigma_t
    dg = np.diag(G); nrm = np.sqrt(np.where(dg > 0, dg, np.nan)) # pairwise estimate not PSD: undefined, skipped
    curve = np.full(max(PHI_LAGS) + 1, np.nan); curve[0] = 1.0
    for u in PHI_LAGS:
        if G.shape[0] > u + 20:
            v = np.diag(G, u) / (nrm[:-u] * nrm[u:])
            curve[u] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    if not np.isfinite([curve[u] for u in PHI_LAGS]).all():
        return np.nan, np.nan, None
    grid = np.interp(np.arange(1, 64), [u for u in PHI_LAGS], [curve[u] for u in PHI_LAGS])
    a = er.retention_through_origin(curve)
    return float(grid.mean()), float(a) if np.isfinite(a) else np.nan, curve


def weight_matrix(a, h, s=0):
    """eq:pairweight for every pair, vectorized: q = a_i a_j, omega = q (1 - q^h) / (h (1 - q)), times q^s for a
    book s days old; nan where a retention is missing (pair_weight.pair_weight_stale, element by element)."""
    a = np.asarray(a, float)
    q = np.clip(np.outer(a, a), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(q >= 1.0, 1.0, q * (1.0 - q ** h) / (h * (1.0 - q)))
    W = np.where(q <= 0.0, 0.0, W) * (q ** s if s else 1.0)
    return np.where(np.isfinite(np.outer(a, a)), W, np.nan)


_DAILY_PHI = None


def daily_phibar():
    """Full-sample market-neutral phibar(63) per fund from etf_results/persistence_daily.csv: the measure that orders
    the persistence strata (and that the divergence check uses). It is defined for every fund, unlike the in-run
    trailing cosine, which needs MIN_PHI_BOOKS books inside the window and is undefined for young or sparse funds."""
    global _DAILY_PHI
    if _DAILY_PHI is None:
        p = f"{er.OUT}/persistence_daily.csv"
        s = pd.read_csv(p, index_col=0)["phibar_neutral_63"] if os.path.exists(p) else pd.Series(dtype=float)
        s = pd.to_numeric(s, errors="coerce"); s.index = s.index.astype(str).str.upper()
        _DAILY_PHI = s
    return _DAILY_PHI


def quartiles_at(phib, active):
    """Quartile 1..4 of trailing persistence among the active funds present (1 = fastest); 0 elsewhere."""
    q = np.zeros(len(phib), int)
    ix = np.where(active & np.isfinite(phib))[0]
    if len(ix) >= 8:
        r = pd.Series(phib[ix]).rank(method="first").values
        q[ix] = np.ceil(4 * r / len(ix)).astype(int)
    return q


# ---------------------------------------------------------------- the experiment, one geometry
def run_geometry(geom, keep, books, Rs, clones, f_mkt, placebo, out_dir, h, incomplete=frozenset()):
    idx = Rs.index; T = len(idx); n = len(keep)
    os.makedirs(f"{out_dir}/chunks", exist_ok=True)
    RAW = Rs.values; N = RAW.shape[1]
    A = ra.availability(RAW) # which names have a return each day (a zero fill is not data)
    rv_raw = clones[keep].reindex(idx).values.astype(np.float32); have = np.isfinite(rv_raw)
    # neutral: equal-weight market factor; stream, holdings replay and target all hedged with the clone's trailing beta
    f_all = np.nan_to_num(np.asarray(f_mkt, np.float32)) if geom == "neutral" else None
    is_plc = np.array([t in placebo for t in keep])
    is_inc = np.array([t in incomplete for t in keep]) # daily book hides a subsidiary sleeve: stratified, not excluded
    eval_t = [ti for ti in range(L, T - h) if ti % STRIDE == 0]
    t0 = time.time(); pers_rows = []; ever_active = np.zeros(n, bool)
    iu = np.triu_indices(n, 1)
    print(f"[{geom}] {n} funds ({int(is_plc.sum())} index-fund controls), {len(eval_t)} monthly dates, horizon {h}", flush=True)
    for k, ti in enumerate(eval_t, 1):
        t = idx[ti]
        s0, wts = er.ewma_window(ti, L, BACK)
        # a fund enters a date if its clone has a return inside the estimation window; adequacy is enforced downstream
        # by the estimator masks and the forward-target rule (er.stream_and_target)
        active = have[s0:ti + 1].any(axis=0)
        ever_active |= active
        # stream So, target Fo over ti+1..ti+h, clone betas (neutral): er.stream_and_target, shared with the covariance comparison
        So, Fo, beta_c = er.stream_and_target(rv_raw, s0, ti, wts, h, f_all)
        D = np.sqrt(np.clip(np.diag(So), 1e-30, None)); Sc = corr_of(So); Fc = corr_of(Fo)
        # one pairwise-normalized covariance per date on the union of the books the estimators and the persistence read
        by_age = er.books_by_age(books, keep, t, AGES + [-FUTURE], active, N)
        esup = er.book_support(by_age)
        psup = [books[tk][1][w[0]:w[1]].indices for i, tk in enumerate(keep)
                if active[i] and (w := persistence_window(books, tk, t)) is not None]
        U = np.union1d(esup, np.concatenate(psup + [np.zeros(0, np.int64)])).astype(np.int64)
        # (neutral: the covariance of the names' residuals on the factor, betas over observed days; the persistence Gram
        # reads it as is, the holdings replay adds the clone-beta hedge term)
        Sig_U, priced_U, factor = er.replay_covariance(RAW, A, s0, ti, wts, U, f_all)
        # retention (trailing, ex ante) for the causal blend weight; persistence for the ordering of the strata from the
        # full-sample market-neutral phibar (daily_phibar), defined for every fund, so the MIN_PHI_BOOKS gate does not thin them
        phib = np.full(n, np.nan); aret = np.full(n, np.nan); dphi = daily_phibar()
        for i, tk in enumerate(keep):
            if active[i]:
                _, a_i, _ = trailing_persistence(books, tk, t, Sig_U, U)
                aret[i] = a_i if np.isfinite(a_i) else 1.0 # retention unfittable (< MIN_PHI_BOOKS books: young/sparse) -> a=1
                phib[i] = float(dphi.get(str(tk).upper(), np.nan))
        q = quartiles_at(phib, active & ~is_plc)
        for i, tk in enumerate(keep):
            if active[i] and np.isfinite(phib[i]):
                pers_rows.append({"date": t, "ticker": tk, "phibar63": phib[i], "a": aret[i], "quartile": q[i], "placebo": is_plc[i]})
        pe = np.searchsorted(U, esup)
        factor = None if factor is None else (factor[0][pe], factor[1])
        G_age = er.holdings_covs(by_age, esup, Sig_U[np.ix_(pe, pe)], priced_U[pe], factor, beta_c)
        # the return-based factor benchmarks: statistical factors of the same support covariance and the named proxy
        # set (fb.PROXIES), loadings from each clone's own record inside the window, factor covariance from the window
        Xw = RAW[s0:ti + 1][:, esup]; Aw = A[s0:ti + 1][:, esup]
        if f_all is not None:
            Xw = np.where(Aw, np.nan_to_num(Xw.astype(np.float32)) - np.outer(f_all[s0:ti + 1], factor[0]).astype(np.float32), np.float32(0.0))
            rw_c, _ = ra.neutral_residuals(rv_raw[s0:ti + 1], have[s0:ti + 1], f_all[s0:ti + 1], wts)
        else:
            rw_c = ra.masked_zero(rv_raw[s0:ti + 1], have[s0:ti + 1])
        rhoF = fb.factor_correlations(Xw, Aw, Sig_U[np.ix_(pe, pe)], priced_U[pe], rw_c, have[s0:ti + 1], wts)
        Xp = RAW[s0:ti + 1][:, PROXY_IX]; Ap = A[s0:ti + 1][:, PROXY_IX]
        if f_all is not None:
            Xp, _ = ra.neutral_residuals(Xp, Ap, f_all[s0:ti + 1], wts)
        rhoP = fb.proxy_correlations(Xp, Ap, rw_c, have[s0:ti + 1], wts)
        del Xw, Aw, rw_c, Xp, Ap
        del Sig_U
        chunk = []; diag = []
        e_fresh = {}
        n_undef = 0
        DD = np.outer(D, D)
        for age in AGES + [-FUTURE]:
            G, okb = G_age[age]
            n_undef += int((by_age[age][1] & ~okb).sum()) # book under the priced share, or non-PSD
            if okb.sum() < 2:
                continue
            Gc = corr_of(G)
            om = weight_matrix(aret, h, max(age, 0)) # ex-ante pair weight, age-shifted
            Bc = om * Gc + (1 - om) * Sc
            ok = np.outer(okb, okb)[iu] & np.isfinite(Fo[iu]) & np.isfinite(So[iu])
            nc = np.sqrt((DD ** 2 + So ** 2) / h)[iu]; nr = ((1 - Sc ** 2) / np.sqrt(h))[iu]
            y = Fo[iu]; yc = Fc[iu]
            f32 = lambda v: v[ok].astype(np.float32)
            rec = {"date_ix": np.full(ok.sum(), ti, np.int32), "i": iu[0][ok].astype(np.int16), "j": iu[1][ok].astype(np.int16),
                   "age": np.full(ok.sum(), age, np.int16),
                   "eS": f32((So[iu] - y) ** 2), "eH": f32((G[iu] - y) ** 2), "eC": f32(((DD * Gc)[iu] - y) ** 2),
                   "eB": f32(((DD * Bc)[iu] - y) ** 2),
                   "cS": f32((Sc[iu] - yc) ** 2), "cH": f32((Gc[iu] - yc) ** 2), "cB": f32((Bc[iu] - yc) ** 2),
                   "cF5": f32((rhoF[5][iu] - yc) ** 2), "cF10": f32((rhoF[10][iu] - yc) ** 2),
                   "cP13": f32((rhoP[13][iu] - yc) ** 2), "cP22": f32((rhoP[22][iu] - yc) ** 2),
                   "nc2": f32(nc ** 2), "nr2": f32(nr ** 2), "omega": f32(om[iu]),
                   "phimin": f32(np.minimum.outer(phib, phib)[iu]),
                   "qmin": np.where(np.outer(is_plc, is_plc)[iu][ok], -1, np.minimum.outer(q, q)[iu][ok]).astype(np.int8),
                   "keyfund": np.where(phib[iu[0]] <= phib[iu[1]], iu[0], iu[1])[ok].astype(np.int16)}
            rec["mixed_placebo"] = (np.outer(is_plc, ~is_plc) | np.outer(~is_plc, is_plc))[iu][ok]
            rec["incomplete"] = (np.outer(is_inc, np.ones(n, bool)) | np.outer(np.ones(n, bool), is_inc))[iu][ok] # the pair touches an incomplete book
            if age == 0:
                e_fresh = {"eH0": pd.Series(rec["eH"], index=pd.MultiIndex.from_arrays([rec["i"], rec["j"]])),
                           "eB0": pd.Series(rec["eB"], index=pd.MultiIndex.from_arrays([rec["i"], rec["j"]])),
                           "cH0": pd.Series(rec["cH"], index=pd.MultiIndex.from_arrays([rec["i"], rec["j"]]))}
            chunk.append(pd.DataFrame(rec))
            # the diagonal: the sizing corollary on the same books (variance forecasts of one fund)
            dg = okb & np.isfinite(np.diag(Fo)) & np.isfinite(np.diag(So))
            diag.append(pd.DataFrame({"date_ix": ti, "i": np.where(dg)[0].astype(np.int16), "age": np.int16(age),
                                      "vS": np.diag(So)[dg].astype(np.float32), "vH": np.diag(G)[dg].astype(np.float32),
                                      "vY": np.diag(Fo)[dg].astype(np.float32),
                                      "nv2": (2 * np.diag(So) ** 2 / h)[dg].astype(np.float32), "phibar": phib[dg], "quartile": q[dg].astype(np.int8),
                                      "placebo": is_plc[dg], "incomplete": is_inc[dg]}))
        if chunk:
            C = pd.concat(chunk, ignore_index=True)
            key = pd.MultiIndex.from_arrays([C.i, C.j])
            for c0 in ("eH0", "eB0", "cH0"):
                C[c0] = e_fresh[c0].reindex(key).values.astype(np.float32) if c0 in e_fresh else np.nan
            C.to_parquet(f"{out_dir}/chunks/{t:%Y%m%d}.parquet", index=False)
            pd.concat(diag, ignore_index=True).to_parquet(f"{out_dir}/chunks/{t:%Y%m%d}_diag.parquet", index=False)
        el = time.time() - t0
        print(f" [{k}/{len(eval_t)}] {t.date()} funds {int(active.sum())} (Q1..Q4 {[int((q == m).sum()) for m in (1, 2, 3, 4)]}) "
              f"support {len(esup)}/{len(U)} undefined books {n_undef} "
              f"{el/60:.1f} min, ETA {el/k*(len(eval_t)-k)/60:.1f} min", flush=True)
    print(f"[{geom}] no active-window gate: {int(ever_active.sum())} of {n} funds have a clone in at least one date's window "
          f"(the two-year admission rule is the history filter)", flush=True)
    pd.DataFrame(pers_rows).to_csv(f"{out_dir}/persistence_trailing.csv", index=False, float_format="%.4f")


# ---------------------------------------------------------------- analysis
def chunks(out_dir, diag=False):
    files = sorted(glob.glob(f"{out_dir}/chunks/*{'_diag' if diag else ''}.parquet"))
    files = [f for f in files if f.endswith("_diag.parquet") == diag]
    for f in files:
        yield pd.read_parquet(f)


def twoway_ols(out_dir, y_of, design_of, k, seed_cols=("keyfund", "date_ix")):
    """OLS of y on the k design columns with standard errors two-way clustered (Cameron-Gelbach-Miller) by the
    pair's less persistent fund and by date, in two passes over the chunks (each chunk is one date)."""
    XtX = np.zeros((k, k)); Xty = np.zeros(k); nobs = 0
    for C in chunks(out_dir):
        Xd, y = design_of(C), y_of(C); ok = np.isfinite(y) & (Xd.sum(axis=1) > 0)
        Xd, y = Xd[ok], y[ok]
        XtX += Xd.T @ Xd; Xty += Xd.T @ y; nobs += len(y)
    used = np.diag(XtX) > 0
    beta = np.zeros(k); beta[used] = np.linalg.solve(XtX[np.ix_(used, used)], Xty[used])
    M_date = np.zeros((k, k)); M_int = np.zeros((k, k)); S_fund = pd.DataFrame(columns=range(k), dtype=float)
    for C in chunks(out_dir):
        Xd, y = design_of(C), y_of(C); ok = np.isfinite(y) & (Xd.sum(axis=1) > 0)
        Xd, y = Xd[ok], y[ok]; e = y - Xd @ beta
        Xe = Xd * e[:, None]
        sd = Xe.sum(axis=0); M_date += np.outer(sd, sd)
        by_fund = pd.DataFrame(Xe).groupby(C.keyfund.values[ok]).sum() # score of each fund cluster on this date
        M_int += by_fund.values.T @ by_fund.values
        S_fund = S_fund.add(by_fund, fill_value=0.0)
    M_fund = S_fund.values.T @ S_fund.values
    Vinv = np.zeros((k, k)); Vinv[np.ix_(used, used)] = np.linalg.inv(XtX[np.ix_(used, used)])
    V = Vinv @ (M_fund + M_date - M_int) @ Vinv
    return beta, V, nobs, used


def design_age_quartile(ages):
    cols = [(a, m) for a in ages for m in (1, 2, 3, 4)]
    col_ix = {c: i for i, c in enumerate(cols)}
    def design(C):
        Xd = np.zeros((len(C), len(cols)))
        for c, i in col_ix.items():
            Xd[:, i] = ((C.age.values == c[0]) & (C.qmin.values == c[1])).astype(float)
        return Xd
    return cols, design


EST_PAIRS = [("eH", "eS"), ("eC", "eS"), ("eB", "eS"), ("cH", "cS"), ("cB", "cS"), ("cF5", "cS"), ("cF10", "cS"),
             ("cP13", "cS"), ("cP22", "cS")]
PROXY_IX = None                    # returns-matrix columns of fb.PROXIES, set in main()


def paired_sums(C, keys, pairs):
    """per group of `keys`, for each (estimator, reference): the sums of both errors over the rows where both are defined
    (columns est, ref_<est>) and the count of those rows (n_<est>). A plain groupby().sum() would skip a NaN on one side
    and still add the other side's error, biasing the ratio."""
    Z = C[list(keys)].copy()
    for est, ref in pairs:
        if est not in C.columns:                                   # chunks written before an estimator existed
            Z[est] = 0.0; Z[f"ref_{est}"] = 0.0; Z[f"n_{est}"] = 0; continue
        m = np.isfinite(C[est].values) & np.isfinite(C[ref].values)
        Z[est] = np.where(m, C[est].values, 0.0); Z[f"ref_{est}"] = np.where(m, C[ref].values, 0.0); Z[f"n_{est}"] = m.astype(np.int64)
    return Z.groupby(list(keys)).sum()


def analyze(out_dir, geom, h, cls="all", seed=7):
    rng = np.random.default_rng(seed)
    role = ("pooled stratified comparison, all classes, global persistence strata" if cls == "pooled" else
            "long-only class alone (near-frozen; holdings should tie the stream)" if cls == "longonly" else
            "single class (effect identified off rotation)")
    # ---- 1. pooled RMSE ratios by quartile x age, with fund-clustered bootstrap intervals on the log ratio; every
    # estimator is compared with the stream on the rows where both are defined (paired_sums)
    acc = None
    for C in chunks(out_dir):
        per = paired_sums(C, ["age", "qmin", "i", "j"], EST_PAIRS)
        acc = per if acc is None else acc.add(per, fill_value=0.0)
    rows = []; fixed_rows = []
    est_name = {"eH": "holdings cov", "eC": "holdings corr, stream vol", "eB": "blend, pair weight",
                "cH": "holdings corr", "cB": "blend corr", "cF5": "factor corr K5", "cF10": "factor corr K10",
                "cP13": "proxy factor corr, core 13", "cP22": "proxy factor corr, extended 22"}

    def emit(per_all, age, stratum, out): # fund-clustered bootstrap ratio-to-stream for one (age, stratum) cell
        per_all = per_all[per_all[[f"n_{e}" for e, _ in EST_PAIRS]].max(axis=1) >= 12]
        if per_all.empty:
            return
        f1 = per_all.index.get_level_values("i").values.astype(int); f2 = per_all.index.get_level_values("j").values.astype(int)
        funds = np.unique(np.concatenate([f1, f2])); nf = int(funds.max()) + 1
        draws = [np.bincount(rng.choice(funds, len(funds), replace=True), minlength=nf) for _ in range(N_BOOT)]
        for est, ref in EST_PAIRS:
            per = per_all[per_all[f"n_{est}"] >= 12]
            if per.empty:
                continue
            fe1 = per.index.get_level_values("i").values.astype(int); fe2 = per.index.get_level_values("j").values.astype(int)
            se, sr = per[est].values, per[f"ref_{est}"].values
            ratio = np.sqrt(se.sum() / sr.sum())
            boots = []
            for cnt in draws:
                w = (cnt[fe1] * cnt[fe2]).astype(float)
                if (w * sr).sum() > 0:
                    boots.append(0.5 * np.log((w * se).sum() / (w * sr).sum()))
            lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
            out.append({"geometry": geom, "channel": "covariance" if est.startswith("e") else "correlation",
                        "estimator": est_name[est], "age": "future" if age < 0 else age, "stratum": stratum,
                        "pairs": len(per), "pair_dates": int(per[f"n_{est}"].sum()), "rmse_ratio_to_stream": ratio,
                        "ci_lo": float(np.exp(lo)), "ci_hi": float(np.exp(hi)), "share_pairs_below_1": float((se < sr).mean())})
    for (age, qm), per_all in acc.groupby(level=["age", "qmin"]): # sample quartiles
        emit(per_all, age, {-1: "placebo (index funds)", 0: "unstratified"}.get(qm, f"Q{qm}"), rows)
    for age, per_all in acc.groupby(level=["age", "i", "j"]).sum().groupby(level="age"):
        emit(per_all, age, "all", rows)
    R = pd.DataFrame(rows); R.to_csv(f"{out_dir}/rmse_by_quartile_age.csv", index=False, float_format="%.6f") # 6 dp so the article's 3-dp cells round unambiguously
    # ---- 1a. the same comparison on the article's fixed persistence bands: cuts on the pair's less-persistent fund
    # (phimin) that do not move with the sample (a fund at 0.6 is always <0.75). Reported alongside the quartiles.
    FIXED = [(0.0, 0.75, "<0.75"), (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.001, ">0.95")]
    accf = None
    for C in chunks(out_dir):
        ph = C.phimin.values; band = np.full(len(C), "", dtype=object)
        for lo, hi, lab in FIXED:
            band[(ph >= lo) & (ph < hi)] = lab
        C2 = C.assign(band=band); C2 = C2[C2.band != ""]
        per = paired_sums(C2, ["age", "band", "i", "j"], EST_PAIRS)
        accf = per if accf is None else accf.add(per, fill_value=0.0)
    if accf is not None:
        for (age, bd), per_all in accf.groupby(level=["age", "band"]):
            emit(per_all, age, bd, fixed_rows)
        for age, per_all in accf.groupby(level=["age", "i", "j"]).sum().groupby(level="age"):
            emit(per_all, age, "all", fixed_rows)
    RF = pd.DataFrame(fixed_rows); RF.to_csv(f"{out_dir}/rmse_by_fixedband_age.csv", index=False, float_format="%.6f")
    # ---- 1b. pairs touching an incomplete book against pairs of complete books, holdings cov vs stream by age. Incomplete
    # books are expected to gain less from a fresh book (part of the exposure is never observed); a missing sleeve is the
    # same at every age.
    comp_rows = []
    acc2 = None
    for C in chunks(out_dir):
        if "incomplete" not in C.columns:
            acc2 = None; break
        per = paired_sums(C, ["age", "incomplete", "i", "j"], [("eH", "eS"), ("cH", "cS")])
        acc2 = per if acc2 is None else acc2.add(per, fill_value=0.0)
    if acc2 is not None and acc2.index.get_level_values("incomplete").nunique() == 2:
        for (age, inc), per_all in acc2.groupby(level=["age", "incomplete"]):
            for est, ref in (("eH", "eS"), ("cH", "cS")):
                per = per_all[per_all[f"n_{est}"] >= 12]
                if per.empty:
                    continue
                comp_rows.append({"geometry": geom, "channel": "covariance" if est == "eH" else "correlation", "age": "future" if age < 0 else age,
                                  "books": "incomplete (hidden sleeve)" if inc else "complete", "pairs": len(per), "pair_dates": int(per[f"n_{est}"].sum()),
                                  "rmse_ratio_to_stream": float(np.sqrt(per[est].sum() / per[f"ref_{est}"].sum())),
                                  "share_pairs_below_1": float((per[est] < per[f"ref_{est}"]).mean())})
        pd.DataFrame(comp_rows).to_csv(f"{out_dir}/rmse_by_completeness_age.csv", index=False, float_format="%.4f")
    # ---- 2. within-pair-date panel regressions, two-way clustered. Covariance loss is normalized by its target sampling
    # variance nc2 (variance levels span orders of magnitude, so the scale must be controlled). The correlation target is
    # bounded in [-1, 1], so its loss is the raw squared-error difference: the analogous normalization by nr2 =
    # ((1 - rho^2)/sqrt(h))^2 degenerates for near-perfectly correlated pairs (the (1 - rho^2) noise formula breaks at
    # the boundary) and is not used.
    reg_rows = []
    specs = [("holdings cov vs stream", "covariance", lambda C: (C.eH - C.eS) / C.nc2, AGES),
             ("blend vs stream", "covariance", lambda C: (C.eB - C.eS) / C.nc2, AGES),
             ("holdings corr vs stream", "correlation", lambda C: (C.cH - C.cS), AGES),
             ("blend corr vs stream", "correlation", lambda C: (C.cB - C.cS), AGES),
             ("factor corr K5 vs stream", "correlation", lambda C: (C.cF5 - C.cS) if "cF5" in C.columns else np.full(len(C), np.nan), AGES),
             ("factor corr K10 vs stream", "correlation", lambda C: (C.cF10 - C.cS) if "cF10" in C.columns else np.full(len(C), np.nan), AGES),
             ("proxy factor corr core 13 vs stream", "correlation", lambda C: (C.cP13 - C.cS) if "cP13" in C.columns else np.full(len(C), np.nan), AGES),
             ("proxy factor corr extended 22 vs stream", "correlation", lambda C: (C.cP22 - C.cS) if "cP22" in C.columns else np.full(len(C), np.nan), AGES),
             ("staleness cost, holdings cov vs fresh", "covariance", lambda C: (C.eH - C.eH0) / C.nc2, AGES[1:]),
             ("staleness cost, holdings corr vs fresh", "correlation", lambda C: (C.cH - C.cH0), AGES[1:]),
             ("staleness cost, blend vs fresh", "covariance", lambda C: (C.eB - C.eB0) / C.nc2, AGES[1:]),
             ("future book vs fresh (positive control)", "covariance", lambda C: (C.eH - C.eH0) / C.nc2, [-FUTURE])]
    for name, channel, y_of, ages in specs:
        cols, design = design_age_quartile(ages)
        yq = lambda C, f=y_of: np.where(C.qmin.values >= 1, f(C).values, np.nan)
        beta, V, nobs, used = twoway_ols(out_dir, yq, design, len(cols))
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        for (a, m), b, s, u in zip(cols, beta, se, used):
            if u:
                reg_rows.append({"geometry": geom, "regression": name, "channel": channel, "age": "future" if a < 0 else a,
                                 "stratum": f"Q{m}", "coef": b, "se": s, "t": b / s if s > 0 else np.nan, "nobs": nobs})
        for a in ages:
            c = np.zeros(len(cols)); c[cols.index((a, 1))] = 1; c[cols.index((a, 4))] = -1
            if used[cols.index((a, 1))] and used[cols.index((a, 4))]:
                d = c @ beta; s = float(np.sqrt(max(c @ V @ c, 0)))
                reg_rows.append({"geometry": geom, "regression": name, "channel": channel, "age": "future" if a < 0 else a,
                                 "stratum": "Q1 - Q4", "coef": d, "se": s, "t": d / s if s > 0 else np.nan, "nobs": nobs})
    G = pd.DataFrame(reg_rows); G.to_csv(f"{out_dir}/panel_regressions.csv", index=False, float_format="%.6f")
    # ---- 3. crossover: the persistence at which the stale book's error ratio to the stream crosses one
    cross_rows = []
    acc = {}
    for C in chunks(out_dir):
        D = C[C.qmin >= 1]
        dec = np.clip(np.floor(D.phimin.values * 20), 0, 19).astype(int) # bins of 0.05 in persistence
        for age in AGES:
            m = D.age.values == age
            for est in ("eH", "eB"):
                mb = m & np.isfinite(D[est].values) & np.isfinite(D.eS.values) # both sides defined
                s = pd.DataFrame({"b": dec[mb], "se": D[est].values[mb], "sr": D.eS.values[mb], "n": 1}).groupby("b").sum()
                key = (age, est); acc[key] = s if key not in acc else acc[key].add(s, fill_value=0.0)
    for (age, est), s in sorted(acc.items()):
        s = s[s.n >= 200].sort_index()
        if s.empty:
            continue
        x = (s.index.values + 0.5) / 20; r = 0.5 * np.log(s.se / s.sr).values
        cross = np.nan
        for u in range(len(x) - 1):
            if r[u] > 0 >= r[u + 1] or r[u] < 0 <= r[u + 1]:
                cross = x[u] + (0 - r[u]) * (x[u + 1] - x[u]) / (r[u + 1] - r[u]); break
        cross_rows.append({"geometry": geom, "estimator": "holdings cov" if est == "eH" else "blend", "age": age,
                           "persistence_bins": " ".join(f"{a:.2f}:{np.exp(b):.3f}" for a, b in zip(x, r)),
                           "crossover_persistence": cross})
    pd.DataFrame(cross_rows).to_csv(f"{out_dir}/crossover.csv", index=False, float_format="%.4f")
    # ---- 4. sizing corollary on the diagonal, by quartile x age
    dsum = {}
    for Dg in chunks(out_dir, diag=True):
        Dg = Dg[np.isfinite(Dg.vH) & np.isfinite(Dg.vS) & np.isfinite(Dg.vY)] # both forecasts and the target defined
        for (age, qm), g in Dg.groupby(["age", "quartile"]):
            s = pd.Series({"n": len(g), "eH": ((g.vH - g.vY) ** 2).sum(), "eS": ((g.vS - g.vY) ** 2).sum(),
                           "dH": (((g.vH - g.vY) ** 2 - (g.vS - g.vY) ** 2) / g.nv2).mean() * len(g)})
            key = (age, qm); dsum[key] = s if key not in dsum else dsum[key] + s
    srow = [{"geometry": geom, "age": "future" if a < 0 else a, "stratum": {0: "unstratified/placebo"}.get(m, f"Q{m}"),
             "fund_dates": int(s.n), "rmse_ratio_holdings_vol_to_stream": float(np.sqrt(s.eH / s.eS)),
             "mean_loss_diff_over_noise": float(s.dH / s.n)} for (a, m), s in sorted(dsum.items())]
    pd.DataFrame(srow).to_csv(f"{out_dir}/sizing_by_quartile_age.csv", index=False, float_format="%.4f")
    # ---- report
    pd.set_option("display.width", 250)
    print(f"\n########## {cls.upper()} | {geom} geometry, horizon {h} -- {role} ##########")
    if R.empty or "channel" not in R.columns: # too few funds/common history -> no surviving pairs (e.g. the small long/short class)
        print(" no pairs survive in this sample (too few funds or too little common history) -- nothing to report")
        return R, G
    def show(DF, title, order):
        print(f"\n-- {title} (fund-clustered 95% CI in the CSV) --")
        for ch, est, nm in (("covariance", "holdings cov", "holdings covariance"),
                             ("covariance", "blend, pair weight", "blend at the pair weight"),
                             ("correlation", "holdings corr", "correlation channel, holdings correlation")):
            P = DF[(DF.channel == ch) & (DF.estimator == est)].pivot(index="stratum", columns="age", values="rmse_ratio_to_stream")
            P = P.reindex([s for s in order if s in P.index])
            print(f"{nm}:\n" + P.round(3).to_string())
    print("RMSE ratio to the stream by stratum and filing age (ratio < 1 = the estimator beats the stream):")
    show(RF, "fixed article bands [rmse_by_fixedband_age.csv]", ["<0.75", "0.75-0.85", "0.85-0.95", ">0.95", "all"])
    show(R, "sample quartiles [rmse_by_quartile_age.csv]", ["Q1", "Q2", "Q3", "Q4", "all"])
    print("\n===== within-pair-date regressions (loss difference in units of target noise; SE two-way clustered by fund and date)")
    print(G[G.stratum.isin(["Q1", "Q4", "Q1 - Q4"])].round(3).to_string(index=False))
    print("\n===== crossover persistence (ratio to the stream = 1)")
    if cross_rows:
        print(pd.DataFrame(cross_rows)[["estimator", "age", "crossover_persistence"]].round(3).to_string(index=False))
    print("\n===== sizing corollary (variance forecast of one fund, holdings vs stream)")
    print(pd.DataFrame(srow).round(3).to_string(index=False))
    return R, G


def build_pooled_chunks(dirs, pooled_dir):
    """Combine the per-class pair chunks into one pooled sample: two forecasters compared within each persistence
    stratum, all funds together. Fund ids are offset per class (within-class pairs never collide across classes) and
    qmin is re-cut to global quartiles of the pooled phimin distribution (1 = most rotating, 4 = most frozen; long-only
    funds land in the frozen strata). analyze() then runs on this directory unchanged, so the clustered inference is
    the same code. Pairs stay within their class (a cross-class pair carries almost no covariance signal and would
    dilute the comparison). Returns pooled_dir, or None when there is nothing to pool."""
    import shutil
    cdir = f"{pooled_dir}/chunks"
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
    vals = [C.phimin.values[np.isfinite(C.phimin.values)] for od in dirs for C in chunks(od)]
    vals = [v for v in vals if len(v)]
    if not vals:
        return None
    brk = np.quantile(np.concatenate(vals), [0.25, 0.5, 0.75]) # global phibar quartile breaks of the pooled sample
    gq = lambda phi: np.where(np.isfinite(phi),
                              np.clip(np.searchsorted(brk, np.nan_to_num(phi, nan=-1.0), side="right") + 1, 1, 4), 0).astype(np.int8)
    os.makedirs(cdir, exist_ok=True)
    base = 0
    for di, od in enumerate(dirs):
        n_dir = 0
        for f in sorted(glob.glob(f"{od}/chunks/*.parquet")):
            C = pd.read_parquet(f)
            if f.endswith("_diag.parquet"): # the single-fund diagonal frames
                C["i"] = C.i.astype(np.int64) + base
                C["quartile"] = gq(C.phibar.values)
                if len(C):
                    n_dir = max(n_dir, int(C.i.max() - base) + 1)
            else:
                for c in ("i", "j", "keyfund"):
                    C[c] = C[c].astype(np.int64) + base
                C["qmin"] = gq(C.phimin.values)
                if len(C):
                    n_dir = max(n_dir, int(max(C.i.max(), C.j.max(), C.keyfund.max()) - base) + 1)
            C.to_parquet(f"{cdir}/{di:02d}_{os.path.basename(f)}", index=False)
        base += n_dir
    print(f" pooled {len(dirs)} classes into {pooled_dir}; global phibar quartile breaks "
          f"{[round(float(b), 3) for b in brk]} (Q1 = most rotating)", flush=True)
    return pooled_dir


def fund_classes():
    """ticker -> strategy class from the census lists: 'mf' (managed futures), 'longshort', 'allocation', else 'longonly'."""
    cls = {}
    for path, c in [("research/etf_universe/census_download_list_allocation.csv", "allocation"),
                    ("research/etf_universe/census_download_list_futures.csv", "mf"),
                    ("research/etf_universe/census_download_list_longshort.csv", "longshort")]:
        if os.path.exists(path):
            for t in pd.read_csv(path).ticker.astype(str):
                cls[t] = c
    ov = "research/etf_universe/etf_class_overrides.csv" # brochure-based (Form ADV Part 2A) corrections, applied last:
    if os.path.exists(ov): # they override the census-list class. A class starting 'excluded_'
        o = pd.read_csv(ov, dtype=str) # (equity_options, fixed_income, ...) drops the fund from the sample.
        for t, c in zip(o.ticker.astype(str), o["class"].astype(str)):
            cls[t] = c
    return cls # everything else is 'longonly'


def index_placebo_of(sample, split):
    """the index funds on disk (status == index, not in_sample) of this split, an index-fund control set with zero rotation
    by construction; it belongs to the A/B split like every other fund, or split B would re-use split A's control pairs."""
    tick = sample[(sample.status == "index") & ~sample.in_sample.astype(bool)].ticker.astype(str)
    return set(t for t in tick if split == "all" or split_of(t) == split)


def main():
    # One pooled comparison of two forecasters across the strategy classes, stratified by the persistence of the
    # pair's less persistent fund. Each class is run separately (pairs are formed within a class only: a cross-class
    # pair carries almost no covariance signal) and the per-class chunks are pooled before the analysis
    # (build_pooled_chunks), so the strata are global and the clustered inference runs once on the pooled sample.
    global FUTURE, L, BACK
    split = arg("split", "A"); geoms = {"both": ["risk", "neutral"]}.get(arg("geom", "both"), [arg("geom", "both")])
    h = arg("h", H); FUTURE = max(1, h // 2)
    L = int(arg("L", L)); BACK = 6 * L # --L=63: stream and asset-covariance halflife (both sides, the matched pairing);
    # over a 63-day memory a phibar(63) > 0.95 book is frozen, so that band ties
    # by construction and any fresh edge left there is information, not staleness
    lo_cap = arg("longonly-cap", 0) # 0 = all funds (default); a positive cap subsamples the long-only class (seed 7)
    which = arg("class", "all"); classes = ["allocation", "mf", "longshort", "longonly"] if which == "all" else [which]
    complete_only = "--complete-only" in sys.argv
    out_root = arg("out", OUT) + ("/complete_only" if complete_only else "")
    tag = lambda cls, geom: f"{out_root}/{cls}_{geom}_{split}{'' if h == H else f'_h{h}'}{'' if L == 252 else f'_L{L}'}"
    sample = pd.read_csv(er.U); cls_of = fund_classes()
    # incomplete books (a subsidiary sleeve hidden from the daily book) are admitted and flagged as a stratum;
    # --complete-only drops them for the robustness run
    admit = sample.in_sample_staleness if "in_sample_staleness" in sample.columns else sample.in_sample
    active = [t for t in sample[admit].ticker.astype(str) if split == "all" or split_of(t) == split]
    flag = sample.get("book_incomplete", pd.Series(False, index=sample.index)).fillna(False).astype(bool)
    incomplete = set(sample[flag].ticker.astype(str)) & set(active)
    if complete_only:
        active = [t for t in active if t not in incomplete]
        print(f" complete books only: dropped {len(incomplete)} incomplete books: {' '.join(sorted(incomplete))}", flush=True)
        incomplete = set()
    elif incomplete:
        print(f" incomplete books (subsidiary sleeve hidden per the filing, or no filing; stratified): {len(incomplete)}: {' '.join(sorted(incomplete))}", flush=True)
    by_class = {c: [t for t in active if cls_of.get(t, "longonly") == c] for c in classes}
    if "longonly" in by_class and lo_cap and len(by_class["longonly"]) > lo_cap:
        by_class["longonly"] = sorted(np.random.default_rng(7).choice(by_class["longonly"], lo_cap, replace=False))
    index_placebo = index_placebo_of(sample, split) # index-fund controls; loaded with the long-only class, not paired in the pooled comparison

    if "--analyze-only" in sys.argv:
        for geom in geoms:
            dirs = [tag(c, geom) for c in classes if os.path.isdir(f"{tag(c, geom)}/chunks")]
            if build_pooled_chunks(dirs, tag("pooled", geom)):
                analyze(tag("pooled", geom), geom, h, cls="pooled")
        return

    need = set().union(*by_class.values())
    if "longonly" in classes:
        need |= index_placebo
    print(f"split {split}, horizon {h}: classes {classes}; funds by class "
          f"{ {c: len(v) for c, v in by_class.items()} }; loading {len(need)} books", flush=True)
    Rs = er.load_returns(); names = list(Rs.columns); name_ix = {c: i for i, c in enumerate(names)}
    global PROXY_IX
    PROXY_IX = fb.proxy_columns(names)                 # the named proxy factors must all be priced (fb.PROXIES)
    books = er.load_books(sorted(need), names, name_ix)
    clones = er.clone_returns(books, Rs)
    priced = set(clones.columns) # data-availability guard only (a computed clone series exists); the admission rule is applied upstream
    # ---- funnel: how many funds (and rotators, phibar_neutral_63 < 0.85) survive each gate, printed so a silent drop is visible
    dphi = daily_phibar(); rot = lambda ts: int(sum(dphi.get(str(t).upper(), 1.0) < 0.85 for t in ts))
    print("\n==================== fund funnel (funds | rotators phibar_neutral_63<0.85) ====================")
    print(f" admitted: {len(active)} | {rot(active)}")
    excl_c = [t for t in active if cls_of.get(t, 'longonly') not in classes]
    print(f" in a compared class {classes}: {sum(len(v) for v in by_class.values())} | {rot([t for v in by_class.values() for t in v])}"
          f" (dropped to excluded_* classes: {len(excl_c)} | {rot(excl_c)})")
    for c in classes:
        kp = sorted(set(by_class[c]) & priced)
        print(f" {c:10s} in-class {len(by_class[c]):4d} | {rot(by_class[c]):3d} -> has a clone {len(kp):4d} | {rot(kp):3d}"
              f" (no computable clone, dropped: {len(by_class[c]) - len(kp)} | {rot(set(by_class[c]) - priced)})")
    print(" remaining active/coverage gates print per date inside run_geometry; strata now use full-sample phibar (all funds).")
    print("================================================================================================\n", flush=True)
    stock = [j for j, c in enumerate(names) if not (c.startswith("FUT:") or c.startswith("CASH") or c.startswith("TSY:"))]
    f_mkt = np.nan_to_num(np.nanmean(Rs.values[:, stock], axis=1)) # equal-weight market of the priced stocks
    fid = f"{er.OUT}/fidelity.csv"; F = pd.read_csv(fid) if os.path.exists(fid) else None
    for cls in classes:
        keep = sorted(set(by_class[cls]) & priced) # pairs stay within-class (a cross-asset pair has no covariance signal); pooled at analysis
        if len(keep) < 4:
            print(f"\n[{cls}] only {len(keep)} priced funds, skipped"); continue
        print(f"\n===== class {cls}: {len(keep)} priced funds =====", flush=True)
        if F is not None:
            Ff = F[F.ticker.isin(keep)]
            print(f" fidelity (diagnostic): clone-vs-NAV corr median {Ff.clone_nav_corr.median():.3f} on {int(Ff.clone_nav_corr.notna().sum())} funds", flush=True)
        for geom in geoms:
            run_geometry(geom, keep, books, Rs, clones, f_mkt, set(), tag(cls, geom), h, incomplete=incomplete)
    for geom in geoms: # one pooled stratified comparison across all classes, global persistence strata
        dirs = [tag(c, geom) for c in classes if os.path.isdir(f"{tag(c, geom)}/chunks")]
        if build_pooled_chunks(dirs, tag("pooled", geom)):
            analyze(tag("pooled", geom), geom, h, cls="pooled")
    print("\nwrote", out_root)


if __name__ == "__main__":
    main()
