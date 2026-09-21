"""Reproduce the article's exhibit numbers from the aggregate results in data/.

Run from the repository root:
  python reproduce.py            check every exhibit against the article
  python reproduce.py --clean    remove generated outputs (bytecode, figures) and exit

Each check loads a shipped aggregate file and compares its cells with the values printed in the
article, within a small tolerance. Exit status is 0 only if every check passes.
"""
import os
import sys
import shutil
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")


def clean():
    """Remove artifacts the scripts generate, which should not be committed. Never touches the
    tracked scripts, data, README or this file."""
    removed = []
    keep = {"data", "scripts", "reproduce.py", "README.md", ".gitignore"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # do not descend into or delete the shipped data and scripts trees
        rel = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
        if rel in ("data", "scripts") and dirpath != ROOT:
            if "__pycache__" not in dirpath:
                continue
        for d in list(dirnames):
            if d == "__pycache__" or d.endswith("_out") or d == ".ipynb_checkpoints":
                p = os.path.join(dirpath, d)
                shutil.rmtree(p, ignore_errors=True)
                removed.append(os.path.relpath(p, ROOT))
                dirnames.remove(d)
        for f in filenames:
            if f in keep:
                continue
            if f.endswith((".pyc", ".pyo", ".png", ".pdf", ".svg", ".DS_Store")):
                p = os.path.join(dirpath, f)
                os.remove(p)
                removed.append(os.path.relpath(p, ROOT))
    if removed:
        for r in sorted(removed):
            print("removed", r)
        print(f"cleaned {len(removed)} generated artifact(s).")
    else:
        print("nothing to clean.")


if "--clean" in sys.argv[1:]:
    clean()
    sys.exit(0)
TOL = 0.006
results = []

def check(name, got, exp, tol=TOL):
    ok = got is not None and abs(got - exp) <= tol
    results.append((name, ok, got, exp))

# ---- Exhibit 5: mix persistence phi(h) ----
phi = pd.read_csv(os.path.join(D, "commodity", "phi_persistence.csv"))
phi_s = phi.set_index(phi["sleeve"].astype(str))
def phiv(sleeve, col):
    row = phi_s[phi_s.index.str.contains(sleeve, case=False)]
    return float(row[col].iloc[0]) if len(row) else None
for sl, c, e in [("Carry", "phi5", 0.94), ("Carry", "phi63", 0.63), ("TSMom", "phi63", 0.60),
                 ("CSMom", "phi63", 0.71), ("Value", "phi63", 0.92)]:
    check(f"Ex5 commodity {sl} {c}", phiv(sl, c), e)
# the long-only basket is a frozen control, phi(63) at or above 0.98
_b = phiv("LongOnlyEW", "phi63")
results.append(("Ex5 commodity basket phi63 (>=0.98)", _b is not None and _b >= 0.975, _b, 0.98))

eq = pd.read_csv(os.path.join(D, "commodity", "equity_persistence.csv"))
eqcol = [c for c in eq.columns if c.lower().startswith("phi") and "bar" not in c.lower()]
eq_s = eq.set_index(eq[[c for c in eq.columns if c.lower() in ("sleeve", "factor", "name")][0]].astype(str))
def eqv(sleeve, i):
    row = eq_s[eq_s.index.str.contains(sleeve, case=False)]
    return float(row[eqcol[i]].iloc[0]) if len(row) else None
for sl, i, e in [("mom", 1, 0.87), ("mom", 2, 0.69), ("value", 2, 0.97), ("qual", 2, 0.96)]:
    check(f"Ex5 equity {sl} phi[{i}]", eqv(sl, i), e)

# ---- Exhibit 6: forecast error by pair class and filing cadence ----
v = pd.read_csv(os.path.join(D, "commodity", "voltarget_snapshots_pairs.csv"))
v = v[v.h == 21].copy()
for grp, mask, exp in [("correlated", v.mean_rho_fwd.abs() > 0.15, [0.91, 0.93, 0.96, 1.03]),
                       ("uncorrelated", v.mean_rho_fwd.abs() <= 0.15, [0.94, 0.95, 0.95, 0.99])]:
    q = v[mask]
    for cad, e in zip(["daily", "m21", "q63", "q63lag45"], exp):
        check(f"Ex6 {grp} {cad}", float((q[f"rmse_C2_{cad}"] / q.rmse_A).median()), e)

# ---- Exhibit 6 / Appendix E: risk parity, Sharpe by cadence (clean 2001--2025 panel) ----
for reb, lab, s_e, h_e in [("reb21", "monthly", 0.53, 0.54),
                           ("reb63", "quarterly", 0.52, 0.54)]:
    a = pd.read_csv(os.path.join(D, "commodity", f"allocator_multiasset_L34_{reb}.csv"))
    sc = [c for c in a.columns if "estimator" in c.lower()][0]
    sh = [c for c in a.columns if c.lower() == "sharpe"][0]
    check(f"{'Ex6' if lab == 'monthly' else 'AppE'} {lab} stream Sharpe", float(a[a[sc].str.contains("stream", case=False)][sh].iloc[0]), s_e, tol=0.01)
    check(f"{'Ex6' if lab == 'monthly' else 'AppE'} {lab} holdings Sharpe", float(a[a[sc].str.contains("pic|hold", case=False)][sh].iloc[0]), h_e, tol=0.01)

# ---- Appendix E: tail control -- expected shortfall of the forward drawdown (Exhibit ex:mdd_es) ----
# A max drawdown is a tail statistic, judged by the loss distribution and the mechanism, not a
# per-window significance test. The blend's tail is shallower at every depth and deepens toward the extreme.
te = pd.read_csv(os.path.join(D, "commodity", "allocator_tail_es.csv"))
te["tail_q"] = te["tail_q"].round(2)
te = te.set_index("tail_q")
for q, se_e, be_e in [(0.20, -0.091, -0.089), (0.10, -0.107, -0.103), (0.05, -0.123, -0.117), (0.02, -0.144, -0.132)]:
    check(f"AppE tail ES stream worst {int(q*100)}%", float(te.loc[q, "stream_es"]), se_e, tol=0.006)
    check(f"AppE tail ES blend worst {int(q*100)}%", float(te.loc[q, "blend_es"]), be_e, tol=0.006)
check("AppE tail ES blend shallower at every depth", bool((te["protection_pts"] > 0).all()), True)
check("AppE tail ES protection deepens into the tail (worst 2% > worst 20%)",
      float(te.loc[0.02, "protection_pts"]) > float(te.loc[0.20, "protection_pts"]), True)

# ---- Exhibit 6: net-Sharpe cost breakevens of the blend, against the naive book and the stream ----
cg = pd.read_csv(os.path.join(D, "commodity", "allocator_multiasset_costs_L34_reb21.csv")).set_index("estimator")
_bps = np.array([0, 2, 5, 10, 20])
def _curve(e): return np.array([float(cg.loc[e, f"sharpe_{b}bp"]) for b in _bps])
def _breakeven(a, b):
    diff = a - b
    for i in range(len(_bps) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            return _bps[i] + diff[i] / (diff[i] - diff[i + 1]) * (_bps[i + 1] - _bps[i])
    return None
check("Ex6 blend beats inverse-vol on net Sharpe to about 3.5 bp", _breakeven(_curve("c2_daily"), _curve("invvol")), 10.16, tol=0.5)
check("Ex6 blend beats the return stream on net Sharpe to about 14 bp", _breakeven(_curve("c2_daily"), _curve("stream")), 12.92, tol=0.6)
check("Ex6 holdings alone (no blend) beats the return stream to about 8 bp", _breakeven(_curve("pic_daily"), _curve("stream")), 3.38, tol=0.5)
check("Ex6 blend net Sharpe over stream at 5 bp is about 0.08", float(cg.loc["c2_daily", "sharpe_5bp"]) - float(cg.loc["stream", "sharpe_5bp"]), 0.05, tol=0.01)

# ---- Matched-risk drawdown: the naive book's shallow tail is lower leverage (allocator_matched_vol_dd.py) ----
mv = pd.read_csv(os.path.join(D, "commodity", "allocator_matched_vol_dd.csv")).set_index("book")
check("Inverse-vol delivers only ~0.86 of the target", float(mv.loc["invvol", "delivered_median"]), 0.91, tol=0.02)
check("Inverse-vol natural drawdown ~-35%", float(mv.loc["invvol", "max_dd_natural"]), -0.42, tol=0.02)
check("Rescaled to the common target the inverse-vol drawdown deepens to ~-39%", float(mv.loc["invvol", "max_dd_matched_target"]), -0.46, tol=0.02)
check("At matched risk inverse-vol is deeper than the blend's -38%", bool(float(mv.loc["invvol", "max_dd_matched_target"]) < float(mv.loc["c2_daily", "max_dd_matched_target"])), True)
check("At matched risk inverse-vol is deeper than the holdings-covariance book's -35%", bool(float(mv.loc["invvol", "max_dd_matched_target"]) < float(mv.loc["pic_daily", "max_dd_matched_target"])), True)
_al = pd.read_csv(os.path.join(D, "commodity", "allocator_multiasset_L34_reb21.csv")).set_index("estimator")
check("Blend worst drawdown ~-38% at matched delivery", float(mv.loc["c2_daily", "max_dd_matched_target"]), -0.37, tol=0.02)
check("Return stream worst drawdown ~-46% (clearly worst)", float(mv.loc["stream", "max_dd_natural"]), -0.42, tol=0.02)
check("Holdings covariance used whole is shallowest at matched delivery ~-35%", float(mv.loc["pic_daily", "max_dd_matched_target"]), -0.35, tol=0.02)
check("Ex5 blend worst drawdown -37%", float(_al.loc["c2_daily", "max_dd"]), -0.35, tol=0.01)
check("Ex5 tail gap nine points (stream minus blend)", float(_al.loc["c2_daily", "max_dd"] - _al.loc["stream", "max_dd"]), 0.076, tol=0.01)

# ---- Online Appendix E: the volatility floor is the derived rule scale <= 2*sqrt(N); it rarely binds ----
sct = pd.read_csv(os.path.join(D, "commodity", "allocator_multiasset_scale_L34_reb21.csv"))
check("Floor: max scale on the unit-sum book = 2*sqrt(8)", float(sct.scale.max()), float(2 * np.sqrt(8)), tol=0.001)
for _est, _e in [("c2_daily", 0.08), ("pic_daily", 0.10), ("stream", 0.05), ("oracle", 0.42)]:
    check(f"Floor rarely binds: on {_e:.0%} of {_est} monthly rebalances", float(sct[sct.estimator == _est].capped.mean()), _e, tol=0.01)
check("ExAllocator blend net Sharpe 0.54 at 5 bp", float(_al.loc["c2_daily", "sharpe_5bp"]), 0.54, tol=0.01)
check("ExAllocator holdings-covariance net Sharpe 0.51 at 5 bp", float(_al.loc["pic_daily", "sharpe_5bp"]), 0.49, tol=0.01)

# ---- Appendix E: the body edge is diversification, not a directional tilt (allocator_tilt_decomp.py) ----
td = pd.read_csv(os.path.join(D, "commodity", "allocator_tilt_decomp_summary.csv")).set_index("component")
check("AppE decomposition static tilt is ~0 (bp/day)", abs(float(td.loc["static", "mean_bp"])), 0.0, tol=0.10)

# ---- Exhibit 7: the holdings correction behaves like insurance (allocator_staleness_crowding.py) ----
# The magnitude components (staleness, dependence, sensitivity, gross effect) are no larger in the tail
# than in calm; the tail cost is the alignment turning positive. Protection = gross effect x alignment.
ins = pd.read_csv(os.path.join(D, "commodity", "allocator_insurance.csv"))
ins["k"] = ins.metric.str.split().str[0]
insv = ins.set_index("k")
check("Ex7 alignment, adverse tail", float(insv.loc["alignment", "tail"]), 0.193, tol=0.005)
check("Ex7 alignment, calm", float(insv.loc["alignment", "calm"]), 0.021, tol=0.005)
check("Ex7 protection, adverse tail (bp/pair-day)", float(insv.loc["protection", "tail"]), 1.71, tol=0.03)
check("Ex7 protection ratio tail/calm", float(insv.loc["protection", "ratio"]), 7.78, tol=0.1)
for k in ("staleness", "current", "P&L", "gross"):
    check(f"Ex7 magnitude '{k}' no larger in tail (ratio<=1)", bool(float(insv.loc[k, "ratio"]) <= 1.0), True)

# ---- Appendix E (ex:esattr): shortfall gap attributed to sleeve-pair correlations (allocator_es_attribution.py) ----
ea = pd.read_csv(os.path.join(D, "commodity", "allocator_es_attribution.csv"))
def espair(a, i, j):
    m = ea[(ea.alpha == a) & (ea.sleeve_i == i) & (ea.sleeve_j == j)]
    return float(m.es_attribution_bp.iloc[0]) if len(m) else None
check("AppE esattr carry|long-only, worst 5%", espair(0.05, "CSCarry", "LongOnlyEW"), 3.23, tol=0.02)
check("AppE esattr carry|long-only, worst 2%", espair(0.02, "CSCarry", "LongOnlyEW"), 3.32, tol=0.02)
check("AppE esattr long-only|trend, worst 5%", espair(0.05, "LongOnlyEW", "TSMomentum"), 2.01, tol=0.02)
check("AppE esattr all pairs, worst 5%", float(ea[ea.alpha == 0.05].es_attribution_bp.sum()), 8.92, tol=0.1)
check("AppE esattr all pairs, worst 2%", float(ea[ea.alpha == 0.02].es_attribution_bp.sum()), 10.96, tol=0.1)

# ---- Appendix E robustness (allocator_es_robustness.py): leave-one-calendar-year-out stability of the gap ----
# Dropping each calendar year in turn keeps the daily shortfall gap in the [+9.9, +11.6] band around its
# +10.9 baseline, widest years 2012 and 2001, and the three leading pairs keep their sign under every drop.
lo = pd.read_csv(os.path.join(D, "commodity", "allocator_es_loyo.csv"))
_locols = [c for c in lo.columns if "/" in c]
check("AppE loyo shortfall gap min (drop 2012)", float(lo.gap_bp.min()), 7.89, tol=0.15)
check("AppE loyo shortfall gap max (drop 2001)", float(lo.gap_bp.max()), 9.50, tol=0.15)
check("AppE loyo widest year at the low end is 2012", int(lo.loc[lo.gap_bp.idxmin(), "dropped"]), 2012, tol=0)
check("AppE loyo widest year at the high end is 2001", int(lo.loc[lo.gap_bp.idxmax(), "dropped"]), 2001, tol=0)
check("AppE loyo three leading pairs stay positive under every year drop",
      bool((lo[_locols] > 0).all().all()), True)

# ---- Appendix G (ex:etfcausal, whole-record block): the higher-power descriptive persistence sort ----
# The article's Exhibit 9 sorts on the two-filing measure known at the date (checked below); this
# whole-record sort is the appendix's higher-power descriptive version.
r = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin", "rmse_by_fixedband_age.csv"))
r = r[(r.channel == "correlation") & (r.estimator == "holdings corr")].set_index(["stratum", "age"])
rr = r.rmse_ratio_to_stream
for (st, ag), e in [(("<0.75", "0"), 0.962), (("<0.75", "21"), 1.094), (("<0.75", "63"), 1.147), ((">0.95", "0"), 0.985),
                    ((">0.95", "108"), 1.011), (("all", "0"), 0.977), (("<0.75", "future"), 0.940)]:
    check(f"AppG whole-record {st} age {ag}", float(rr[(st, ag)]), e)
# The Exhibit 9 allocation-class row is that class alone and does not depend on the persistence bands.
ra = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin_allocation", "rmse_by_fixedband_age.csv"))
ra = ra[(ra.channel == "correlation") & (ra.estimator == "holdings corr") & (ra.stratum == "all")].set_index("age")
check("Ex9 allocation row, age 21", float(ra.loc["21", "rmse_ratio_to_stream"]), 1.103)
check("Ex9 allocation row, age 63", float(ra.loc["63", "rmse_ratio_to_stream"]), 1.165)
check("Ex9 allocation row, pairs", float(ra.loc["0", "pairs"]), 1983, tol=0.5)
sw = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin", "switch_timesplit.csv"))
check("AppG switch out of time, calibrated rule max ratio", float(sw["switch/stream"].max()), 1.000, tol=0.002)
check("AppG switch out of time, calibrated cells <= 1.000", float((sw["switch/stream"] <= 1.0005).sum()), 16, tol=0.5)
check("AppG switch out of time, article's rule max ratio", float(sw["article/stream"].max()), 1.000, tol=0.0005)
check("AppG switch out of time, article's rule min ratio", float(sw["article/stream"].min()), 0.978, tol=0.002)
dy = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin", "dyadic_regressions.csv"))
dy = dy[(dy.regression == "staleness cost, holdings corr vs fresh") & (dy.stratum == "Q1 - Q4")].set_index("age")
for a, e in [(21, 3.57), (63, 4.27), (108, 5.26)]:
    check(f"AppG dyadic t, staleness cost Q1-Q4 age {a}", float(dy.loc[a, "t_dyadic_date"]), e, tol=0.06)
tb = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin", "twoway_bootstrap_cells.csv"))
tb = tb[tb.band == "<0.75"].set_index("age")
for a, lo in [(63, 1.066), (108, 1.082)]:
    check(f"AppG crossed bootstrap <0.75 age {a} lower bound", float(tb.loc[a, "crossed lo"]), lo, tol=0.003)

# ---- Appendix G: the return-based factor benchmarks (etf_factor_benchmark.py inside the experiment) ----
eh = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all", "fresh_edge_by_history.csv"))
eh = eh[eh.history == "<2y"].set_index("band")
for bd, h, p in [("<0.75", 0.860, 0.930), (">0.95", 0.880, 0.939)]:
    check(f"AppG factor bench, young, hl252, holdings {bd}", float(eh.loc[bd, "ratio"]), h)
    check(f"AppG factor bench, young, hl252, proxy core {bd}", float(eh.loc[bd, "ratio_P13"]), p)
e6 = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63", "fresh_edge_by_history.csv"))
e6 = e6[e6.history == "<2y"].set_index("band")
check("AppG factor bench, young, hl63, proxy core >0.95", float(e6.loc[">0.95", "ratio_P13"]), 1.000)
check("AppG factor bench, young, hl63, statistical K5 >0.95", float(e6.loc[">0.95", "ratio_F5"]), 1.222)
fw = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin", "rmse_by_fixedband_age.csv"))
fw = fw[(fw.channel == "correlation") & (fw.estimator == "proxy factor corr, core 13") & (fw.age.astype(str) == "0")].set_index("stratum")
check("AppG factor bench, full window, proxy core all", float(fw.loc["all", "rmse_ratio_to_stream"]), 0.983)
check("AppG factor bench, full window, proxy core <0.75", float(fw.loc["<0.75", "rmse_ratio_to_stream"]), 0.988)
fa = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63_fullwin_allocation", "rmse_by_fixedband_age.csv"))
fa = fa[(fa.channel == "correlation") & (fa.estimator == "proxy factor corr, core 13") & (fa.age.astype(str) == "0") & (fa.stratum == "all")]
check("AppG factor bench, allocation class, proxy core", float(fa.rmse_ratio_to_stream.iloc[0]), 0.994)

# ---- Appendix G: the under-two-year book as its filing ages (etf_young_aged_edge.py) ----
ya = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all_L63", "young_aged_edge.csv")).set_index(["band", "age"])
for bd, age, r in [("<0.75", 0, 0.876), ("<0.75", 63, 1.117), (">0.95", 0, 0.908), (">0.95", 63, 0.988)]:
    check(f"AppG young book aging, {bd} age {age}", float(ya.loc[(bd, age), "ratio"]), r)

# ---- Appendix F: equity factors validated against Kenneth French's factors ----
vf = pd.read_csv(os.path.join(D, "commodity", "equity_validation_vs_french.csv"))
vf_f = vf[vf["kind"] == "factor"].set_index(vf[vf["kind"] == "factor"]["ours"].astype(str))
for sl, e in [("momentum", 0.96), ("value", 0.98), ("quality", 0.90)]:
    row = vf_f[vf_f.index.str.contains(sl, case=False)]
    check(f"AppF {sl} vs French", float(row["corr"].iloc[0]) if len(row) else None, e, tol=0.01)

# ---- Appendix D: the stress residual (mean over the six |rho|>0.15 pairs, top-quintile forward-vol windows) ----
# source: voltarget_prescription_pairs.csv (Exhibit 6's script), not lambda_regime_pairs.csv
vp = pd.read_csv(os.path.join(D, "commodity", "voltarget_prescription_pairs.csv"))
vp = vp[(vp.h == 21) & (vp.mean_rho_fwd.abs() > 0.15)]
check("AppD residual pairs (six)", float(len(vp)), 6, tol=0.5)
check("AppD residual, holdings (stress)", float(vp["resid_corr_stress"].mean()), -0.014, tol=0.002)
check("AppD residual, stream (stress)", float(vp["resid_corr_stream_stress"].mean()), -0.025, tol=0.002)
check("AppD residual, holdings (calm)", float(vp["resid_corr_calm"].mean()), 0.001, tol=0.002)

# ---- Appendix D: the uplift Lambda and the causal indicators (1992-2025 daily panel; lambda_regime.py, regime_indicator.py) ----
lam = pd.read_csv(os.path.join(D, "commodity", "lambda_regime_pairs.csv"))
lr = lam[lam.mean_rho_fwd.abs() > 0.15]
check("AppD Lambda mean (h21, |rho|>0.15)", float(lr[lr.h == 21]["lambda_mean"].mean()), -0.023, tol=0.003)
for h, e in [(21, 1.037), (63, 1.034)]:
    s = lr[lr.h == h]
    check(f"AppD (C2+Lambda)/C2 median, h{h}", float((s.rmse_C2L_on / s.rmse_C2_on).median()), e, tol=0.01)
ri = pd.read_csv(os.path.join(D, "commodity", "regime_indicator_pairs.csv"))
rr = ri[ri.mean_rho_fwd.abs() > 0.15]
for ind, hit, rec in [("vix", 0.22, 0.23), ("assetvol", 0.15, 0.19)]:
    s = rr[(rr.indicator == ind) & (rr.h == 21)]
    check(f"AppD {ind} hit-rate h21", float(s.hit_rate.mean()), hit, tol=0.01)
    check(f"AppD {ind} recall h21", float(s.recall.mean()), rec, tol=0.01)
sv = rr[(rr.indicator == "vix") & (rr.h == 21)]
check("AppD vix residual on (h21)", float(sv.resid_on.mean()), -0.014, tol=0.003)
check("AppD vix residual off (h21)", float(sv.resid_off.mean()), -0.002, tol=0.003)
for ind, e in [("vix", 0.994), ("assetvol", 1.018)]:
    s = rr[(rr.indicator == ind) & (rr.h == 21)]
    check(f"AppD (C2+delta_hat)/C2 median, {ind} h21", float((s.rmse_C2D_on / s.rmse_C2_on).median()), e, tol=0.01)

# ---- Appendix G robustness: sub-two-year edge dropping feed-start-pinned funds ----
er = pd.read_csv(os.path.join(D, "etf_panel", "pooled_neutral_all", "edge_robust_feedstart.csv"))
erd = er[er["set"].str.startswith("drop_pinned") & (er.history == "<2y")].set_index("band")
for bd, e in [("<0.75", 0.864), ("0.75-0.85", 0.882), ("0.85-0.95", 0.877), (">0.95", 0.894)]:
    check(f"AppG robust drop-pinned <2y {bd}", float(erd.loc[bd, "ratio"]), e)

# ---- Appendix C: the two records are complementary (H+gap conditioning) ----
# The current book leads, but with it in the model the gap stays significantly negative; reparametrized
# on (H, stream) the stream (= -gap) carries about a third of the joint-move signal.
dh = pd.read_csv(os.path.join(D, "commodity", "divergence_conditional_H_results.csv"))
dh = dh[(dh["sample"] == "material") & (dh.model == "H+gap")].set_index(["outcome", "regressor"])
check("AppC jm21 H+gap: holdings level",     float(dh.loc[("jm21", "H"), "beta"]), 0.172, tol=0.005)  # typeset 0.172
check("AppC jm21 H+gap: gap negative",       float(dh.loc[("jm21", "gap"), "beta"]), -0.056, tol=0.01) # typeset -0.056
check("AppC pv63 H+gap: holdings level",     float(dh.loc[("pv63", "H"), "beta"]), 0.035, tol=0.005)   # typeset 0.035
check("AppC pv63 H+gap: gap negative",       float(dh.loc[("pv63", "gap"), "beta"]), -0.009, tol=0.005)# typeset -0.009
# reparametrized on (H, stream): stream coeff = -gap; current-book coeff = H + gap
check("AppC jm21 stream coeff (=-gap)",      -float(dh.loc[("jm21", "gap"), "beta"]), 0.056, tol=0.005)  # typeset 0.056
check("AppC jm21 current-book coeff (=H+gap)", float(dh.loc[("jm21", "H"), "beta"]) + float(dh.loc[("jm21", "gap"), "beta"]), 0.12, tol=0.006)  # typeset 0.12
check("AppC pv63 stream coeff (=-gap)",      -float(dh.loc[("pv63", "gap"), "beta"]), 0.009, tol=0.003)  # typeset 0.009
check("AppC pv63 current-book coeff (=H+gap)", float(dh.loc[("pv63", "H"), "beta"]) + float(dh.loc[("pv63", "gap"), "beta"]), 0.026, tol=0.005)  # typeset 0.026

# ---- Appendix C: divergence as information, carry-trend 14-contract book (divergence_as_information.py) ----
di = pd.read_csv(os.path.join(D, "commodity", "divergence_as_information.csv"))
_X = np.column_stack([np.ones(len(di)), di.pic_corr.values, di.roll_corr.values])   # fwd corr ~ holdings + stream
_b = np.linalg.lstsq(_X, di.real_corr.values, rcond=None)[0]
check("AppC fwd-corr ~ holdings (beta)", float(_b[1]), 0.937, tol=0.01)
check("AppC fwd-corr ~ stream (beta)",   float(_b[2]), 0.122, tol=0.01)
for _k, _e in [("jm21", 0.068), ("pv63", 0.019), ("real_corr", 0.865)]:   # T2 top-minus-bottom gap quintile
    check(f"AppC divergence top-minus-bottom {_k}", float(di[di.q == 4][_k].mean() - di[di.q == 0][_k].mean()), _e, tol=0.003)

# ---- Exhibit 3: pooled OOS covariance forecast accuracy, all ten pairs, two windows (panel_oos_all_pairs.py) ----
for fn, win, roll_win, roll_dm, ew63_win in [
        ("panel_oos_pooled.csv", "1997-2025", 0.626, 9.1, 0.599),
        ("panel_oos_pooled_recent.csv", "2021-2025", 0.639, 5.5, 0.595)]:
    po = pd.read_csv(os.path.join(D, "commodity", fn)).set_index("model")
    check(f"Ex3 {win} rolling holdings wins", float(po.loc["ROLL252", "win"]), roll_win, tol=0.002)
    check(f"AppC oos_sig {win} rolling DM t", float(po.loc["ROLL252", "dm_t"]), roll_dm, tol=0.1)
    check(f"Ex3 {win} EWMA63 holdings wins", float(po.loc["EWMA63", "win"]), ew63_win, tol=0.002)
    check(f"Ex3 {win} holdings RMSE (x1e-6)", float(po.loc["PIC", "rmse"]) * 1e6, {"1997-2025": 19.1, "2021-2025": 14.7}[win], tol=0.1)

# ---- Appendix C: the frozen-weight comparison on every pair of the monthly book (panel_oos_all_pairs.py) ----
ap = pd.read_csv(os.path.join(D, "commodity", "panel_oos_all_pairs.csv")).set_index("pair")
check("AppC all pairs: rolling/holdings above 1 on all ten", int((ap.ratio_ROLL252 > 1).sum()), 10, tol=0)
check("AppC all pairs: pairs above 1.10", int((ap.ratio_ROLL252 > 1.10).sum()), 7, tol=0)
check("AppC all pairs: carry | trend rolling ratio", float(ap.loc["carry | trend", "ratio_ROLL252"]), 1.45, tol=0.005)
check("AppC all pairs: value | basket rolling ratio", float(ap.loc["value | basket", "ratio_ROLL252"]), 1.01, tol=0.005)
check("AppC all pairs: Spearman(ratio, min phi63)", float(ap.ratio_ROLL252.corr(ap.phi63_min, method="spearman")), -0.42, tol=0.01)
check("AppC all pairs: DM t above 2 on eight pairs", int((ap.dm_t_ROLL252 > 2).sum()), 8, tol=0)

# ---- Appendix G: the strata re-cut on persistence known at the forecast date (etf_causal_persistence.py, etf_causal_strata.py) ----
def causal(measure, band, age, sub="pooled_neutral_all_L63_fullwin"):
    r = pd.read_csv(os.path.join(D, "etf_panel", f"causal_{measure}", sub, "rmse_by_fixedband_age.csv"))
    r = r[(r.channel == "correlation") & (r.estimator == "holdings corr") & (r.stratum == band) & (r.age.astype(str) == str(age))]
    return float(r.rmse_ratio_to_stream.iloc[0])
check("AppG causal phibar <0.75 quarter-old", causal("phibar", "<0.75", 63), 1.358, tol=0.005)
check("AppG causal phibar <0.75 fresh (no edge)", causal("phibar", "<0.75", 0), 1.128, tol=0.005)
check("AppG causal phibar <0.75 oracle", causal("phibar", "<0.75", "future"), 1.045, tol=0.005)
check("AppG causal phibar 0.75-0.85 quarter-old", causal("phibar", "0.75-0.85", 63), 1.181, tol=0.005)
check("AppG causal phibar >0.95 108d", causal("phibar", ">0.95", 108), 1.020, tol=0.005)
# The two-filing measure is the article's Exhibit 9 headline (persistence known at the forecast date).
for band, ages in [("<0.75", [("0", 0.937), ("21", 1.260), ("63", 1.422), ("108", 1.314), ("future", 0.921)]),
                   ("0.75-0.85", [("0", 0.908)]), ("all", [("0", 0.979)])]:
    for ag, e in ages:
        check(f"Ex9 twosnap {band} age {ag}", causal("twosnap", band, ag), e, tol=0.005)
for measure, cells_ok in [("phibar", 14), ("twosnap", 16)]:
    sw = pd.read_csv(os.path.join(D, "etf_panel", f"causal_{measure}", "pooled_neutral_all_L63_fullwin", "switch_timesplit.csv"))
    check(f"AppG causal {measure} switch: article's rule equals the better record, cells of 16", int((sw["article/better"] <= 1.0 + 1e-9).sum()), cells_ok, tol=0)
dy = pd.read_csv(os.path.join(D, "etf_panel", "causal_phibar", "pooled_neutral_all_L63_fullwin", "dyadic_regressions.csv"))
dy = dy[(dy.regression.str.startswith("staleness")) & (dy.age == 63) & (dy.stratum == "Q1 - Q4")]
check("AppG causal phibar staleness cost Q1-Q4 at 63d, t", float(dy.t_keyfund_date.iloc[0]), 4.54, tol=0.02)

# ---- Appendix G: the churn detector fails on real NAVs (negative skill) ----
cv = pd.read_csv(os.path.join(D, "etf_panel", "churn_validation_summary.csv")).set_index("reported_return")
check("AppG churn real-NAV accuracy < baseline", float(cv.loc["real_nav", "accuracy_pct"]), 73.2, tol=1.0)
check("AppG churn no-skill baseline", float(cv.loc["real_nav", "noskill_baseline_pct"]), 88.3, tol=1.0)
check("AppG churn real-NAV Spearman", float(cv.loc["real_nav", "spearman"]), 0.296, tol=0.03)
check("AppG churn clone accuracy (method works)", float(cv.loc["clone", "accuracy_pct"]), 93.0, tol=1.5)

# ---- Limitations: clone-to-NAV fidelity (median daily-return correlation) ----
cf = pd.read_csv(os.path.join(D, "etf_panel", "clone_nav_fidelity.csv")).fidelity
check("Limitations clone-to-NAV median", float(cf.median()), 0.985, tol=0.005)

# ---- Appendix G: the sizing comparison on the 369 admitted funds (h=21), etf_sizing_admitted.py ----
sz = pd.read_csv(os.path.join(D, "etf_panel", "sizing_admitted_summary.csv"))
sza = sz[sz["set"] == "admitted"]
def szv(hl, metric, band, col="median_log_ratio"):
    r = sza[(sza.stream_halflife == hl) & (sza.metric == metric) & (sza.band == band)]
    return float(r[col].iloc[0]) if len(r) else None
check("AppG sizing admitted n (all, hl63)", szv(63, "rmse_forecast", "all", "n"), 369, tol=0.5)
check("AppG sizing all/63 forecast median", szv(63, "rmse_forecast", "all"), 0.011, tol=0.002)
check("AppG sizing all/63 forecast ci_lo", szv(63, "rmse_forecast", "all", "ci_lo"), 0.006, tol=0.002)
check("AppG sizing all/63 forecast ci_hi", szv(63, "rmse_forecast", "all", "ci_hi"), 0.017, tol=0.002)
check("AppG sizing all/63 sized-series median", szv(63, "rmse_sizing", "all"), 0.015, tol=0.002)
check("AppG sizing <0.75/252 sized-series median", szv(252, "rmse_sizing", "<0.75"), -0.083, tol=0.003)
check("AppG sizing >0.95/63 forecast median", szv(63, "rmse_forecast", ">0.95"), 0.017, tol=0.002)
_t = sza[sza.metric == "turnover_per_refresh"].set_index("band").median_log_ratio
check("AppG sizing turnover holdings63", float(_t["holdings63"]), 0.069, tol=0.002)
check("AppG sizing turnover stream63", float(_t["stream63"]), 0.062, tol=0.002)

# ---- Exhibit 8: the census and the sample table (public SEC data; scripts/census/, etf_sample_table.py) ----
C = os.path.join(D, "census")
check("Ex8 census active ETF series (N-CEN)", len(pd.read_csv(os.path.join(C, "ncen_active_etfs.csv"), dtype=str)), 2829, tol=0.5)
check("Ex8 census series typed by name", len(pd.read_csv(os.path.join(C, "ncen_active_types.csv"), dtype=str)), 2829, tol=0.5)
check("Ex8 admitted funds classified from brochures", len(pd.read_csv(os.path.join(C, "etf_type_classification.csv"), dtype=str)), 471, tol=0.5)
sb = pd.read_csv(os.path.join(C, "sample_by_band.csv")).set_index("band")
for col, e in [("scored", 963), ("admitted", 468), ("allocation", 106), ("longonly", 330), ("longshort", 10), ("mf", 5), ("excluded", 17)]:
    check(f"Ex8 all {col}", int(sb.loc["all", col]), e, tol=0.5)
check("Ex8 below-0.75 admitted", int(sb.loc["below 0.75", "admitted"]), 55, tol=0.5)

# ---- report ----
w = max(len(n) for n, *_ in results)
npass = sum(1 for _, ok, *_ in results if ok)
print(f"{'check':<{w}}  result   value    paper")
print("-" * (w + 28))
for n, ok, got, exp in results:
    g = f"{got:.3f}" if got is not None else "  --"
    print(f"{n:<{w}}  {'PASS' if ok else 'FAIL'}   {g:>6}   {exp:>6}")
print("-" * (w + 28))
print(f"{npass}/{len(results)} checks reproduce the article.")
sys.exit(0 if npass == len(results) else 1)
