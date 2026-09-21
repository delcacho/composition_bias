# The Composition Bias in Multi-Strategy Portfolios: reproducibility

Code and derived data for the article *The Composition Bias in Multi-Strategy Portfolios*
(Carlos del Cacho).

An allocator can estimate the correlation between two strategies from their return histories or
from their current holdings, and the two are stale in opposite directions. The article measures
the resulting bias on a laboratory of commodity-futures and equity-factor sleeves and on a panel
of 446 transparent active ETFs, and gives a rule for which record to trust as a function of a
manager's composition persistence.

This repository lets a reader **regenerate every exhibit's numbers from the derived aggregate
results included here**, and documents the full raw-to-results pipeline for anyone who licenses
the underlying data themselves.

## What is and is not included

The raw inputs are licensed and cannot be redistributed:

- **Futures contract prices** (Barchart) behind the commodity book;
- **Compustat North America** security and fundamental files (via WRDS) behind the equity sleeves
  and the pricing of the ETF books;
- **ETF Global** daily constituent files (via Massive) behind the active-ETF panel.

Because of this, the repository ships **intermediate aggregate outputs only**: persistence
curves, per-pair and per-band forecast-error ratios, regression coefficients, crossover points.
These are pooled statistics; no per-fund or per-contract priced series, and no raw feed, is
included. The scripts that ingest the raw feeds into intermediate form (the "raw to intermediate"
stage) are also not included, because they require the licensed data and the author's data-access
layer; they are listed under [The raw-to-intermediate stage](#the-raw-to-intermediate-stage) so
the provenance is complete.

Public inputs that anyone can obtain are named where they are used: the Fama-French factors from
Kenneth R. French's data library (validation of the equity sleeves), the VIX from FRED (the stress
subsection), and SEC filings (Form N-CEN, Form ADV Part 2A) for the active-ETF sample. The
census and classification layer built on those filings is public data end to end, so it **is**
shipped, code and outputs: see [The census and classification layer](#the-census-and-classification-layer).

## Layout

```
scripts/            analysis, statistics and figure code (46 files)
  census/           the active-ETF census and strategy classification from SEC filings, runnable
  equity_factors/   single-stock factor sleeve construction and validation
  paper_tests/      the simulation (test_bias.py) and its helpers
data/
  census/           the N-CEN census, the name typing, the brochure classification, the sample table
  commodity/        per-sleeve and per-pair aggregates for the commodity book
  etf_panel/        pooled active-ETF comparison outputs, by variant
```

## Environment

Python 3.11. `pip install numpy pandas scipy pyarrow matplotlib statsmodels pandas-datareader`. No
network access is needed to reproduce the exhibits from the shipped aggregates.

## Reproducing the exhibits

Two tiers.

**Tier A: from the shipped aggregates (fully supported here).** Every exhibit's reported numbers
are the cells of the files in `data/`, or, for the simulation, are produced from scratch. For
example, Exhibit 9's ratios are the columns of
`data/etf_panel/pooled_neutral_all_L63_fullwin/rmse_by_fixedband_age.csv`; the persistence table
(Exhibit 5) is `data/commodity/phi_persistence.csv`; the simulation (Exhibit 2) runs with
`python scripts/paper_tests/test_bias.py` and needs no data.

**Tier B: the full pipeline (needs the reader's own licensed data).** The analysis scripts in
`scripts/` produce the aggregates in `data/` from the intermediate priced panels, which are in turn
built from the raw feeds by the (not included) ingestion stage. A reader with the licensed data can
rebuild the intermediate panels with their own ingestion and then run the analysis scripts to
regenerate `data/` from end to end.

### Verifying

`python reproduce.py` loads each shipped aggregate and compares its cells with the values printed
in the article, within a small tolerance. It prints a pass or fail for every exhibit and exits
non-zero if any check misses; all one hundred and sixty-five checks reproduce the article.

### Exhibit provenance

"Aggregate" = the exhibit's numbers are in the named `data/` file. "Documented" = the exhibit is a
figure or table built in the raw-to-intermediate stage that is not shipped; its numbers are in the
article.

| Exhibit | Shows | Script(s) | Data / status |
|---|---|---|---|
| 1 | Energy episode, two signals converge | `commodity_crowding_analysis.py`, `crop_energy_figure.py` | Documented (raw contracts) |
| 2 | The composition bias over a simulated crowding episode | `paper_tests/test_bias.py` | Runs from scratch, no data |
| 3 | Out-of-sample covariance forecast accuracy, all ten pairs pooled, two windows | `panel_oos_all_pairs.py` | `data/commodity/panel_oos_pooled.csv` (1997-2025), `panel_oos_pooled_recent.csv` (2021-2025); the per-pair ratios in `panel_oos_all_pairs.csv` (Online Appendix C) |
| 4 | Divergence as information: forward joint-move size by quintile of the holdings-minus-stream gap (carry-trend); works pairwise, fades at the book level | `divergence_as_information.py`, `divergence_conditional_H.py`, `divergence_across_pairs.py`, `divergence_tail_book.py` | `data/commodity/divergence_as_information.csv`, `divergence_conditional_H_results.csv` (regression, all-pairs table and the risk-parity-book null are Online Appendix C) |
| 5 | Mix persistence by horizon and sleeve | `phi_persistence.py`, `equity_factors/equity_persistence_daily.py` | `data/commodity/phi_persistence.csv` (commodity), `equity_persistence.csv` (equity) |
| 6 | Risk parity across the eight-sleeve multi-asset book | `allocator_multiasset.py` | `data/commodity/allocator_multiasset_L34_reb{1,21,63}.csv` |
| 7 | The holdings correction behaves like insurance (alignment factorization) | `allocator_staleness_crowding.py` | `data/commodity/allocator_insurance.csv` |
| 8 | Sharpe net of transaction cost, blend, holdings-alone, stream and inverse-vol, with the breakevens | `allocator_cost_breakeven.py` | `data/commodity/allocator_multiasset_costs_L34_reb21.csv` |
| 9 | The daily-holdings ETF panel by persistence band and class | `etf_sample_table.py`, `scripts/census/` | `data/census/sample_by_band.csv` |
| 10 | Real managers, active ETFs, by persistence known at the forecast date | `etf_causal_persistence.py` (two-filing sort), `etf_staleness_experiment.py` (panel and allocation row) | `data/etf_panel/causal_twosnap/pooled_neutral_all_L63_fullwin/*.csv` (bands), `pooled_neutral_all_L63_fullwin_allocation/*.csv` (allocation row) |
| 11 | A calibrated rule for which record to use | (synthesis of Exhibits 6 and 10) | none |

Appendix exhibits draw on the same code and data: the stress
residual (`lambda_regime.py`, `regime_indicator.py`, with
`data/commodity/lambda_regime_pairs.csv`), the forward-drawdown expected shortfall of the two
books (`allocator_tail_es.py`, with `data/commodity/allocator_tail_es.csv`, Exhibit A9) and the
body-edge tilt decomposition (`allocator_tilt_decomp.py`, with `allocator_tilt_decomp_summary.csv`),
the pathwise Aumann-Shapley attribution of the shortfall gap to each sleeve-pair correlation
(`allocator_pnl_attribution.py`, `allocator_es_attribution.py`, with
`data/commodity/allocator_es_attribution.csv`, Exhibit A10) and its leave-one-year-out and
block-bootstrap robustness (`allocator_es_robustness.py`), the filing-cadence forecast error by pair
class (`voltarget_prescription.py`, `voltarget_snapshots.py`, with `data/commodity/voltarget_*_pairs.csv`),
the detail behind the divergence section (Exhibit 4): the holdings-level-plus-gap regression, the
all-pairs table, and the risk-parity-book test showing the pair signal fades at the book level
(`divergence_conditional_H.py`, `divergence_across_pairs.py`, `divergence_predicts_losses.py`,
`divergence_tail_book.py`, with `data/commodity/divergence_conditional_H_results.csv`),
the diagonal companion to that section, the holdings-implied-versus-stream volatility gap on the
eight-sleeve book (`vol_gap_rp_book.py`, with `data/commodity/vol_gap_rebook.csv`),
the equity-sleeve validation
(`equity_factors/validate_french.py`), and the active-ETF robustness variants
(`etf_edge_by_history.py`, `etf_edge_robust_feedstart.py`, `etf_fullwindow_race.py`, `switch_timesplit.py`, `etf_churn_validation.py`, with the
`data/etf_panel/pooled_neutral_all*` variants: `_L63` the matched halflife, `_fullwin` the
full-record subset, `_h63` the quarter horizon, `_L63_fullwin_allocation` the allocation class),
and the ex-ante re-cut of the persistence strata (`etf_causal_persistence.py`, the expanding-window
phibar(63) and the two-filing phi(63) from each fund's books up to the forecast date;
`etf_causal_strata.py`, the same full-window chunks re-binned on those measures and passed through the
unchanged analysis, switch and dyadic scripts; outputs in `data/etf_panel/causal_phibar/` and
`data/etf_panel/causal_twosnap/`).

### Data dictionary

- `data/etf_panel/pooled_neutral_all*/rmse_by_fixedband_age.csv`: root-mean-square forecast-error
  ratio to the return stream, by persistence band and age of the book, one row per estimator. The
  directory suffixes are the run: none is the every-pair comparison at halflife 252, `_L63` the matched
  halflife, `_L63_fullwin` the full-record subset (the whole-record descriptive sort of Online
  Appendix G), `_fullwin` its halflife-252 counterpart, `_h63` the quarter horizon,
  `_L63_fullwin_allocation` the multi-asset allocation class that is the article's Exhibit 9
  allocation row.
- `data/etf_panel/causal_phibar/`, `causal_twosnap/`: the same files for the full-window comparison
  with the strata assigned from persistence known at the forecast date (expanding-window phibar(63);
  the two-filing phi(63) of the rule). `causal_twosnap/` is the sort the article's Exhibit 9 reports;
  `causal_phibar/` is the expanding-window sort of Online Appendix G. Pooled and allocation-only, with
  `coverage_*.csv` giving the pair-dates re-binned. Fund-level persistence is on the licensed side and
  is not shipped.
- `.../panel_regressions.csv`: clustered-regression coefficients (loss and staleness cost) by age
  and persistence quartile.
- `.../sizing_by_quartile_age.csv`: the volatility-sizing test, holdings against stream.
- `.../fresh_edge_by_history.csv`, `switch_timesplit.csv`: the record-length split and the
  out-of-time scoring of the switch; `fresh_by_band.csv` (`etf_fresh_by_band.py`) the fresh
  edge regressed on the persistence-band dummies; `pooled_risk_all/` the same comparison with
  the market factor left in; `dyadic_regressions.csv` and `twoway_bootstrap_cells.csv`
  (`etf_dyadic_inference.py`) the dependence checks, dyadic-by-date clustering of the two key
  regressions and fund, quarter-block and crossed bootstraps of the ratio cells.
- The return-based factor benchmarks (`etf_factor_benchmark.py`, called inside
  `etf_staleness_experiment.py`; synthetic checks in `test_etf_factor_benchmark.py`) are the
  estimators `factor corr K5`, `factor corr K10` (statistical factors of the support covariance)
  and `proxy factor corr, core 13` / `extended 22` (a named ETF proxy set fixed before the run) in
  every `rmse_by_fixedband_age.csv` and `panel_regressions.csv`, and the `_F5`, `_F10`, `_P13`,
  `_P22` columns of `fresh_edge_by_history.csv`.
- `data/census/etf_class_overrides.csv` is the class map applied last by every script that reads
  the classes (35 rows, regenerated from the complete brochure classification on 2026-09-21).
  The 29-row map it replaced was written from the first five classification batches only, and the
  panels reported before that date carried thirteen funds in the allocation class that the completed
  classification types as fixed income, options or semi-transparent, or as long-only or long/short equity.
- `.../edge_robust_feedstart.csv`: the record-length split recomputed with, and without, the funds
  whose first daily holdings sit within a trading quarter of the panel's 2017 feed start, to show
  the sub-two-year edge is genuine youth and not the feed's own truncation.
- `data/commodity/phi_persistence.csv`, `equity_persistence.csv`: mix persistence phi(h) by sleeve
  and horizon, for the commodity and the equity-factor sleeves.
- `data/commodity/equity_validation_vs_french.csv`: correlation of each equity sleeve with Kenneth
  French's published factor.
- `data/commodity/voltarget_*_pairs.csv`: per-sleeve-pair forecast-error ratios by filing cadence.
- `data/commodity/allocator_multiasset_L34_reb{1,21,63}.csv`: risk-parity delivery, turnover and
  Sharpe by estimator, rebalanced daily, monthly and quarterly.
- `data/commodity/allocator_tail_es.csv`: expected shortfall of the 63-day forward drawdown of the
  holdings blend and the return stream, by tail fraction, Exhibit A9 (`allocator_tail_es.py`).
- `data/commodity/allocator_tilt_decomp_summary.csv`: the daily blend-minus-stream return split into
  scale, static sleeve tilt and timing (`allocator_tilt_decomp.py`).
- `data/commodity/allocator_insurance.csv`: the alignment factorization of the tail protection,
  Exhibit 7, split into the adverse tail and the calm body; staleness, current dependence, P&L
  sensitivity, gross effect, alignment and protection per sleeve-pair-day (`allocator_staleness_crowding.py`).
- `data/commodity/allocator_es_attribution.csv`: the holdings-minus-stream expected-shortfall gap
  attributed to each sleeve-pair correlation at four tail depths, Exhibit A10, from the pathwise
  Aumann-Shapley decomposition (`allocator_es_attribution.py`, on `allocator_pnl_attribution.py`).
- `data/commodity/allocator_es_loyo.csv`: the leave-one-calendar-year-out stability of that shortfall
  gap, the total and the three leading pairs recomputed dropping each year (`allocator_es_robustness.py`).
- `data/commodity/panel_oos_pooled.csv`, `panel_oos_pooled_recent.csv`: Exhibit 3, the covariance
  forecast RMSE, win share and Diebold-Mariano statistic pooled over the ten sleeve pairs, over the
  full 1997-2025 window and the 2021-2025 subsample (`panel_oos_all_pairs.py`).
- `data/commodity/lambda_regime_pairs.csv`, `regime_indicator_pairs.csv`: the stress uplift Lambda
  and the causal VIX / asset-volatility indicators by pair, on the 1992-2025 daily panel
  (`lambda_regime.py`, `regime_indicator.py`); the stress-window residuals of the same paragraph are the
  `resid_corr_*` columns of `voltarget_prescription_pairs.csv`.
- `data/etf_panel/divergence_check.csv`: the one-fund divergence slope on the ETF panel.
- `data/etf_panel/clone_nav_fidelity.csv`: per-fund daily-return correlation between the priced
  clone and the fund's reported NAV (median 0.985), the fidelity behind the clone-not-NAV limitation.
- `data/etf_panel/churn_validation_summary.csv`: whether the composition persistence can be
  recovered from a periodic filing and a NAV, tested on the ETF panel against the true daily
  persistence. On real NAVs the estimate has negative skill (class accuracy below the no-skill
  baseline); it works only when the reported return is the clean clone, so a NAV cannot substitute
  for holdings.
- `data/commodity/divergence_conditional_H_results.csv`: forward joint risk regressed on the
  holdings level and the gap together (pair fixed effects, block bootstrap); with the holdings level
  in the model the gap collapses to zero, so the divergence forecast is the holdings level, not the
  disagreement.
- `data/census/`: the public-filing layer, described in the next section. `ncen_etf_census.csv` is
  one row per ETF series and N-CEN report year (2018 onward); `ncen_active_etfs.csv` the 2,829
  active, non-index series with their tickers and years; `ncen_active_types.csv` the same series
  with the strategy type read from the fund's name; `census_download_list*.csv` the tickers pulled
  from the holdings vendor, by stratum; `etf_brochure_crosswalk.csv` each admitted fund's adviser
  CRD numbers from N-CEN and the brochure it was classified from; `etf_type_classification.csv` the
  brochure-based class of the 471 admitted funds (type, whether it rotates, transparency,
  confidence, the source passage); `etf_class_overrides.csv` the 35 funds whose brochure class
  differs from the name class; `sample_by_band.csv` the article's sample table, Exhibit 8. The
  crosswalk and classification files carry each fund's persistence reading, a single number per
  fund derived from the licensed holdings, so that the classes can be checked against the bands.

## The census and classification layer

The active-ETF sample starts from public filings and is rebuilt from them here. The scripts in
`scripts/census/` keep the path constants of the author's tree (`research/etf_universe/`,
`ncen_data/`, `adv_data/brochures/`); their shipped outputs are in `data/census/`.

1. `ncen_census.py` downloads every quarterly Form N-CEN structured data set from sec.gov and
   builds the census: one row per ETF series per report year with the fund's own index flag, fund
   of funds and money market flags, net assets, tickers, adviser and termination. The active,
   non-index series are the population (`ncen_active_etfs.csv`).
2. `etf_census_batches.py split` writes the population in batches; agents type each series from
   its name with the fixed prompt `etf_census_review_prompt.txt`; `merge` joins the labels back
   (`ncen_active_types.csv`) and writes the download lists by stratum.
3. `adv_fetch.py brochures-bulk` fetches each adviser's Form ADV Part 2A brochure from the SEC
   FOIA archives (or `brochures` for a targeted fetch) and extracts the Item 4 and Item 8 text.
4. `etf_brochure_crosswalk.py` reads each admitted fund's adviser CRD from the N-CEN filing
   (`ADVISER.tsv`, no name matching) and points it at the brochure text on disk.
5. `etf_type_batches.py` writes classification batches; agents classify each fund from its
   brochure with the fixed prompt `etf_brochure_type_prompt.txt`; `etf_type_merge.py` merges the
   results (`etf_type_classification.csv`) and writes the overrides applied to the classes.
6. `scripts/etf_sample_table.py` produces the sample table (`sample_by_band.csv`) from the
   admission rule's output, the persistence readings and these classes.

The admission rule itself (`etf_universe_rule.py`, thresholds fixed before any forecast was run) reads
the licensed holdings, so its output, the list of admitted funds, is on the licensed side; the
funds it admitted are the rows of the classification file.

## The raw-to-intermediate stage

These steps ingest the licensed feeds and public filings into the intermediate priced panels. They
are not included, because they require the licensed data and the author's data-access layer; they
are named here for completeness: the commodity-book build (`public_panel_build.py`,
`panel_oos_benchmarks.py`, `commodity_crowding_analysis.py`, `joint_loss_probability.py`), the
equity-sleeve build (`extract_compustat.py`, `build_ff_annual.py`, `build_equity_holdings.py`) and
the active-ETF pricing (`massive.py`, `etf_coverage.py`, `etf_universe_rule.py`, `etf_race.py`).
The census and classification stage that precedes the pricing is public and shipped, above.

The active-ETF analysis scripts in `scripts/` (`etf_staleness_experiment.py`,
`etf_sizing_race.py`, `etf_divergence_check.py`, `etf_edge_by_history.py`,
`etf_fullwindow_race.py`, `switch_timesplit.py`) import a shared pricing and covariance library,
`etf_race.py` and `etf_coverage.py`, which reads the licensed ETF Global books to build the priced
clones. That library and the priced clones are on the licensed side and are not shipped, so these
scripts are a record of the method rather than code that runs from this repository alone; their
outputs are the aggregates in `data/etf_panel/`.
