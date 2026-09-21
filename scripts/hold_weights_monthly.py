"""
hold_weights_monthly.py -- the monthly-rebalanced public book: the daily textbook book re-formed once a
month and held. This is the book behind the article's second half (Exhibits 5-7, appendix A4, A6, A9,
A12): the same asset returns, with each sleeve's stacked weights frozen at a month-end and held
until the next month-end.

usage: python research/hold_weights_monthly.py <daily_out_dir> <monthly_out_dir> [--rule=prev_month_end|first_day] [--returns=<run_dir>]

Reads <daily_out_dir>/preqp_weights_stacked.csv and asset_returns_by_asset.csv
Writes <monthly_out_dir>/preqp_weights_stacked.csv (held weights) and a copy of asset_returns_by_asset.csv,
so the run dir has the layout every pipeline script reads.
Rule prev_month_end (default): the book formed on a month's last trading day applies from that day until
the next month-end (the forming day counts); before the first month-end, the first day's book. Rule
first_day: the book of the first trading day of month m, held.
"""
import os
import shutil
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
rule = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rule=")), "prev_month_end")
returns_dir = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--returns=")), src) # run dir holding asset_returns_by_asset.csv
os.makedirs(dst, exist_ok=True)
W = pd.read_csv(os.path.join(src, "preqp_weights_stacked.csv"), index_col=0)
W.index = pd.to_datetime(W.index)
month = W.index.to_period("M")
if rule == "prev_month_end":
    # the book formed on each month's last trading day applies from that day until the next month-end
    # (the day the book is formed counts); before the first month-end, the first day's book
    last_of_month = W.groupby(month).tail(1)
    held = last_of_month.reindex(W.index).ffill()
    held.iloc[0] = W.iloc[0]; held = held.ffill()
else:
    first_of_month = W.groupby(month).head(1)
    ref = first_of_month.copy(); ref.index = ref.index.to_period("M")
    held = ref.reindex(month); held.index = W.index
held.to_csv(os.path.join(dst, "preqp_weights_stacked.csv"), float_format="%.10g")
shutil.copyfile(os.path.join(returns_dir, "asset_returns_by_asset.csv"), os.path.join(dst, "asset_returns_by_asset.csv"))
print(f"{rule}: {len(held)} days, {held.shape[1]} sleeve||asset columns -> {dst}/preqp_weights_stacked.csv (+ asset returns copied)")
