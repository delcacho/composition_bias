"""
panel_pivot.py -- dense (dates x gvkey) return matrix from the long stock panel, filled directly.

pivot_table(index="date", columns="gvkey", values="ret") on the 24.5M-row panel groups before it reshapes and needs several
times the matrix in memory; pivot_mean accumulates sums and counts into the dense matrix directly, so the peak is the
matrix itself, and returns exactly what pivot_table returns: the mean over duplicate (date, gvkey) rows, NaN where a cell
has no row, dates and columns sorted unless given.
"""
import numpy as np
import pandas as pd


def pivot_mean(frame, dates=None, cols=None, date_col="date", key_col="gvkey", value_col="ret", dtype="float64"):
    """dense DataFrame (dates x cols) of the mean value per (date, key); rows outside `dates`/`cols` are ignored."""
    dates = pd.DatetimeIndex(sorted(pd.unique(frame[date_col]))) if dates is None else pd.DatetimeIndex(dates)
    cols = pd.Index(sorted(pd.unique(frame[key_col]))) if cols is None else pd.Index(cols)
    di = dates.get_indexer(pd.DatetimeIndex(frame[date_col])); ci = cols.get_indexer(frame[key_col])
    ok = (di >= 0) & (ci >= 0)
    S = np.zeros((len(dates), len(cols)), dtype="float64"); N = np.zeros((len(dates), len(cols)), dtype="float64")
    np.add.at(S, (di[ok], ci[ok]), frame[value_col].to_numpy(dtype="float64")[ok])
    np.add.at(N, (di[ok], ci[ok]), 1.0)
    with np.errstate(invalid="ignore"):
        M = np.where(N > 0, S / np.where(N > 0, N, 1.0), np.nan)
    del S, N
    return pd.DataFrame(M.astype(dtype, copy=False), index=dates, columns=cols)
