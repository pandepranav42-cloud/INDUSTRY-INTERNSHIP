"""Pre-processing step of the framework (Fig. 3, step 2 of the paper).

Two window geometries are used in the paper (Section 4.3):

* Model 1 / market baseline geometry - the inputs are the values at the SAME
  half-hour on each of the previous TW days (e.g. the 10:00 readings of the
  last 10 days predict tomorrow's 10:00 reading).
* Model 2 geometry - the inputs are the TW CONSECUTIVE half-hours immediately
  before the predicted one (e.g. 7:00 ... 9:30 predict 10:00 the same day),
  which respects the continuity of the time series.

Problem formulation (Section 2): given a series x = {x1..xT}, a window TW and
a prediction step s, the number of samples is n = T - TW - s + 1 and the
predictor is  f(x, W, TW) -> y_pred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 48


def day_matrix(series: pd.Series) -> pd.DataFrame:
    """Pivot a 30-min series into an (n_days x 48) matrix (rows=dates)."""
    df = series.to_frame("v")
    df["date"] = df.index.normalize()
    df["slot"] = df.index.hour * 2 + df.index.minute // 30
    return df.pivot(index="date", columns="slot", values="v")


def consecutive_windows(df: pd.DataFrame, features: list[str], tw: int,
                        target: str = "demand", step: int = 1):
    """Model 2 windows: TW previous half-hours of every feature -> demand at
    t (s = 1 step ahead).  Returns X (n, tw*k), y (n,), target timestamps."""
    n = len(df) - tw - step + 1
    X = np.hstack([
        np.lib.stride_tricks.sliding_window_view(df[f].to_numpy(), tw)[:n]
        for f in features])
    y = df[target].to_numpy()[tw + step - 1:]
    t = df.index[tw + step - 1:]
    return X, y, t


def same_slot_windows(mats: list[np.ndarray], target_mat: np.ndarray,
                      day_positions, tw: int, extra_per_day=None):
    """Model 1 / baseline geometry.

    mats           list of (n_days, 48) arrays - demand first, then exogenous
    target_mat     (n_days, 48) demand matrix
    day_positions  integer day positions used as prediction targets
    extra_per_day  optional (n_days, k) daily scalars (e.g. max temperature
                   of the target day) appended to every row of that day

    Returns X, y and meta = [(day_position, slot), ...] for every sample.
    """
    rows, ys, meta = [], [], []
    for p in day_positions:
        block = np.hstack([m[p - tw:p, :].T for m in mats])   # (48, tw*len(mats))
        if extra_per_day is not None:
            block = np.hstack([block,
                               np.repeat(extra_per_day[p][None, :],
                                         SLOTS_PER_DAY, axis=0)])
        rows.append(block)
        ys.append(target_mat[p, :])
        meta.extend((p, s) for s in range(SLOTS_PER_DAY))
    return np.vstack(rows), np.concatenate(ys), meta
