"""Evaluation metric of the paper: root mean square error, Eq. (3).

RMSE = sqrt( sum_i (y_i - y_hat_i)^2 / n )        (smaller is better)

For Dataset 1 the paper reports a percentage RMSE; here it is normalised by
the mean absolute actual demand of the test set (RMSE %).
"""
from __future__ import annotations

import numpy as np


def rmse(y, y_hat) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def rmse_pct(y, y_hat) -> float:
    return 100.0 * rmse(y, y_hat) / float(np.mean(np.abs(y)))
