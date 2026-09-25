"""The predictors of Section 3 of the paper.

Average baseline  - the existing market practice: the mean of the same
                    historical time stamps over the previous TW days.
LR   (Eq. 1)      - y_hat = a1*x1 + a2*x2 + ... + an*xn + b
NLR  (Eq. 2)      - y_hat = a1*x1^2 + ... + an*xn^2 + b1*x1 + ... + bn*xn + c
                    (second-order polynomial regression: squared + linear
                    terms, no cross terms - exactly as written in the paper)

All three are interpretable, non-black-box models: every coefficient can be
read off directly, which is the whole point of the paper.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class AverageBaseline:
    """Market baseline: mean of the same half-hour over the previous TW days.

    Assumes the first `tw` columns of X are the same-slot demand window."""

    def __init__(self, tw: int):
        self.tw = tw

    def fit(self, X, y=None):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, :self.tw].mean(axis=1)


class LR:
    """Linear regression, Eq. (1)."""

    def __init__(self):
        self.m = LinearRegression()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    @property
    def coefficients(self):
        return self.m.coef_, self.m.intercept_


class NLR:
    """Second-order polynomial (non-linear) regression, Eq. (2)."""

    def __init__(self):
        self.m = LinearRegression()

    @staticmethod
    def _phi(X: np.ndarray) -> np.ndarray:
        return np.hstack([X ** 2, X])

    def fit(self, X, y):
        self.m.fit(self._phi(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self._phi(X))

    @property
    def coefficients(self):
        return self.m.coef_, self.m.intercept_
