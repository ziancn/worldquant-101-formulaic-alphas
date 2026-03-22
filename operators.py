"""
This file houses operators mentioned in the paper.

From the operation dimension perspective, there are two main categories:
    1. Operate along the axis of time series (one ticker over a period of time)
    2. Operate along the axis of universe (aka "cross-sectional": all memebers at a point of time)

The naming rules of the paper is not clear, which could be confusing sometimes when you are not 
familiar with the operators (have to jump to the appendix at the end of the paper all the time).

Therefore for all operators here I will use a prefix to indicate the direction of the operator:
    1. Time series axis : ts_, rolling_
    2. Cross-sectional  : cs_

For certain metric like `correlation`, the requirment of time series is already embedded, hence the 
prefix will be omitted for simplicity.
"""


import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy.stats import rankdata


def signed_power(x, p):
    return np.sign(x) * np.abs(x)**p


def ts_sum(df: pd.DataFrame, window=10):
    """Summation of data in the time series"""
    return df.rolling(window).sum()


def ts_sma(df: pd.DataFrame, window=10):
    """Simple moving average: average of data in the time series"""
    return df.rolling(window).mean()


def ts_stddev(df: pd.DataFrame, window=10):
    """Standard deviation of data in the time series"""
    return df.rolling(window).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, window=10):
    """Correlation of x and y in the time series"""
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window=10):
    """Covariance of x and y in the time series"""
    return x.rolling(window).cov(y)


# def _get_rank_in_rolling_window(na: npt.NDArray):
#     """The rank of the last value in the array."""
#     return rankdata(na)[-1]


def ts_rank(df: pd.DataFrame, window=10):
    """Rank of each data in its rolling time series window"""
    # Lambda x typehint: Numpy NDArray, `apply` by default = 0, on row(index) axis
    return df.rolling(window).apply(lambda x: rankdata(x)[-1])


# def rolling_prod(na: np.array):
#     """The product of the values in the array"""
#     return np.prod(na)


def ts_product(df: pd.DataFrame, window=10):
    """Product of data in the time series"""
    return df.rolling(window).apply(np.prod)


def ts_min(df: pd.DataFrame, window=10):
    """Minimum value in the time series"""
    return df.rolling(window).min()


def ts_max(df: pd.DataFrame, window=10):
    """Maximum value in the time series"""
    return df.rolling(window).max()


def delta(df: pd.DataFrame, period=1):
    """Today's value minus the value 'period' days ago"""
    return df.diff(period)


def delay(df: pd.DataFrame, period=1):
    """Value of the data `period` days ago. Same as DataFrame shift"""
    return df.shift(period)


def cs_rank(df: pd.DataFrame):
    """Cross-sectional rank(percentile to normalise). This is the ranking along column axis"""
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, k=1):
    """Return a pandas DataFrame rescaled df such that sum(abs(df)) = k"""
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df: pd.DataFrame, window=10):
    """Return the index of the max value in the rolling window, 1 based"""
    return df.rolling(window).apply(np.argmax) + 1 


def ts_argmin(df: pd.DataFrame, window=10):
    """Return the index of the min value in the rolling window, 1 based"""
    return df.rolling(window).apply(np.argmin) + 1