"""Shared statistics: robust rolling z-scores and spike-then-revert detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826  # makes the MAD comparable to a standard deviation for normal data


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def robust_z(s: pd.Series, window: int, min_periods: int, floor: float = 1e-12) -> pd.Series:
    """z-score of each value against the median and MAD of the *previous* `window` values.

    Uses only past data, so a bad value cannot hide itself by inflating its own scale,
    and median/MAD instead of mean/std so one earlier spike does not blunt later ones.
    """
    past = s.shift(1)
    med = past.rolling(window, min_periods=min_periods).median()
    mad = past.rolling(window, min_periods=min_periods).apply(_mad, raw=True)
    scale = (MAD_TO_SD * mad).clip(lower=floor)
    return (s - med) / scale


def reverts_next_day(x: pd.Series, tol: float) -> pd.Series:
    """True at t when the move at t+1 cancels the move at t to within tol (as a share of |x_t|).

    A price printed 10x too high for one day shows up as +230% then -230% in log terms;
    that is a bad print, not two separate events.
    """
    nxt = x.shift(-1)
    return (np.sign(nxt) == -np.sign(x)) & ((x + nxt).abs() <= tol * x.abs())


def per_column_diff(wide: pd.DataFrame, fn=lambda s: s.diff()) -> pd.DataFrame:
    """Apply fn to each column's non-missing values only, so a missing day does not create a gap."""
    return wide.apply(lambda s: fn(s.dropna()).reindex(s.index))
