"""Synthetic sample data, so the project runs and is testable with no internet access.

This data is made up. Results on it show the checks work; numbers for a resume must
come from a run on real data (see README).
  - equities: one market factor plus fat-tailed (Student-t) idiosyncratic returns
  - a second "vendor" whose closes differ by a slowly stepping adjustment factor, the way
    two vendors' dividend adjustments differ
  - Treasury curve from a Nelson-Siegel model with random-walk level, slope and curvature;
    the FRED copy carries the same values for the tenors FRED publishes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .fetch import FRED_SERIES, PRICE_COLS, TREASURY_COLUMNS, YIELD_COLS


def trading_days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))


def sample_prices(symbols: list[str], start: str = "2023-01-03", n_days: int = 500,
                  seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = trading_days(start, n_days)
    t4 = lambda size, vol: rng.standard_t(4, size) * vol / np.sqrt(2)  # t(4) has variance 2
    market = t4(n_days, 0.010)
    rows, alt = [], []
    for sym in symbols:
        beta = rng.uniform(0.7, 1.3)
        r = beta * market + t4(n_days, 0.015)
        close = rng.uniform(20, 500) * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.004, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        vol = rng.lognormal(15, 0.4, n_days).round()
        df = pd.DataFrame({"source": "sample", "symbol": sym, "date": days.strftime("%Y-%m-%d"),
                           "open": open_.round(2), "high": high.round(2), "low": low.round(2),
                           "close": close.round(2), "volume": vol})
        rows.append(df)
        # second vendor: dividend-style adjustment steps of 0.2-1.0% about once a quarter
        steps = np.zeros(n_days)
        for i in rng.choice(np.arange(20, n_days), size=n_days // 63, replace=False):
            steps[i] = -rng.uniform(0.002, 0.010)
        factor = np.exp(np.cumsum(steps) - np.sum(steps))
        a = df.copy()
        a["source"] = "sample_alt"
        for col in ("open", "high", "low", "close"):
            a[col] = (a[col] * factor).round(2)
        alt.append(a)
    return (pd.concat(rows, ignore_index=True)[PRICE_COLS],
            pd.concat(alt, ignore_index=True)[PRICE_COLS])


def nelson_siegel(m: np.ndarray, b0: float, b1: float, b2: float, lam: float = 0.0609) -> np.ndarray:
    x = lam * m
    f1 = (1 - np.exp(-x)) / x
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x))


def sample_curve(start: str = "2023-01-03", n_days: int = 500,
                 seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = trading_days(start, n_days)
    tenors = [(lab, mon) for lab, mon in TREASURY_COLUMNS.values()]
    m = np.array([mon for _, mon in tenors], dtype=float)
    b = np.array([4.3, 0.9, -1.2])
    rows = []
    for d in days:
        b = b + rng.normal(0, [0.035, 0.045, 0.06]) + 0.01 * (np.array([4.2, 0.0, -0.5]) - b)
        y = nelson_siegel(m, *b) + rng.normal(0, 0.008, len(m))
        for (lab, mon), v in zip(tenors, y):
            rows.append(("treasury_sample", d.strftime("%Y-%m-%d"), lab, mon, round(float(v), 2)))
    tre = pd.DataFrame(rows, columns=YIELD_COLS)
    fred_tenors = {lab for lab, _ in FRED_SERIES.values()}
    fred = tre[tre.tenor.isin(fred_tenors)].assign(source="fred_sample")
    return tre, fred.reset_index(drop=True)
