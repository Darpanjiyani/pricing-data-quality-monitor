import numpy as np
import pandas as pd

from pdq.checks import equity_cross_source, equity_return_outliers
from conftest import prices_frame, random_walk


def universe(n_days=120, n_syms=8):
    return {f"S{i}": random_walk(n_days, i) for i in range(n_syms)}


def test_bad_print_reported_once(cfg):
    w = universe()
    w["S0"][80] = round(w["S0"][80] * 10, 2)
    out = equity_return_outliers(prices_frame(w), cfg)
    hit = out[out.key == "S0"]
    assert list(hit.check) == ["bad_print_spike"]       # the next day's drop back is not a second event


def test_unadjusted_split(cfg):
    w = universe()
    w["S1"] = w["S1"][:90] + [round(c / 3, 2) for c in w["S1"][90:]]
    out = equity_return_outliers(prices_frame(w), cfg)
    hit = out[out.key == "S1"]
    assert list(hit.check) == ["possible_unadjusted_split"] and "1-for-3" in hit.detail.iloc[0]


def test_market_wide_move_not_flagged(cfg):
    w = universe()
    for s in w:                                          # everyone drops 8% the same day
        w[s] = w[s][:70] + [round(c * 0.92, 2) for c in w[s][70:]]
    assert equity_return_outliers(prices_frame(w), cfg).empty


def test_cross_source_ignores_dividend_drift(cfg):
    w = universe(60, 2)
    a = prices_frame(w, source="a")
    b = a.copy(); b["source"] = "b"
    factor = np.where(pd.to_datetime(b.date) < "2024-02-15", 0.993, 1.0)   # 0.7% adjustment step
    b["close"] = b.close * factor
    assert equity_cross_source(a, b, cfg, ("a", "b")).empty
    bad = b.date.unique()[40]
    b.loc[(b.symbol == "S0") & (b.date == bad), "close"] *= 1.05
    out = equity_cross_source(a, b, cfg, ("a", "b"))
    assert list(zip(out.check, out.date)) == [("cross_source_price", bad)]
