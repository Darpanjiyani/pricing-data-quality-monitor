import io

import numpy as np
import pandas as pd

from pdq.checks import curve_kinks, missing_tenors, yield_jumps
from pdq.db import connect, replace_source, run_sql_check
from pdq.fetch import parse_fred_csv, parse_treasury_csv
from pdq.sample import sample_curve


def curve():
    y, ya = sample_curve(n_days=150, seed=3)
    return y.copy(), ya.copy()


def test_spike_flagged_once_and_neighbours_quiet(cfg):
    y, _ = curve()
    d = sorted(y.date.unique())[100]
    y.loc[(y.date == d) & (y.tenor == "5Y"), "yield"] += 0.6
    j = yield_jumps(y, cfg)
    assert list(zip(j.check, j.date, j.key)) == [("yield_spike", d, "5Y")]
    k = curve_kinks(y, cfg)
    assert set(k[k.date == d].key) <= {"5Y"}             # 3Y and 7Y are not blamed


def test_end_tenor_spike_does_not_blame_neighbour(cfg):
    y, _ = curve()
    d = sorted(y.date.unique())[90]
    y.loc[(y.date == d) & (y.tenor == "30Y"), "yield"] += 0.5
    k = curve_kinks(y, cfg)
    assert k[k.date == d].empty


def test_missing_tenor(cfg):
    y, _ = curve()
    d = sorted(y.date.unique())[50]
    y.loc[(y.date == d) & (y.tenor == "2Y"), "yield"] = np.nan
    m = missing_tenors(y, cfg)
    assert list(zip(m.date, m.key)) == [(d, "2Y")]


def test_range_and_cross_source(cfg):
    y, ya = curve()
    d = sorted(y.date.unique())[60]
    y.loc[(y.date == d) & (y.tenor == "10Y"), "yield"] *= 10
    ya.loc[(ya.date == d) & (ya.tenor == "2Y"), "yield"] += 0.05
    conn = connect()
    replace_source(conn, "yields", y); replace_source(conn, "yields", ya)
    r = run_sql_check(conn, "yield_range", {"source": "treasury_sample", "min_yield": -1, "max_yield": 25})
    assert list(r.key) == ["10Y"]
    x = run_sql_check(conn, "yield_cross_source",
                      {"primary": "treasury_sample", "secondary": "fred_sample", "xsrc_tol_bp": 2})
    assert set(x.key) == {"10Y", "2Y"}


def test_parse_treasury_csv():
    raw = pd.read_csv(io.StringIO(
        'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
        "12/31/2025,3.74,3.75,3.67,3.67,3.63,3.59,3.48,3.47,3.55,3.73,3.94,4.18,4.79,4.84\n"
        "12/30/2025,3.65,,3.65,3.65,3.63,3.59,3.47,3.45,3.50,3.68,3.89,4.14,4.76,4.81\n"))
    out = parse_treasury_csv(raw)
    assert len(out) == 28 and out.date.max() == "2025-12-31"
    assert out[(out.tenor == "10Y") & (out.date == "2025-12-31")]["yield"].item() == 4.18
    assert out[(out.tenor == "1.5M") & (out.date == "2025-12-30")]["yield"].isna().item()


def test_parse_fred_csv_drops_blank_days():
    raw = pd.read_csv(io.StringIO("observation_date,DGS10\n2025-01-02,4.57\n2025-01-09,\n2025-01-10,4.77\n"))
    out = parse_fred_csv(raw, "DGS10", "10Y", 120)
    assert list(out.date) == ["2025-01-02", "2025-01-10"]


def test_parse_yahoo_frame_shape():
    from pdq.fetch import parse_yahoo_frame
    idx = pd.DatetimeIndex(["2025-01-02", "2025-01-03"], name="Date")
    cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    data = pd.DataFrame(np.arange(24, dtype=float).reshape(2, 12) + 1, index=idx, columns=cols)
    data.loc[:, "MSFT"] = np.nan                         # a symbol that failed to download
    out = parse_yahoo_frame(data, ["AAPL", "MSFT"])
    assert list(out.symbol.unique()) == ["AAPL"] and list(out.date) == ["2025-01-02", "2025-01-03"]
    assert out.close.tolist() == [4.0, 16.0]
