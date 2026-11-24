import numpy as np

from pdq.db import connect, replace_source, run_sql_check
from conftest import prices_frame, random_walk


def load(df):
    conn = connect()
    replace_source(conn, "prices", df)
    return conn


def test_duplicates_and_validity():
    df = prices_frame({"A": [10, 11, 12, 13]})
    dup = df.iloc[[1]].assign(close=11.5)
    df.loc[2, ["high", "low"]] = [11.0, 13.0]            # high below low
    conn = load(__import__("pandas").concat([df, dup]))
    d = run_sql_check(conn, "duplicates", {"source": "t"})
    assert list(d.date) == [df.date[1]] and "different closes" in d.detail[0]
    v = run_sql_check(conn, "price_validity", {"source": "t", "eps": 1e-4})
    assert df.date[2] in set(v.date)


def test_stale_run_flags_days_after_first():
    closes = random_walk(10, 1)
    closes[4:8] = [closes[3]] * 4                        # 4 repeats after day 3
    conn = load(prices_frame({"A": closes}))
    s = run_sql_check(conn, "stale_prices", {"source": "t", "min_repeats": 3})
    assert len(s) == 4 and "4 repeats" in s.detail.iloc[0]


def test_short_repeat_not_stale():
    closes = random_walk(10, 2)
    closes[4:6] = [closes[3]] * 2
    conn = load(prices_frame({"A": closes}))
    assert run_sql_check(conn, "stale_prices", {"source": "t", "min_repeats": 3}).empty


def test_missing_day_needs_market_coverage():
    walks = {s: random_walk(6, i) for i, s in enumerate("ABCDE")}
    walks["A"][3] = np.nan                               # A missing while 4 of 5 report
    for s in walks:
        walks[s][5] = np.nan if s != "B" else walks[s][5]  # only B on the last day: not a trading day
    conn = load(prices_frame(walks))
    m = run_sql_check(conn, "missing_days", {"source": "t", "coverage": 0.8})
    assert list(m.key) == ["A"]
