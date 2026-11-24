"""Plant known errors in clean data, with a label for each, so the checks can be scored.

Error types mirror what goes wrong in real vendor feeds:
  equity : stale_run, missing_day, duplicate_row, bad_print, unadjusted_split,
           invalid_ohlc, cross_source_bad_value
  rates  : yield_spike, decimal_slip, missing_tenor, cross_source_rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_CHECKS = {
    "stale_run": {"stale_price"},
    "missing_day": {"missing_day"},
    "duplicate_row": {"duplicate_row"},
    "bad_print": {"bad_print_spike", "return_outlier"},
    "unadjusted_split": {"possible_unadjusted_split", "cross_source_level_shift"},
    "invalid_ohlc": {"price_validity"},
    "cross_source_bad_value": {"cross_source_price"},
    "yield_spike": {"yield_spike", "curve_kink", "cross_source_yield"},
    "decimal_slip": {"yield_out_of_range"},
    "missing_tenor": {"missing_tenor"},
    "cross_source_rate": {"cross_source_yield"},
}
STALE_REPEATS = 4
WARMUP = 30  # leave the first days alone: rolling statistics are still filling


class _Picker:
    """Pick (key, date index) targets that do not overlap earlier picks."""

    def __init__(self, keys, n_dates, rng, gap=6):
        self.keys, self.n, self.rng, self.gap, self.used = list(keys), n_dates, rng, gap, []

    def pick(self, span=1):
        for _ in range(10_000):
            k = self.keys[self.rng.integers(len(self.keys))]
            i = int(self.rng.integers(WARMUP, self.n - span - 3))
            if all(k != uk or abs(i - ui) > self.gap + span for uk, ui in self.used):
                self.used.append((k, i))
                return k, i
        raise RuntimeError("could not place another error; lower n_per_type")


def inject(prices: pd.DataFrame, prices_alt: pd.DataFrame | None, yields: pd.DataFrame,
           yields_alt: pd.DataFrame | None, n_per_type: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    p = prices.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    pa = None if prices_alt is None else prices_alt.copy()
    y = yields.copy()
    ya = None if yields_alt is None else yields_alt.copy()
    labels = []
    lab = lambda et, ds, d, k: labels.append(dict(error_type=et, dataset=ds, date=d, key=k))

    dates = sorted(p.date.unique())
    idx = {(s, d): i for i, (s, d) in enumerate(zip(p.symbol, p.date))}
    pk = _Picker(p.symbol.unique(), len(dates), rng)
    ohlc = ["open", "high", "low", "close"]
    drop, dupes = [], []

    for _ in range(n_per_type):
        s, i = pk.pick(STALE_REPEATS + 1)            # stale feed repeats the whole bar
        src = idx[(s, dates[i])]
        for j in range(1, STALE_REPEATS + 1):
            r = idx[(s, dates[i + j])]
            p.loc[r, ohlc] = p.loc[src, "close"]
            lab("stale_run", "equity", dates[i + j], s)

        s, i = pk.pick()                             # row never delivered
        drop.append(idx[(s, dates[i])]); lab("missing_day", "equity", dates[i], s)

        s, i = pk.pick()                             # loaded twice, second copy slightly off
        row = p.loc[[idx[(s, dates[i])]]].copy(); row["close"] = (row["close"] * 1.01).round(2)
        dupes.append(row); lab("duplicate_row", "equity", dates[i], s)

        s, i = pk.pick()                             # bad print: decimal shift or fat finger
        p.loc[idx[(s, dates[i])], ohlc] *= rng.choice([10, 0.1, 1.25, 0.8])
        lab("bad_print", "equity", dates[i], s)

        s, i = pk.pick(60)                           # split not adjusted from date i onward
        k = int(rng.choice([2, 3, 4]))
        m = (p.symbol == s) & (p.date >= dates[i])
        p.loc[m, ohlc] = (p.loc[m, ohlc] / k).round(2); p.loc[m, "volume"] *= k
        lab("unadjusted_split", "equity", dates[i], s)

        s, i = pk.pick()                             # high and low swapped
        r = idx[(s, dates[i])]
        p.loc[r, ["high", "low"]] = p.loc[r, ["low", "high"]].to_numpy()
        lab("invalid_ohlc", "equity", dates[i], s)

        if pa is not None:                           # second vendor off by 5% for a day
            s, i = pk.pick()
            m = (pa.symbol == s) & (pa.date == dates[i])
            pa.loc[m, ohlc] = (pa.loc[m, ohlc] * rng.choice([1.05, 0.95])).round(2)
            lab("cross_source_bad_value", "equity", dates[i], s)

    p = pd.concat([p.drop(index=drop), *dupes], ignore_index=True)

    ydates = sorted(y.date.unique())
    yk = _Picker(y.tenor.unique(), len(ydates), rng)
    alt_tenors = set() if ya is None else set(ya.tenor.unique())
    for _ in range(n_per_type):
        t, i = yk.pick()
        m = (y.tenor == t) & (y.date == ydates[i])
        y.loc[m, "yield"] += rng.choice([-1, 1]) * rng.uniform(0.25, 0.75)
        lab("yield_spike", "rates", ydates[i], t)

        t, i = yk.pick()
        m = (y.tenor == t) & (y.date == ydates[i])
        y.loc[m, "yield"] *= 10
        lab("decimal_slip", "rates", ydates[i], t)

        t, i = yk.pick()
        m = (y.tenor == t) & (y.date == ydates[i])
        y.loc[m, "yield"] = np.nan
        lab("missing_tenor", "rates", ydates[i], t)

        if ya is not None:
            for _ in range(100):
                t, i = yk.pick()
                if t in alt_tenors:
                    break
            m = (ya.tenor == t) & (ya.date == ydates[i])
            ya.loc[m, "yield"] += 0.05
            lab("cross_source_rate", "rates", ydates[i], t)

    return p, pa, y, ya, pd.DataFrame(labels)
