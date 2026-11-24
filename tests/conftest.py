import numpy as np
import pandas as pd
import pytest

from pdq.config import load_config
from pdq.sample import sample_curve, sample_prices


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample(cfg):
    p, pa = sample_prices(cfg["universe"]["symbols"][:12], n_days=200)
    y, ya = sample_curve(n_days=200)
    return p, pa, y, ya


def prices_frame(closes: dict, start="2024-01-01", source="t"):
    """closes: symbol -> list of closes (np.nan = no row). OHLC built around the close."""
    days = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = []
    for sym, cs in closes.items():
        for d, c in zip(days, cs):
            if c is None or (isinstance(c, float) and np.isnan(c)):
                continue
            rows.append(dict(source=source, symbol=sym, date=d.strftime("%Y-%m-%d"),
                             open=c, high=c * 1.01, low=c * 0.99, close=c, volume=1000.0))
    return pd.DataFrame(rows)


def random_walk(n, seed, start=100.0, vol=0.01):
    rng = np.random.default_rng(seed)
    return list(np.round(start * np.exp(np.cumsum(rng.normal(0, vol, n))), 2))
