"""Download end-of-day data from free public sources.

Equities
  Yahoo Finance through the `yfinance` package (adjusted for splits, not for dividends).
  The equity cross-source check needs a second vendor; free ones tried (Stooq, Nasdaq.com)
  block scripted downloads, so it runs on the synthetic demo data only.
Treasury yield curve
  primary   : U.S. Treasury daily par yield curve CSV (home.treasury.gov)
  secondary : FRED constant-maturity series DGS1MO ... DGS30 (fred.stlouisfed.org)

Every function returns a long table in the same shape as the database tables:
  prices: source, symbol, date, open, high, low, close, volume
  yields: source, date, tenor, tenor_months, yield
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import urllib.request
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (pricing-data-quality-monitor)"}

TREASURY_COLUMNS = {
    "1 Mo": ("1M", 1), "1.5 Month": ("1.5M", 1.5), "2 Mo": ("2M", 2), "3 Mo": ("3M", 3),
    "4 Mo": ("4M", 4), "6 Mo": ("6M", 6), "1 Yr": ("1Y", 12), "2 Yr": ("2Y", 24),
    "3 Yr": ("3Y", 36), "5 Yr": ("5Y", 60), "7 Yr": ("7Y", 84), "10 Yr": ("10Y", 120),
    "20 Yr": ("20Y", 240), "30 Yr": ("30Y", 360),
}
FRED_SERIES = {
    "DGS1MO": ("1M", 1), "DGS3MO": ("3M", 3), "DGS6MO": ("6M", 6), "DGS1": ("1Y", 12),
    "DGS2": ("2Y", 24), "DGS3": ("3Y", 36), "DGS5": ("5Y", 60), "DGS7": ("7Y", 84),
    "DGS10": ("10Y", 120), "DGS20": ("20Y", 240), "DGS30": ("30Y", 360),
}
PRICE_COLS = ["source", "symbol", "date", "open", "high", "low", "close", "volume"]
YIELD_COLS = ["source", "date", "tenor", "tenor_months", "yield"]


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ------------------------------------------------------------------ Treasury
def treasury_curve(start: str, end: str | None = None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
               f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
               f"&field_tdr_date_value={year}&page&_format=csv")
        raw = pd.read_csv(io.BytesIO(_get(url)))
        frames.append(parse_treasury_csv(raw))
    out = pd.concat(frames, ignore_index=True)
    return out[(out.date >= start) & (out.date <= end)].reset_index(drop=True)


def parse_treasury_csv(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns=lambda c: c.strip())
    raw["date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    cols = [c for c in raw.columns if c in TREASURY_COLUMNS]
    long = raw.melt(id_vars="date", value_vars=cols, var_name="col", value_name="yield")
    long["tenor"] = long.col.map(lambda c: TREASURY_COLUMNS[c][0])
    long["tenor_months"] = long.col.map(lambda c: TREASURY_COLUMNS[c][1])
    long["source"] = "treasury"
    # Tenors that did not exist yet (e.g. 1.5 Month before 2025) are blank for the whole
    # year; drop those so they are not reported as missing.
    present = long.groupby("tenor")["yield"].transform(lambda s: s.notna().any())
    return long[present][YIELD_COLS]


# ------------------------------------------------------------------ FRED
def fred_curve(start: str, end: str | None = None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    frames = []
    for sid, (tenor, months) in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"
        raw = pd.read_csv(io.BytesIO(_get(url)))
        frames.append(parse_fred_csv(raw, sid, tenor, months))
    return pd.concat(frames, ignore_index=True)


def parse_fred_csv(raw: pd.DataFrame, sid: str, tenor: str, months: float) -> pd.DataFrame:
    date_col = "observation_date" if "observation_date" in raw.columns else "DATE"
    df = pd.DataFrame({
        "source": "fred",
        "date": pd.to_datetime(raw[date_col]).dt.strftime("%Y-%m-%d"),
        "tenor": tenor,
        "tenor_months": months,
        "yield": pd.to_numeric(raw[sid], errors="coerce"),  # FRED writes holidays as blank or "."
    })
    return df.dropna(subset=["yield"])[YIELD_COLS]


# ------------------------------------------------------------------ Equities
def yahoo_prices(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    # yfinance's default cache folder can clash on Windows ("Failed to create TzCache"); use our own
    try:
        yf.set_tz_cache_location(os.path.join(tempfile.gettempdir(), "pdq-yfinance-cache"))
    except Exception:
        pass
    data = yf.download(symbols, start=start, end=end, auto_adjust=False, actions=False,
                       group_by="ticker", progress=False, threads=True)
    return parse_yahoo_frame(data, symbols)


def parse_yahoo_frame(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """yfinance with group_by='ticker' returns columns (ticker, field) on a date index."""
    frames = []
    have = set(data.columns.get_level_values(0)) if isinstance(data.columns, pd.MultiIndex) else set()
    for s in symbols:
        if s not in have:
            log.warning("yahoo: no data for %s", s)
            continue
        d = data[s].dropna(how="all")
        if d.empty:
            log.warning("yahoo: no data for %s", s)
            continue
        frames.append(pd.DataFrame({
            "source": "yahoo", "symbol": s,
            "date": pd.to_datetime(d.index).strftime("%Y-%m-%d"),
            "open": d["Open"].to_numpy(), "high": d["High"].to_numpy(), "low": d["Low"].to_numpy(),
            "close": d["Close"].to_numpy(), "volume": d["Volume"].to_numpy(),
        }))
    if not frames:
        raise RuntimeError("Yahoo returned no data for any symbol")
    return pd.concat(frames, ignore_index=True)[PRICE_COLS]
