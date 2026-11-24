"""All data quality checks. SQL checks live in sql/checks/; statistical checks are here.

Each check returns rows with the columns in pdq.ISSUE_COLUMNS:
  check, dataset ('equity' | 'rates'), date, key (symbol | tenor), severity, value, detail
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import ISSUE_COLUMNS
from .db import read_table, run_sql_check
from .stats import per_column_diff, reverts_next_day, robust_z


def _issues(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def _after_stale(close: pd.Series, k: int) -> pd.Series:
    """True when the previous k closes did not change: today's move is a catch-up after a
    stale run (already reported by stale_price), not a new event."""
    flat = (close.diff() == 0).astype(float)
    return flat.shift(1).rolling(k).sum() >= k


def _wide_close(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.dropna(subset=["close"]).groupby(["date", "symbol"])["close"].mean()
            .unstack("symbol").sort_index())


# ====================================================================== equities
def equity_return_outliers(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Market-adjusted return outliers, split into bad prints, unadjusted splits and real outliers.

    1. log return per symbol, minus that day's cross-sectional median (removes market-wide moves)
    2. robust z-score against the previous `z_window` days
    3. for each outlier:
         moved back the next day          -> bad_print_spike (error)
         price ratio matches a split ratio -> possible_unadjusted_split (error)
         otherwise                         -> return_outlier (warn, may be real news)
    """
    c = cfg["equity"]
    close = _wide_close(prices)
    close = close.where(close > 0)
    ret = per_column_diff(np.log(close))
    market = ret.median(axis=1)
    resid = ret.sub(market, axis=0)

    ratios = [r for k in c["split_ratios"] for r in (k, 1 / k)]
    rows = []
    for sym in resid.columns:
        x = resid[sym].dropna()
        if len(x) < c["z_min_periods"] + 1:
            continue
        z = robust_z(x, c["z_window"], c["z_min_periods"])
        catch_up = _after_stale(close[sym].dropna(), c["stale_min_repeats"]).reindex(x.index, fill_value=False)
        flagged = (z.abs() > c["z_threshold"]) & ~catch_up
        revert = reverts_next_day(x, c["revert_tol"])
        skip = set()
        for i, t in enumerate(x.index):
            if not flagged.iloc[i] or t in skip:
                continue
            raw = float(np.exp(ret.at[t, sym]))
            base = (f"close {close.at[t, sym]:.4g}, return {raw - 1:+.1%} vs market "
                    f"{np.expm1(market.at[t]):+.1%}, robust z {z.iloc[i]:.1f}")
            if revert.iloc[i]:
                check, sev, why = "bad_print_spike", "error", "reverted next day"
                if i + 1 < len(x):
                    skip.add(x.index[i + 1])
            elif any(abs(raw / r - 1) <= c["split_tol"] for r in ratios):
                ratio = min(ratios, key=lambda r: abs(raw / r - 1))
                label = f"{ratio:g}-for-1" if ratio >= 1 else f"1-for-{1 / ratio:g}"
                check, sev, why = "possible_unadjusted_split", "error", f"ratio matches a {label} split, no reversal"
            else:
                check, sev, why = "return_outlier", "warn", "no reversal; check for news"
            rows.append(dict(check=check, dataset="equity", date=t, key=sym, severity=sev,
                             value=round(float(z.iloc[i]), 2), detail=f"{base}; {why}"))
    return _issues(rows)


def equity_cross_source(primary: pd.DataFrame, secondary: pd.DataFrame, cfg: dict,
                        names: tuple[str, str]) -> pd.DataFrame:
    """Compare two vendors through the ratio of their closes.

    Vendors adjust for dividends differently, so levels drift apart slowly and a level
    comparison would flag everything. The day-over-day change in the ratio is near zero
    unless one vendor has a bad value (ratio jumps and comes back) or adjusted a split
    or corporate action the other did not (ratio steps and stays).
    """
    tol = cfg["equity"]["xsrc_ratio_tol"]
    a, b = _wide_close(primary), _wide_close(secondary)
    common = a.columns.intersection(b.columns)
    rows = []
    for sym in common:
        pair = pd.concat([a[sym], b[sym]], axis=1, keys=["a", "b"]).dropna()
        if len(pair) < 3:
            continue
        lr = np.log(pair.a / pair.b)
        d = lr.diff().dropna()
        k = cfg["equity"]["stale_min_repeats"]
        catch_up = (_after_stale(pair.a, k) | _after_stale(pair.b, k)).reindex(d.index, fill_value=False)
        flagged = (np.expm1(d.abs()) > tol) & ~catch_up
        revert = reverts_next_day(d, 0.35)
        skip = set()
        for i, t in enumerate(d.index):
            if not flagged.iloc[i] or t in skip:
                continue
            if revert.iloc[i]:
                check = "cross_source_price"
                why = "one-day disagreement"
                if i + 1 < len(d):
                    skip.add(d.index[i + 1])
            else:
                check = "cross_source_level_shift"
                why = "ratio stepped and stayed (adjustment applied by one vendor only?)"
            rows.append(dict(check=check, dataset="equity", date=t, key=sym, severity="warn",
                             value=round(float(np.expm1(d.iloc[i])), 4),
                             detail=f"{names[0]} {pair.a[t]:.4g} vs {names[1]} {pair.b[t]:.4g}; "
                                    f"ratio moved {np.expm1(d.iloc[i]):+.1%}; {why}"))
    return _issues(rows)


# ====================================================================== rates
def _wide_yields(yields: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    months = yields.groupby("tenor")["tenor_months"].first().sort_values()
    wide = (yields.groupby(["date", "tenor"])["yield"].mean().unstack("tenor")
            .reindex(columns=months.index).sort_index())
    return wide, months


def missing_tenors(yields: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A tenor left blank on a date when it is normally published (inside its active span)."""
    wide, _ = _wide_yields(yields)
    rows = []
    for tenor in wide.columns:
        s = wide[tenor]
        if s.notna().sum() == 0:
            continue
        span = s.loc[s.first_valid_index(): s.last_valid_index()]
        if span.notna().mean() < cfg["rates"]["tenor_coverage"]:
            continue
        for t in span.index[span.isna()]:
            rows.append(dict(check="missing_tenor", dataset="rates", date=t, key=tenor,
                             severity="error", value=None, detail="tenor blank on a publication date"))
    return _issues(rows)


def yield_jumps(yields: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One-day yield moves that are too large in basis points or in robust z terms."""
    c = cfg["rates"]
    wide, _ = _wide_yields(yields)
    chg = per_column_diff(wide) * 100  # basis points
    rows = []
    for tenor in chg.columns:
        x = chg[tenor].dropna()
        if len(x) < c["z_min_periods"] + 1:
            continue
        z = robust_z(x, c["z_window"], c["z_min_periods"], floor=0.5)
        flagged = (x.abs() > c["jump_abs_bp"]) | (z.abs() > c["jump_z"])
        revert = reverts_next_day(x, 0.35)
        skip = set()
        for i, t in enumerate(x.index):
            if not flagged.iloc[i] or t in skip:
                continue
            if revert.iloc[i]:
                check, sev, why = "yield_spike", "error", "reverted next day"
                if i + 1 < len(x):
                    skip.add(x.index[i + 1])
            else:
                check, sev, why = "yield_jump", "warn", "no reversal; check for market news"
            rows.append(dict(check=check, dataset="rates", date=t, key=tenor, severity=sev,
                             value=round(float(x.iloc[i]), 1),
                             detail=f"{wide.at[t, tenor]:.2f}%, moved {x.iloc[i]:+.0f} bp, "
                                    f"robust z {z.iloc[i]:.1f}; {why}"))
    return _issues(rows)


def curve_kinks(yields: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A tenor out of line with its neighbours, compared with how far out of line it usually is.

    Real curves have permanent bends (for example around the 1M-3M bills), so the check
    compares today's gap from the straight line between neighbours with the tenor's own
    recent median gap. A bad value also bends its neighbours' lines by about half as much,
    so only the tenor with the largest excess in a run of flagged neighbours is reported.
    """
    c = cfg["rates"]
    wide, months = _wide_yields(yields)
    tenors = list(wide.columns)
    m = months.to_numpy(dtype=float)
    dev = pd.DataFrame(index=wide.index, columns=tenors, dtype=float)
    for i in range(1, len(tenors) - 1):
        left, mid, right = wide[tenors[i - 1]], wide[tenors[i]], wide[tenors[i + 1]]
        w = (m[i] - m[i - 1]) / (m[i + 1] - m[i - 1])
        dev[tenors[i]] = (mid - (left + w * (right - left))) * 100
    excess = dev - dev.shift(1).rolling(c["kink_window"], min_periods=c["z_min_periods"]).median()
    jumped = set(zip(*[yield_jumps(yields, cfg)[col] for col in ("date", "key")]))
    flagged = excess.abs() > c["kink_bp"]
    rows = []
    for t in wide.index:
        e = excess.loc[t].abs()
        for i in range(1, len(tenors) - 1):
            tenor = tenors[i]
            if not flagged.at[t, tenor]:
                continue
            nb = [e.iloc[j] for j in (i - 1, i + 1) if 0 < j < len(tenors) - 1 and not np.isnan(e.iloc[j])]
            if any(v > e.iloc[i] for v in nb):
                continue
            # a neighbour with its own jump today (often an end tenor, which has no kink test)
            # is the likelier culprit; yield_jumps reports it
            if (t, tenor) not in jumped and ((t, tenors[i - 1]) in jumped or (t, tenors[i + 1]) in jumped):
                continue
            rows.append(dict(check="curve_kink", dataset="rates", date=t, key=tenor, severity="warn",
                             value=round(float(excess.at[t, tenor]), 1),
                             detail=f"{wide.at[t, tenor]:.2f}% sits {dev.at[t, tenor]:+.0f} bp off the line "
                                    f"between {tenors[i - 1]} and {tenors[i + 1]} "
                                    f"({excess.at[t, tenor]:+.0f} bp more than usual)"))
    return _issues(rows)


def inversion_changes(yields: pd.DataFrame) -> pd.DataFrame:
    """Info only: days the 2Y-10Y spread changes sign. Inversions are real, not errors."""
    wide, _ = _wide_yields(yields)
    if not {"2Y", "10Y"} <= set(wide.columns):
        return _issues([])
    spread = ((wide["10Y"] - wide["2Y"]) * 100).dropna()
    sign = np.sign(spread).replace(0, np.nan).ffill()
    # a flip counts only when the new sign holds for the next 5 dates (ignores noise around zero)
    held = sign.rolling(5).apply(lambda w: float(len(set(w)) == 1), raw=True).shift(-4) == 1
    flips = spread[(sign != sign.shift(1)) & held].iloc[1:]
    rows = [dict(check="curve_inversion_change", dataset="rates", date=t, key="2s10s", severity="info",
                 value=round(float(v), 1),
                 detail=("2s10s turned negative (inverted)" if v < 0 else "2s10s turned positive")
                        + f" at {v:+.0f} bp") for t, v in flips.items()]
    return _issues(rows)


# ====================================================================== runner
def run_all(conn: sqlite3.Connection, cfg: dict, equity_sources: tuple[str, str | None],
            rate_sources: tuple[str, str | None]) -> pd.DataFrame:
    e1, e2 = equity_sources
    r1, r2 = rate_sources
    out = []
    if e1:
        ec = cfg["equity"]
        out += [
            run_sql_check(conn, "price_validity", {"source": e1, "eps": 1e-4}),
            run_sql_check(conn, "duplicates", {"source": e1}),
            run_sql_check(conn, "stale_prices", {"source": e1, "min_repeats": ec["stale_min_repeats"]}),
            run_sql_check(conn, "missing_days", {"source": e1, "coverage": ec["coverage"]}),
        ]
        p1 = read_table(conn, "prices", e1)
        out.append(equity_return_outliers(p1, cfg))
        if e2:
            out.append(equity_cross_source(p1, read_table(conn, "prices", e2), cfg, (e1, e2)))
    if r1:
        rc = cfg["rates"]
        out.append(run_sql_check(conn, "yield_range",
                                 {"source": r1, "min_yield": rc["min_yield"], "max_yield": rc["max_yield"]}))
        y1 = read_table(conn, "yields", r1)
        out += [missing_tenors(y1, cfg), yield_jumps(y1, cfg), curve_kinks(y1, cfg), inversion_changes(y1)]
        if r2:
            out.append(run_sql_check(conn, "yield_cross_source",
                                     {"primary": r1, "secondary": r2, "xsrc_tol_bp": rc["xsrc_tol_bp"]}))
    out = [o.assign(value=pd.to_numeric(o["value"], errors="coerce")) for o in out if not o.empty]
    if not out:
        return _issues([])
    issues = pd.concat(out, ignore_index=True)
    return issues.sort_values(["date", "dataset", "key", "check"]).reset_index(drop=True)
