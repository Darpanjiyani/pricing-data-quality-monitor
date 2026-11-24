# Pricing Data Quality Monitor

Daily data quality checks for **end-of-day equity prices** and the **U.S. Treasury par yield curve**, with a detection test that plants known errors and measures how many the checks catch.

Each day it:
- loads prices for 30 large-cap stocks
- loads the Treasury curve from two publishers (Treasury and FRED) and compares them
- runs 16 checks (SQL and statistical)
- writes an HTML report and a CSV of every flag

The detection test then reports **recall, precision and false alarms per error type**.

```
pdq fetch     ->  SQLite (prices, yields)  ->  pdq check  ->  reports/report.html + issues.csv
                                           ->  pdq evaluate (plant errors, score the checks)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[live,dev]"

pdq demo                 # synthetic data, no internet needed: reports/demo/report.html
pytest -q                # 16 tests

pdq fetch                # real data from 2023-01-01 (change in config.toml)
pdq check --with-eval    # checks + detection test on the real data: reports/report.html
```

`pdq daily` runs fetch and check together. The GitHub Action in `.github/workflows/daily.yml` runs it every weekday after the close and uploads the report as a build artifact.

## Data sources (all free, no account)

| Data | Primary | Comparison source |
|---|---|---|
| Equities | Yahoo Finance via `yfinance` (split-adjusted, not dividend-adjusted) | none on live data (see below) |
| Treasury curve | U.S. Treasury daily par yield curve CSV, 14 tenors 1M–30Y | FRED constant-maturity series DGS1MO–DGS30, 11 tenors |

The equity cross-source checks (`cross_source_price`, `cross_source_level_shift`) need a second price vendor. The free ones tried, Stooq and Nasdaq.com, block scripted downloads, so these two checks run only on the synthetic demo data. Adding a keyed vendor later (Tiingo, Alpha Vantage) means one loader in `fetch.py` that returns the same columns.

## The checks

| Check | Data | How it works | Severity |
|---|---|---|---|
| `price_validity` | equity | SQL: missing or non-positive prices, high < low, open/close outside the high–low range, negative volume | error |
| `duplicate_row` | equity | SQL: same symbol and date loaded twice | error |
| `stale_price` | equity | SQL gaps-and-islands with window functions: the same close on 3+ days in a row after the first | warn |
| `missing_day` | equity | SQL: no row on a date when at least 80% of symbols report, between the symbol's first and last row | error |
| `bad_print_spike` | equity | Market-adjusted log return with a robust z-score (median/MAD of the previous 60 days) above 6, **and** the next day reverses it | error |
| `possible_unadjusted_split` | equity | Outlier whose price ratio is within 3% of 2:1, 3:1, 4:1, 5:1, 10:1 or 3:2 (either way) and does not reverse | error |
| `return_outlier` | equity | Any other outlier: may be real news, so it needs a person to look | warn |
| `cross_source_price` | equity | Day-over-day change in the ratio of the two vendors' closes above 2%, and the ratio comes back the next day | warn |
| `cross_source_level_shift` | equity | Same, but the ratio stays: one vendor applied an adjustment the other did not | warn |
| `yield_out_of_range` | rates | SQL: yield outside −1% to 25% (catches 4.5 loaded as 45) | error |
| `missing_tenor` | rates | A tenor blank on a date where it is normally published | error |
| `yield_spike` | rates | One-day move over 40 bp or robust z over 6, reversed the next day | error |
| `yield_jump` | rates | Same, not reversed: probably a real market move | warn |
| `curve_kink` | rates | Tenor's gap from the straight line between its neighbours, compared with its usual gap; over 20 bp more than usual | warn |
| `cross_source_yield` | rates | SQL: Treasury vs FRED for the same tenor and date differ by more than 2 bp | warn |
| `curve_inversion_change` | rates | 10Y–2Y spread changes sign and holds for 5 days | info |

All thresholds live in `config.toml`.

### Design choices worth knowing (interview prep)

- **Robust z-scores, using past data only.** The median and MAD come from the previous 60 days, not including today. A bad value cannot widen its own yardstick, and one old spike does not blunt later ones the way it would with mean and standard deviation.
- **Market-adjusted returns.** Each stock's return minus that day's cross-sectional median. A day when everything falls 8% is a market event, not 30 data errors (there is a test for exactly this).
- **Spike-then-revert logic.** A price printed 10× too high shows up as +230% then −230% in log terms. Treating that as one bad print, and not flagging the day after, keeps the reversal from being counted as a second error.
- **Catch-up suppression.** After a stale run ends, the price jumps to where it should be. That jump is already explained by the `stale_price` flag, so it is not raised again as an outlier.
- **Ratio, not level, for vendor comparison.** Vendors adjust for dividends differently, so their levels drift apart slowly. The day-over-day change in their ratio is near zero unless something is wrong.
- **Kink attribution.** A bad 5Y value also bends the lines used to test 3Y and 7Y. The check reports only the tenor with the largest excess, and does not blame an interior tenor when its end-tenor neighbour (1M or 30Y) has its own jump.
- **Inversions are not errors.** A negative 2s10s spread is real (it stayed negative from mid-2022 to late 2024). It is reported as info when it changes sign, never as a data problem.

## The detection test

`pdq evaluate` (or `pdq check --with-eval`):

1. Runs every check on the data as loaded. This is the **baseline**: on real data, these flags are real problems or real market events.
2. Plants errors in a copy, 10 of each type by default. It stores a label for every planted error in `evaluation_labels.csv`. The planted types:
   - equity: stale runs (4 repeated bars), dropped rows, duplicates with a 1% different close, bad prints (×10, ×0.1, ×1.25, ×0.8), unadjusted 2/3/4-for-1 splits, swapped high/low, a 5% one-day error in the comparison vendor
   - rates: 25–75 bp spikes, ×10 decimal slips, blank tenors, a 5 bp error in the FRED copy
3. Re-runs the checks and scores them:
   - **caught:** any error or warning lands on the planted (dataset, date, key)
   - **classified:** the flag came from the check meant for that error type
   - **false alarm:** a new flag (not in the baseline) that matches no planted error

On the synthetic demo data the checks catch 140/140 planted errors with about 98% precision. **That number is from made-up data and is not for quoting.** Synthetic errors are easier to see than real ones. Use the numbers from a run on real data.

## Project layout

```
config.toml              thresholds and ticker list
sql/schema.sql           SQLite tables: prices, yields, issues
sql/checks/*.sql         the SQL checks (window functions, gaps-and-islands, cross joins)
src/pdq/fetch.py         Treasury, FRED and Yahoo loaders
src/pdq/checks.py        statistical checks + the runner that calls the SQL ones
src/pdq/stats.py         robust z-score, spike-then-revert helper
src/pdq/inject.py        error planting with labels
src/pdq/evaluate.py      recall / precision / false alarms
src/pdq/report.py        HTML report with charts
src/pdq/sample.py        synthetic data (Nelson-Siegel curve, fat-tailed returns)
tests/                   pytest suite
```

## Ideas for next steps

- Add more asset classes:
  - corporate bond prices
  - futures settlements
  - the daily FX rates that FRED publishes
- Add a reference-data check: ticker changes and delistings against an exchange symbol list.
- Track flag counts over time and alert when a day's count breaks from its own history (monitoring the monitor).
