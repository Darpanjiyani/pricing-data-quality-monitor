"""Score the checks: plant labelled errors, run the checks, count what was caught.

Method
  1. Run every check on the data as loaded (baseline). On real data these flags may be
     real problems or real market events; they are reported separately.
  2. Plant n errors of each type (pdq.inject) and run the checks again.
  3. A planted error is *caught* when any warn/error issue lands on its (dataset, date, key).
     It is *classified correctly* when one of those issues comes from the check meant for it.
  4. A *false alarm* is a new flag (not in the baseline) that matches no planted error.
"""

from __future__ import annotations

import pandas as pd

from .checks import run_all
from .db import connect, replace_source
from .inject import EXPECTED_CHECKS, inject


def _keys(issues: pd.DataFrame) -> set:
    act = issues[issues.severity != "info"]
    return set(zip(act.dataset, act.date, act.key))


def evaluate(prices, prices_alt, yields, yields_alt, cfg, n_per_type=10, seed=42):
    def run(p, pa, y, ya):
        conn = connect()
        for t, df in (("prices", p), ("prices", pa), ("yields", y), ("yields", ya)):
            if df is not None:
                replace_source(conn, t, df)
        e = (p.source.iloc[0], None if pa is None else pa.source.iloc[0])
        r = (y.source.iloc[0], None if ya is None else ya.source.iloc[0])
        return run_all(conn, cfg, e, r)

    baseline = run(prices, prices_alt, yields, yields_alt)
    p, pa, y, ya, labels = inject(prices, prices_alt, yields, yields_alt, n_per_type, seed)
    after = run(p, pa, y, ya)

    flagged = _keys(after)
    new = flagged - _keys(baseline)
    by_key = after[after.severity != "info"].groupby(["dataset", "date", "key"])["check"].agg(set)

    labels["caught"] = [k in flagged for k in zip(labels.dataset, labels.date, labels.key)]
    labels["classified"] = [
        bool(by_key.get(k, set()) & EXPECTED_CHECKS[et])
        for et, k in zip(labels.error_type, zip(labels.dataset, labels.date, labels.key))
    ]
    label_keys = set(zip(labels.dataset, labels.date, labels.key))
    false_alarms = new - label_keys
    true_new = new & label_keys

    per_type = (labels.groupby(["dataset", "error_type"])
                .agg(planted=("caught", "size"), caught=("caught", "sum"), classified=("classified", "sum"))
                .reset_index())
    per_type["recall"] = per_type.caught / per_type.planted

    n_rows = len(prices) + len(yields.dropna(subset=["yield"]))
    summary = {
        "planted": len(labels),
        "caught": int(labels.caught.sum()),
        "recall": labels.caught.mean(),
        "classified_correctly": labels.classified.mean(),
        "new_flags": len(new),
        "false_alarms": len(false_alarms),
        "precision": len(true_new) / len(new) if new else float("nan"),
        "baseline_flags": len(_keys(baseline)),
        "rows_checked": n_rows,
        "baseline_flags_per_10k_rows": 1e4 * len(_keys(baseline)) / n_rows,
    }
    fa = after[[k in false_alarms for k in zip(after.dataset, after.date, after.key)]]
    missed = labels[~labels.caught]
    return summary, per_type, labels, fa, missed, baseline
