"""Command line: pdq demo | fetch | check | evaluate | daily

  pdq demo                 synthetic data end to end (no internet needed)
  pdq fetch                download real data into the database
  pdq check                run every check on the database, write the report and CSV
  pdq evaluate             plant known errors in a copy of the database data and score the checks
  pdq daily                fetch + check (what the scheduled GitHub Action runs)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import fetch
from .checks import run_all
from .config import load_config
from .db import connect, read_table, replace_source, save_issues, sources
from .evaluate import evaluate
from .inject import inject
from .report import write_report
from .sample import sample_curve, sample_prices

log = logging.getLogger("pdq")
PREFER = {
    ("prices", 0): ["yahoo", "sample"], ("prices", 1): ["sample_alt"],
    ("yields", 0): ["treasury", "treasury_sample"], ("yields", 1): ["fred", "fred_sample"],
}


def pick_sources(conn):
    out = {}
    for (table, slot), names in PREFER.items():
        have = sources(conn, table)
        out[(table, slot)] = next((n for n in names if n in have), None)
    return (out[("prices", 0)], out[("prices", 1)]), (out[("yields", 0)], out[("yields", 1)])


def _write_eval(out: Path, ev) -> None:
    summary, per_type, labels, fa, missed, _ = ev
    pd.DataFrame([summary]).to_csv(out / "evaluation_summary.csv", index=False)
    per_type.to_csv(out / "evaluation_by_type.csv", index=False)
    labels.to_csv(out / "evaluation_labels.csv", index=False)
    fa.to_csv(out / "evaluation_false_alarms.csv", index=False)


def cmd_fetch(args, cfg):
    conn = connect(args.db)
    start = args.start or cfg["universe"]["start"]
    syms = cfg["universe"]["symbols"]
    jobs = [("yields", "Treasury", lambda: fetch.treasury_curve(start)),
            ("yields", "FRED", lambda: fetch.fred_curve(start)),
            ("prices", "Yahoo", lambda: fetch.yahoo_prices(syms, start))]
    for table, name, job in jobs:
        try:
            df = job()
            replace_source(conn, table, df)
            log.info("%s: %d rows", name, len(df))
        except Exception as e:  # keep going: one source down should not stop the rest
            log.warning("%s failed: %s", name, e)


def cmd_check(args, cfg, evaluation=None):
    conn = connect(args.db)
    (e1, e2), (r1, r2) = pick_sources(conn)
    if not e1 and not r1:
        sys.exit("No data in the database. Run `pdq fetch` (real data) or `pdq demo` first.")
    issues = run_all(conn, cfg, (e1, e2), (r1, r2))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_issues(conn, run_id, issues)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    issues.to_csv(out / "issues.csv", index=False)
    yields = read_table(conn, "yields", r1) if r1 else None
    n_prices = conn.execute("SELECT COUNT(*) FROM prices WHERE source=?", (e1,)).fetchone()[0] if e1 else 0
    n_yields = conn.execute("SELECT COUNT(*) FROM yields WHERE source=? AND yield IS NOT NULL", (r1,)).fetchone()[0] if r1 else 0
    meta = {"Equity sources": f"{e1} (primary), {e2 or 'none'} (comparison)",
            "Rates sources": f"{r1} (primary), {r2 or 'none'} (comparison)",
            "Rows checked": f"{n_prices:,} price rows, {n_yields:,} yield points",
            "Run id": run_id}
    if e1 and e1.startswith("sample"):
        meta["Note"] = ("SYNTHETIC DATA with planted errors (see planted_errors.csv): "
                        "shows the checks work; not a result to quote")
    write_report(out / "report.html", issues, meta, yields, evaluation)
    log.info("%d flags (%s). Report: %s", len(issues),
             ", ".join(f"{k} {v}" for k, v in issues.severity.value_counts().items()), out / "report.html")
    return issues


def cmd_evaluate(args, cfg):
    conn = connect(args.db)
    (e1, e2), (r1, r2) = pick_sources(conn)
    get = lambda t, s: read_table(conn, t, s) if s else None
    ev = evaluate(get("prices", e1), get("prices", e2), get("yields", r1), get("yields", r2),
                  cfg, n_per_type=args.n, seed=args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _write_eval(out, ev)
    s = ev[0]
    log.info("planted %d, caught %d (recall %.1f%%), precision %.1f%%, false alarms %d",
             s["planted"], s["caught"], 100 * s["recall"], 100 * s["precision"], s["false_alarms"])
    print(ev[1].to_string(index=False))
    return ev


def cmd_demo(args, cfg):
    conn = connect(args.db)
    p, pa = sample_prices(cfg["universe"]["symbols"])
    y, ya = sample_curve()
    for t, df in (("prices", p), ("prices", pa), ("yields", y), ("yields", ya)):
        replace_source(conn, t, df)
    conn.close()
    ev = cmd_evaluate(args, cfg)
    # Show the report on the copy with planted errors, so it has something to find.
    p2, pa2, y2, ya2, labels = inject(p, pa, y, ya, args.n, args.seed)
    conn = connect(args.db)
    for t, df in (("prices", p2), ("prices", pa2), ("yields", y2), ("yields", ya2)):
        replace_source(conn, t, df)
    conn.close()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    labels.to_csv(Path(args.out) / "planted_errors.csv", index=False)
    cmd_check(args, cfg, evaluation=ev)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdq", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["demo", "fetch", "check", "evaluate", "daily"])
    ap.add_argument("--db", default="data/pdq.db")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default=None, help="first date to fetch (default: config universe.start)")
    ap.add_argument("--n", type=int, default=10, help="errors planted per type in evaluate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-eval", action="store_true", help="check: also run the detection test")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    if args.command == "demo":
        if args.db == "data/pdq.db":
            args.db = "data/demo.db"
        if args.out == "reports":
            args.out = "reports/demo"
        cmd_demo(args, cfg)
    elif args.command == "fetch":
        cmd_fetch(args, cfg)
    elif args.command == "check":
        cmd_check(args, cfg, cmd_evaluate(args, cfg) if args.with_eval else None)
    elif args.command == "evaluate":
        cmd_evaluate(args, cfg)
    elif args.command == "daily":
        cmd_fetch(args, cfg)
        cmd_check(args, cfg, cmd_evaluate(args, cfg) if args.with_eval else None)


if __name__ == "__main__":
    main()
