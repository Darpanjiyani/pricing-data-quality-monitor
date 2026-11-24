"""SQLite storage and the SQL-based checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import ISSUE_COLUMNS
from .config import SQL_DIR


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    return conn


def replace_source(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Replace every row of the sources present in df (a re-load is idempotent)."""
    if df.empty:
        return
    for src in df["source"].unique():
        conn.execute(f"DELETE FROM {table} WHERE source = ?", (src,))
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()


def read_table(conn: sqlite3.Connection, table: str, source: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table} WHERE source = ?", conn, params=(source,))


def sources(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[0] for r in conn.execute(f"SELECT DISTINCT source FROM {table} ORDER BY source")]


def run_sql_check(conn: sqlite3.Connection, name: str, params: dict) -> pd.DataFrame:
    sql = (SQL_DIR / "checks" / f"{name}.sql").read_text()
    df = pd.read_sql_query(sql, conn, params=params)
    return df[ISSUE_COLUMNS]


def save_issues(conn: sqlite3.Connection, run_id: str, issues: pd.DataFrame) -> None:
    conn.execute("DELETE FROM issues WHERE run_id = ?", (run_id,))
    if not issues.empty:
        issues.assign(run_id=run_id)[["run_id", *ISSUE_COLUMNS]].to_sql(
            "issues", conn, if_exists="append", index=False)
    conn.commit()
