"""Daily HTML report: summary, charts and issue tables in one self-contained file."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill): categorical slots in fixed order, recessive ink and grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEV_ORDER = ["error", "warn", "info"]


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_issues_per_week(issues: pd.DataFrame) -> str | None:
    act = issues[issues.severity != "info"].copy()
    if act.empty:
        return None
    act["week"] = pd.to_datetime(act.date).dt.to_period("W").dt.start_time
    wk = act.groupby(["week", "dataset"]).size().unstack(fill_value=0)
    wk = wk.reindex(columns=[c for c in ("equity", "rates") if c in wk.columns])
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURFACE)
    bottom = None
    for col, color in zip(wk.columns, (BLUE, ORANGE)):
        ax.bar(wk.index, wk[col], width=5, bottom=bottom, color=color, label=col,
               edgecolor=SURFACE, linewidth=1)
        bottom = wk[col] if bottom is None else bottom + wk[col]
    _style(ax, "Flags per week (errors and warnings)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    return _png(fig)


def chart_2s10s(yields: pd.DataFrame) -> str | None:
    w = yields.pivot_table(index="date", columns="tenor", values="yield")
    if not {"2Y", "10Y"} <= set(w.columns):
        return None
    s = ((w["10Y"] - w["2Y"]) * 100).dropna()
    s.index = pd.to_datetime(s.index)
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=SURFACE)
    ax.plot(s.index, s.values, color=BLUE, linewidth=2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.annotate(f"{s.iloc[-1]:+.0f} bp", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
    _style(ax, "10Y minus 2Y Treasury spread (bp), error-flagged points removed; below zero = inverted")
    return _png(fig)


def chart_latest_curve(yields: pd.DataFrame) -> str | None:
    w = yields.pivot_table(index="date", columns="tenor", values="yield")
    months = yields.groupby("tenor")["tenor_months"].first().sort_values()
    w = w.reindex(columns=months.index).dropna(how="all")
    if len(w) < 2:
        return None
    last, prev = w.index[-1], w.index[max(0, len(w) - 22)]
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURFACE)
    for d, color in ((prev, ORANGE), (last, BLUE)):
        row = w.loc[d].dropna()
        ax.plot(range(len(row)), row.values, color=color, linewidth=2, marker="o", markersize=4, label=d)
    ax.set_xticks(range(len(months)), months.index)
    _style(ax, "Treasury par yield curve (%): latest vs about one month earlier (error-flagged points removed)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    return _png(fig)


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p class='muted'>None.</p>"
    more = f"<p class='muted'>Showing {max_rows} of {len(df)} rows; all rows are in the CSV.</p>" if len(df) > max_rows else ""
    return df.head(max_rows).to_html(index=False, escape=True, border=0, classes="t", na_rep="") + more


def write_report(path, issues: pd.DataFrame, meta: dict, yields: pd.DataFrame | None = None,
                 evaluation: tuple | None = None) -> None:
    latest = issues.date.max() if not issues.empty else None
    counts = issues.groupby(["check", "dataset", "severity"]).size().rename("flags").reset_index()
    counts = (counts.assign(_s=counts.severity.map(SEV_ORDER.index), _f=-counts["flags"])
              .sort_values(["_s", "_f"]).drop(columns=["_s", "_f"]))
    sev = issues.severity.value_counts().reindex(SEV_ORDER, fill_value=0)
    last_n = issues[issues.date >= sorted(issues.date.unique())[-5]] if latest else issues
    charts = [chart_issues_per_week(issues)]
    if yields is not None and not yields.empty:
        # chart the data with error-flagged points removed, so one bad print does not flatten the axis
        bad = issues[(issues.dataset == "rates") & (issues.severity == "error")]
        bad = set(zip(bad.date, bad.key))
        clean = yields[[k not in bad for k in zip(yields.date, yields.tenor)]]
        charts += [chart_latest_curve(clean), chart_2s10s(clean)]
    imgs = "".join(f"<img src='data:image/png;base64,{c}' alt='chart'>" for c in charts if c)

    ev_html = ""
    if evaluation:
        summary, per_type, _labels, fa, missed, _base = evaluation
        ev_html = f"""
<h2>Detection test (planted errors)</h2>
<p>{summary['planted']} known errors were planted in a copy of the data. The checks caught
<b>{summary['caught']}</b> (recall <b>{summary['recall']:.1%}</b>); {summary['classified_correctly']:.1%} were
flagged by the check meant for that error type. Of {summary['new_flags']} new flags,
{summary['false_alarms']} matched no planted error (precision <b>{summary['precision']:.1%}</b>).</p>
{_table(per_type.assign(recall=per_type.recall.map('{:.0%}'.format)))}
<h3>Missed errors</h3>{_table(missed)}
<h3>False alarms</h3>{_table(fa)}"""

    meta_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in meta.items())
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Pricing data quality report {latest or ''}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:{INK};
       background:#f9f9f7; max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }} h2 {{ font-size: 17px; margin-top: 32px; }}
.muted {{ color:{INK2}; font-size: 13px; }}
.cards {{ display:flex; gap:12px; margin:16px 0; }}
.card {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:8px; padding:12px 16px; min-width:120px; }}
.card .n {{ font-size: 26px; font-weight: 700; }} .card .l {{ color:{INK2}; font-size: 12px; }}
table.t, table.meta {{ border-collapse: collapse; font-size: 12.5px; background:{SURFACE}; }}
table.t th, table.t td, table.meta th, table.meta td {{ padding: 4px 10px; border-bottom: 1px solid {GRID}; text-align:left; vertical-align: top; }}
table.t th {{ color:{INK2}; font-weight:600; }}
img {{ max-width: 100%; display:block; margin: 12px 0; border:1px solid {GRID}; border-radius:8px; }}
</style></head><body>
<h1>Pricing data quality report</h1>
<p class="muted">Generated {datetime.now():%Y-%m-%d %H:%M}. Latest data date: {latest or 'n/a'}.</p>
<table class="meta">{meta_rows}</table>
<div class="cards">
  <div class="card"><div class="n">{sev['error']}</div><div class="l">errors</div></div>
  <div class="card"><div class="n">{sev['warn']}</div><div class="l">warnings</div></div>
  <div class="card"><div class="n">{sev['info']}</div><div class="l">info</div></div>
  <div class="card"><div class="n">{len(issues[issues.date == latest]) if latest else 0}</div><div class="l">flags on {latest or '-'}</div></div>
</div>
{imgs}
<h2>Flags by check</h2>{_table(counts)}
<h2>Last 5 data dates</h2>{_table(last_n.sort_values(['date', 'severity'], ascending=[False, True]))}
{ev_html}
<h2>All errors</h2>{_table(issues[issues.severity == 'error'].sort_values('date', ascending=False))}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
