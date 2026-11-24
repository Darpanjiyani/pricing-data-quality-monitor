-- Gaps-and-islands: runs of identical closes. A run of 1 + :min_repeats days or more is stale;
-- every day after the first is flagged (the first one is the last real print).
WITH p AS (
  SELECT symbol, date, MAX(close) AS close
  FROM prices WHERE source = :source AND close IS NOT NULL
  GROUP BY symbol, date
), f AS (
  SELECT symbol, date, close,
         CASE WHEN close = LAG(close) OVER (PARTITION BY symbol ORDER BY date) THEN 0 ELSE 1 END AS new_run
  FROM p
), g AS (
  SELECT *, SUM(new_run) OVER (PARTITION BY symbol ORDER BY date ROWS UNBOUNDED PRECEDING) AS run_id
  FROM f
), r AS (
  SELECT symbol, run_id, COUNT(*) AS n, MIN(date) AS start_date
  FROM g GROUP BY symbol, run_id
  HAVING COUNT(*) - 1 >= :min_repeats
)
SELECT 'stale_price' AS "check", 'equity' AS dataset, g.date, g.symbol AS key, 'warn' AS severity,
       g.close AS value,
       'close ' || g.close || ' unchanged since ' || r.start_date || ' (' || (r.n - 1) || ' repeats)' AS detail
FROM g JOIN r USING (symbol, run_id)
WHERE g.date > r.start_date;
