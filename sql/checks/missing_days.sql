-- A date is a trading day when at least :coverage of the symbols report it.
-- A symbol missing on such a day, between its first and last row, is flagged.
WITH p AS (
  SELECT DISTINCT symbol, date FROM prices WHERE source = :source
), n AS (
  SELECT COUNT(DISTINCT symbol) AS n FROM p
), d AS (
  SELECT date FROM p GROUP BY date HAVING COUNT(*) >= :coverage * (SELECT n FROM n)
), s AS (
  SELECT symbol, MIN(date) AS first_date, MAX(date) AS last_date FROM p GROUP BY symbol
)
SELECT 'missing_day' AS "check", 'equity' AS dataset, d.date, s.symbol AS key, 'error' AS severity,
       NULL AS value, 'no row on a trading day' AS detail
FROM d CROSS JOIN s
LEFT JOIN p ON p.date = d.date AND p.symbol = s.symbol
WHERE p.symbol IS NULL
  AND d.date BETWEEN s.first_date AND s.last_date;
