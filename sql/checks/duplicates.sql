-- The same symbol and date loaded more than once.
SELECT 'duplicate_row' AS "check", 'equity' AS dataset, date, symbol AS key, 'error' AS severity,
       COUNT(*) AS value,
       'row appears ' || COUNT(*) || ' times'
         || CASE WHEN MIN(close) <> MAX(close) THEN ' with different closes' ELSE '' END AS detail
FROM prices
WHERE source = :source
GROUP BY symbol, date
HAVING COUNT(*) > 1;
