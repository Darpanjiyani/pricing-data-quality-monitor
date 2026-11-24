-- Rows that cannot be right: missing or non-positive prices, high below low,
-- open/close outside the day's high-low range, negative volume.
-- A tiny tolerance absorbs vendor rounding (close printed a hair above the high).
SELECT 'price_validity' AS "check", 'equity' AS dataset, date, symbol AS key, 'error' AS severity,
       close AS value,
       CASE
         WHEN close IS NULL THEN 'close is missing'
         WHEN close <= 0 OR open <= 0 OR high <= 0 OR low <= 0 THEN 'non-positive price'
         WHEN high < low THEN 'high ' || high || ' below low ' || low
         WHEN close > high * (1 + :eps) OR close < low * (1 - :eps) THEN 'close ' || close || ' outside high-low range'
         WHEN open  > high * (1 + :eps) OR open  < low * (1 - :eps) THEN 'open ' || open || ' outside high-low range'
         ELSE 'negative volume'
       END AS detail
FROM prices
WHERE source = :source
  AND (close IS NULL OR close <= 0 OR open <= 0 OR high <= 0 OR low <= 0
       OR high < low
       OR close > high * (1 + :eps) OR close < low * (1 - :eps)
       OR open  > high * (1 + :eps) OR open  < low * (1 - :eps)
       OR volume < 0);
