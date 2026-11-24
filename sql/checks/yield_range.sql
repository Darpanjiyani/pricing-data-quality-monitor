-- Yields outside a plausible band (catches decimal slips such as 4.5 loaded as 45).
SELECT 'yield_out_of_range' AS "check", 'rates' AS dataset, date, tenor AS key, 'error' AS severity,
       yield AS value,
       'yield ' || yield || '% outside [' || :min_yield || ', ' || :max_yield || ']' AS detail
FROM yields
WHERE source = :source AND yield IS NOT NULL
  AND (yield < :min_yield OR yield > :max_yield);
