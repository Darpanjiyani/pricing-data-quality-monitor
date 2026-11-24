-- Same tenor and date from two publishers. FRED republishes Treasury's numbers,
-- so anything above a couple of basis points is a load or source problem.
SELECT 'cross_source_yield' AS "check", 'rates' AS dataset, a.date, a.tenor AS key, 'warn' AS severity,
       ROUND((a.yield - b.yield) * 100, 2) AS value,
       :primary || ' ' || a.yield || ' vs ' || :secondary || ' ' || b.yield
         || ' (' || ROUND((a.yield - b.yield) * 100, 1) || ' bp)' AS detail
FROM yields a
JOIN yields b ON b.date = a.date AND b.tenor = a.tenor AND b.source = :secondary
WHERE a.source = :primary
  AND a.yield IS NOT NULL AND b.yield IS NOT NULL
  AND ABS(a.yield - b.yield) * 100 > :xsrc_tol_bp;
