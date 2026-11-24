-- No primary keys on purpose: duplicate rows are one of the problems the checks look for.
CREATE TABLE IF NOT EXISTS prices (
    source  TEXT NOT NULL,      -- 'yahoo', 'stooq', 'sample', ...
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,      -- ISO yyyy-mm-dd
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL
);
CREATE INDEX IF NOT EXISTS ix_prices ON prices (source, symbol, date);

CREATE TABLE IF NOT EXISTS yields (
    source       TEXT NOT NULL, -- 'treasury', 'fred', ...
    date         TEXT NOT NULL,
    tenor        TEXT NOT NULL, -- '1M', '10Y', ...
    tenor_months REAL NOT NULL,
    yield        REAL           -- percent; NULL = published blank
);
CREATE INDEX IF NOT EXISTS ix_yields ON yields (source, tenor, date);

CREATE TABLE IF NOT EXISTS issues (
    run_id   TEXT NOT NULL,
    "check"  TEXT NOT NULL,
    dataset  TEXT NOT NULL,     -- 'equity' or 'rates'
    date     TEXT NOT NULL,
    key      TEXT NOT NULL,     -- symbol or tenor
    severity TEXT NOT NULL,     -- 'error', 'warn', 'info'
    value    REAL,
    detail   TEXT
);
