from pdq.evaluate import evaluate


def test_detection_on_sample(cfg, sample):
    summary, per_type, labels, fa, missed, _ = evaluate(*sample, cfg, n_per_type=5, seed=3)
    assert summary["recall"] >= 0.95
    assert summary["precision"] >= 0.85
    assert set(per_type.error_type) == {
        "stale_run", "missing_day", "duplicate_row", "bad_print", "unadjusted_split", "invalid_ohlc",
        "cross_source_bad_value", "yield_spike", "decimal_slip", "missing_tenor", "cross_source_rate"}
