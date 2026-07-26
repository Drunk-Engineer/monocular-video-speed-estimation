import numpy as np

from monocular_speed.labels import clean_speed_series


def test_clean_speed_series_fills_invalid_values_and_limits_steps() -> None:
    cleaned = clean_speed_series([10.0, np.nan, 500.0, 14.0], max_step=1.0)
    assert np.isfinite(cleaned).all()
    assert cleaned.tolist() == [10.0, 11.0, 12.0, 13.0]


def test_clean_speed_series_respects_declared_range() -> None:
    cleaned = clean_speed_series([-5.0, 5.0, 121.0], minimum=0.0, maximum=120.0)
    assert np.all((cleaned >= 0.0) & (cleaned <= 120.0))


def test_clean_speed_series_removes_isolated_spike_bidirectionally() -> None:
    cleaned = clean_speed_series([10.0, 80.0, 10.0], max_step=1.0)
    assert cleaned.tolist() == [10.0, 10.0, 10.0]
