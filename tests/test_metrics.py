import numpy as np
import pytest

from monocular_speed.metrics import regression_metrics


def test_regression_metrics_use_prediction_minus_reference_bias() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.array([1.0, 1.0, 2.0, 2.0])
    result = regression_metrics(reference, prediction)

    assert result["mae_kmh"] == pytest.approx(0.5)
    assert result["rmse_kmh"] == pytest.approx(np.sqrt(0.5))
    assert result["bias_kmh"] == pytest.approx(0.0)
    assert result["n_frames"] == 4


def test_regression_metrics_ignore_non_finite_pairs() -> None:
    result = regression_metrics(
        np.array([1.0, np.nan, 3.0]),
        np.array([2.0, 2.0, 3.0]),
    )
    assert result["n_frames"] == 2
    assert result["mae_kmh"] == pytest.approx(0.5)
