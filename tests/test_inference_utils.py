import numpy as np
import pytest
import torch

from monocular_speed.inference import triangular_weights
from monocular_speed.video import padded_window, window_starts


def test_triangular_weights_are_symmetric_and_positive() -> None:
    weights = triangular_weights(30)
    assert np.all(weights > 0)
    assert np.allclose(weights, weights[::-1])
    assert weights[14] == pytest.approx(weights[15])
    assert weights[14] > weights[0]


def test_window_starts_cover_complete_and_partial_tails() -> None:
    assert window_starts(300, window=30, stride=15)[-1] == 270
    assert window_starts(295, window=30, stride=15)[-1] == 270
    with pytest.raises(ValueError):
        window_starts(29, window=30, stride=15)


def test_padded_window_repeats_last_frame() -> None:
    values = torch.arange(35, dtype=torch.float32).view(1, 35, 1, 1)
    segment = padded_window(values, start=20, window=30)
    assert segment.shape == (1, 30, 1, 1)
    assert torch.all(segment[:, 15:] == 34)
