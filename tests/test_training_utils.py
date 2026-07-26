import pandas as pd
import pytest
import torch

from monocular_speed.dataset import ClipDataset, balanced_speed_sampler
from monocular_speed.training import build_two_sided_weights, std_floor_regularizer


def test_two_sided_weights_emphasize_low_and_high_speeds_after_warmup() -> None:
    labels = torch.tensor([5.0, 25.0, 45.0, 80.0])
    cold = build_two_sided_weights(
        labels,
        high_threshold_kmh=45.0,
        high_factor=1.8,
        high_tau_kmh=5.0,
        low_center_kmh=5.0,
        low_factor=2.5,
        low_tau_kmh=5.0,
        warmup_alpha=0.0,
    )
    warm = build_two_sided_weights(
        labels,
        high_threshold_kmh=45.0,
        high_factor=1.8,
        high_tau_kmh=5.0,
        low_center_kmh=5.0,
        low_factor=2.5,
        low_tau_kmh=5.0,
        warmup_alpha=1.0,
    )
    assert torch.allclose(cold, torch.ones_like(cold))
    assert warm[0] > warm[1]
    assert warm[-1] > warm[1]


def test_std_floor_penalizes_constant_predictions() -> None:
    labels = torch.tensor([[0.1, 0.2, 0.3]])
    constant = torch.full_like(labels, 0.5)
    varying = torch.tensor([[0.4, 0.5, 0.6]])
    assert std_floor_regularizer(constant, labels, 0.04) == pytest.approx(0.04)
    assert std_floor_regularizer(varying, labels, 0.04) == pytest.approx(0.0)


def test_balanced_sampler_requires_label_mean() -> None:
    assert balanced_speed_sampler(pd.DataFrame({"clip_path": ["a.pt"]})) is None
    sampler = balanced_speed_sampler(
        pd.DataFrame({"label_mean": [5.0, 10.0, 50.0, 90.0]})
    )
    assert sampler is not None
    assert sampler.num_samples == 4


def test_clip_dataset_temporal_augmentation_preserves_shape(tmp_path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    payload = {
        "clip": torch.ones(3, 30, 8, 8),
        "labels": torch.arange(30, dtype=torch.float32),
        "meta": {},
    }
    torch.save(payload, clips / "clip.pt")
    pd.DataFrame(
        [{"clip_path": "clips/clip.pt", "label_mean": 14.5}]
    ).to_csv(tmp_path / "index.csv", index=False)
    dataset = ClipDataset(
        tmp_path / "index.csv",
        temporal_crop=True,
        temporal_crop_min_ratio=0.8,
        time_mask_probability=1.0,
        time_mask_length=3,
        smooth_window=5,
    )
    clip, labels = dataset[0]
    assert clip.shape == (3, 30, 8, 8)
    assert labels.shape == (30,)
    assert torch.count_nonzero(clip == 0) > 0
