from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from .video import decode_video, frames_to_tensor


def _uniform_indices(length: int, target_length: int) -> torch.Tensor:
    if length <= 0 or target_length <= 0:
        raise ValueError("Temporal lengths must be positive.")
    return torch.linspace(0, length - 1, target_length).round().long()


def _nan_safe_moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("smooth_window must be an odd positive integer.")
    pad = window // 2
    finite = torch.isfinite(values)
    numerator_values = torch.where(finite, values, torch.zeros_like(values))
    numerator = F.conv1d(
        F.pad(numerator_values[None, None], (pad, pad), mode="replicate"),
        torch.ones(1, 1, window, dtype=values.dtype),
    ).squeeze()
    denominator = F.conv1d(
        F.pad(finite.float()[None, None], (pad, pad), mode="replicate"),
        torch.ones(1, 1, window, dtype=values.dtype),
    ).squeeze()
    return numerator / denominator.clamp_min(1.0)


def prepare_dataset(
    video: str | Path,
    labels: str | Path,
    output: str | Path,
    *,
    fps: float = 30.0,
    image_size: int = 224,
    window: int = 30,
) -> Path:
    frames = decode_video(video, target_fps=fps)
    table = pd.read_csv(labels)
    if "reference_speed_kmh" in table:
        references = table["reference_speed_kmh"].to_numpy(dtype=np.float32)
    elif "speed" in table:
        references = table["speed"].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Labels require reference_speed_kmh or speed.")
    usable = min(len(frames), len(references))
    usable -= usable % window
    if usable < window:
        raise ValueError("Video and labels do not contain one complete clip.")
    tensor = frames_to_tensor(frames[:usable], image_size=image_size)
    output_path = Path(output)
    clips_path = output_path / "clips"
    clips_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for clip_index, start in enumerate(range(0, usable, window)):
        clip_file = clips_path / f"clip_{clip_index:06d}.pt"
        payload = {
            "clip": tensor[:, start : start + window].contiguous(),
            "labels": torch.from_numpy(references[start : start + window].copy()),
            "meta": {"start_frame": start, "end_frame": start + window - 1},
        }
        torch.save(payload, clip_file)
        records.append(
            {
                "clip_path": str(clip_file.relative_to(output_path)),
                "start_frame": start,
                "end_frame": start + window - 1,
                "label_mean": float(np.nanmean(references[start : start + window])),
            }
        )
    index_path = output_path / "index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False, lineterminator="\n")
    return index_path


class ClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        index_path: str | Path,
        *,
        target_frames: int = 30,
        temporal_crop: bool = False,
        temporal_crop_min_ratio: float = 0.8,
        time_mask_probability: float = 0.0,
        time_mask_length: int = 3,
        label_shift: int = 0,
        smooth_window: int = 5,
    ) -> None:
        self.index_path = Path(index_path)
        self.records = pd.read_csv(self.index_path)
        self.target_frames = int(target_frames)
        self.temporal_crop = bool(temporal_crop)
        self.temporal_crop_min_ratio = float(temporal_crop_min_ratio)
        self.time_mask_probability = float(time_mask_probability)
        self.time_mask_length = int(time_mask_length)
        self.label_shift = int(label_shift)
        self.smooth_window = int(smooth_window)
        if not 0 < self.temporal_crop_min_ratio <= 1:
            raise ValueError("temporal_crop_min_ratio must be in (0, 1].")
        if not 0 <= self.time_mask_probability <= 1:
            raise ValueError("time_mask_probability must be in [0, 1].")
        if self.time_mask_length < 0:
            raise ValueError("time_mask_length must be non-negative.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        relative = Path(self.records.iloc[index]["clip_path"])
        payload = torch.load(
            self.index_path.parent / relative,
            map_location="cpu",
            weights_only=True,
        )
        clip = payload["clip"].float().clone()
        labels = payload["labels"].float().clone()
        if clip.ndim != 4 or labels.ndim != 1 or clip.shape[1] != labels.shape[0]:
            raise ValueError("Prepared clips require [C,T,H,W] video and [T] labels.")
        temporal_length = clip.shape[1]
        if self.temporal_crop and temporal_length > 2:
            ratio = random.uniform(self.temporal_crop_min_ratio, 1.0)
            crop_length = max(2, int(temporal_length * ratio))
            if crop_length < temporal_length:
                start = random.randint(0, temporal_length - crop_length)
                clip = clip[:, start : start + crop_length]
                labels = labels[start : start + crop_length]
                temporal_length = crop_length
        if (
            self.time_mask_probability > 0
            and self.time_mask_length > 0
            and temporal_length > 1
            and random.random() < self.time_mask_probability
        ):
            mask_length = min(self.time_mask_length, temporal_length - 1)
            start = random.randint(0, temporal_length - mask_length)
            clip[:, start : start + mask_length] = 0
        if temporal_length != self.target_frames:
            indices = _uniform_indices(temporal_length, self.target_frames)
            clip = clip[:, indices]
            labels = labels[indices]
        if self.label_shift:
            amount = abs(self.label_shift)
            shifted = torch.full_like(labels, float("nan"))
            if amount < len(labels):
                if self.label_shift > 0:
                    shifted[:-amount] = labels[amount:]
                else:
                    shifted[amount:] = labels[:-amount]
            labels = shifted
        labels = _nan_safe_moving_average(labels, self.smooth_window)
        return clip, labels


def balanced_speed_sampler(
    records: pd.DataFrame,
    *,
    bins: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 120.0),
) -> WeightedRandomSampler | None:
    if "label_mean" not in records or records.empty:
        return None
    values = records["label_mean"].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return None
    values = np.nan_to_num(values, nan=float(np.nanmean(values)))
    bin_ids = np.clip(np.digitize(values, bins[1:-1], right=False), 0, len(bins) - 2)
    counts = np.bincount(bin_ids, minlength=len(bins) - 1).astype(float)
    counts[counts == 0] = 1.0
    weights = torch.from_numpy((1.0 / counts[bin_ids]).astype(np.float64))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
