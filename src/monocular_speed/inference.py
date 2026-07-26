from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .model import choose_device, load_model
from .video import decode_video, frames_to_tensor, padded_window, window_starts


def triangular_weights(length: int, edge: float = 1.0, peak: float = 2.0) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float32)
    center = (length - 1) / 2.0
    positions = np.arange(length, dtype=np.float32)
    distance = np.abs(positions - center) / max(center, 1e-6)
    return (edge + (peak - edge) * (1.0 - distance)).astype(np.float32)


def infer_video(
    video: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    fps: float = 30.0,
    image_size: int = 224,
    window: int = 30,
    stride: int = 15,
    calibration: tuple[float, float] | None = None,
) -> np.ndarray:
    frames = decode_video(video, target_fps=fps)
    tensor = frames_to_tensor(frames, image_size=image_size)
    runtime_device = choose_device(device)
    model = load_model(
        checkpoint,
        device=runtime_device,
        output_frames=window,
    )
    weights = triangular_weights(window)
    prediction_sum = np.zeros(len(frames), dtype=np.float64)
    weight_sum = np.zeros(len(frames), dtype=np.float64)
    with torch.inference_mode():
        for start in window_starts(len(frames), window=window, stride=stride):
            segment = padded_window(tensor, start, window=window)
            prediction = (
                model(segment.unsqueeze(0).to(runtime_device))["perframe_kmh"]
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )
            usable = min(window, len(frames) - start)
            prediction_sum[start : start + usable] += prediction[:usable] * weights[:usable]
            weight_sum[start : start + usable] += weights[:usable]
    if np.any(weight_sum == 0):
        raise RuntimeError("Inference windows did not cover every decoded frame.")
    predictions = (prediction_sum / weight_sum).astype(np.float32)
    if calibration is not None:
        slope, intercept = calibration
        predictions = slope * predictions + intercept
    return np.clip(predictions, 0.0, 120.0).astype(np.float32)


def load_calibration(path: str | Path) -> tuple[float, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("fitted_on") != "val":
        raise ValueError("Calibration metadata must identify the fitting split.")
    return float(data["a"]), float(data["b"])


def write_predictions(
    predictions: np.ndarray,
    output: str | Path,
    *,
    fps: float = 30.0,
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame_index", "timestamp_s", "predicted_speed_kmh"])
        for index, value in enumerate(predictions):
            writer.writerow([index, f"{index / fps:.6f}", f"{float(value):.6f}"])
    return output_path
