from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def decode_video(path: str | Path, target_fps: float = 30.0) -> list[np.ndarray]:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not decoded:
        raise ValueError(f"No decodable frames found: {video_path}")
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = target_fps
    if abs(source_fps - target_fps) <= 0.01:
        return decoded
    duration = len(decoded) / source_fps
    target_count = max(1, int(round(duration * target_fps)))
    source_indices = np.rint(np.arange(target_count) * source_fps / target_fps).astype(int)
    source_indices = np.clip(source_indices, 0, len(decoded) - 1)
    return [decoded[index] for index in source_indices]


def frames_to_tensor(frames: list[np.ndarray], image_size: int = 224) -> torch.Tensor:
    array = np.stack(frames).astype(np.float32) / 255.0
    values = torch.from_numpy(array).permute(0, 3, 1, 2)
    values = F.interpolate(
        values,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return values.permute(1, 0, 2, 3).contiguous()


def window_starts(frame_count: int, window: int = 30, stride: int = 15) -> list[int]:
    if frame_count < window:
        raise ValueError(f"At least {window} decoded frames are required; found {frame_count}.")
    complete = list(range(0, frame_count - window + 1, stride))
    if complete[-1] + window < frame_count:
        complete.append(complete[-1] + stride)
    return complete


def padded_window(values: torch.Tensor, start: int, window: int = 30) -> torch.Tensor:
    segment = values[:, start : start + window]
    if segment.shape[1] == window:
        return segment
    missing = window - segment.shape[1]
    tail = segment[:, -1:].expand(-1, missing, -1, -1)
    return torch.cat([segment, tail], dim=1)
