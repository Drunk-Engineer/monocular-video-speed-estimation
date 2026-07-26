from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .video import decode_video


def _remove_single_spikes(values: np.ndarray, max_jump: float) -> np.ndarray:
    cleaned = values.copy()
    for index in range(1, len(cleaned) - 1):
        previous, current, following = cleaned[index - 1 : index + 2]
        if not np.isfinite([previous, current, following]).all():
            continue
        if (
            abs(current - previous) > max_jump
            and abs(current - following) > max_jump
            and abs(previous - following) <= max_jump
        ):
            cleaned[index] = (previous + following) / 2.0
    return cleaned


def _strict_slew_limit(values: np.ndarray, max_jump: float, iterations: int = 3) -> np.ndarray:
    cleaned = values.copy()
    for _ in range(max(1, iterations)):
        for index in range(1, len(cleaned)):
            cleaned[index] = np.clip(
                cleaned[index],
                cleaned[index - 1] - max_jump,
                cleaned[index - 1] + max_jump,
            )
        for index in range(len(cleaned) - 2, -1, -1):
            cleaned[index] = np.clip(
                cleaned[index],
                cleaned[index + 1] - max_jump,
                cleaned[index + 1] + max_jump,
            )
    return cleaned


def clean_speed_series(
    values: list[float] | np.ndarray,
    *,
    minimum: float = 0.0,
    maximum: float = 120.0,
    max_step: float = 1.0,
) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    series[(series < minimum) | (series > maximum)] = np.nan
    series = series.interpolate(limit_direction="both").ffill().bfill()
    if series.isna().any():
        raise ValueError("Speed series contains no usable values.")
    result = _remove_single_spikes(series.to_numpy(dtype=np.float64), max_step)
    result = _strict_slew_limit(result, max_step)
    return np.clip(result, minimum, maximum)


def _ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        8,
    )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    hsv_value = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:, :, 2]
    _, value_threshold = cv2.threshold(
        hsv_value,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return [adaptive, otsu, cv2.bitwise_not(otsu), value_threshold]


def _tesseract_candidate(pytesseract, image: np.ndarray) -> tuple[int | None, float]:
    data = pytesseract.image_to_data(
        image,
        config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT,
    )
    best_value: int | None = None
    best_confidence = 0.0
    for text, raw_confidence in zip(data["text"], data["conf"], strict=False):
        matches = re.findall(r"\d{1,3}", str(text))
        try:
            confidence = max(0.0, float(raw_confidence) / 100.0)
        except (TypeError, ValueError):
            confidence = 0.0
        for match in matches:
            value = int(match)
            if 0 <= value <= 120 and confidence > best_confidence:
                best_value = value
                best_confidence = confidence
    return best_value, best_confidence


def _moving_median(values: deque[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[middle])
    return int(round((ordered[middle - 1] + ordered[middle]) / 2.0))


def extract_overlay_speeds(
    video: str | Path,
    *,
    fps: float = 30.0,
    roi: tuple[float, float, float, float] = (0.80, 0.74, 0.18, 0.22),
    smooth_window: int = 5,
    spike_kmh: float = 12.0,
    minimum_confidence: float = 0.5,
) -> np.ndarray:
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError("Install the OCR extra with: pip install -e '.[ocr]'") from error
    frames = decode_video(video, target_fps=fps)
    recent: deque[int] = deque(maxlen=max(1, int(smooth_window)))
    smoothed: list[float] = []
    x, y, width, height = roi
    for frame in frames:
        frame_height, frame_width = frame.shape[:2]
        crop = frame[
            int(y * frame_height) : int((y + height) * frame_height),
            int(x * frame_width) : int((x + width) * frame_width),
        ]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        candidate: int | None = None
        confidence = 0.0
        for variant in _ocr_variants(crop):
            value, probability = _tesseract_candidate(pytesseract, variant)
            if probability > confidence:
                candidate, confidence = value, probability
        previous = _moving_median(recent)
        if candidate is not None and confidence >= minimum_confidence:
            if previous is not None and abs(candidate - previous) > spike_kmh:
                candidate = previous
            recent.append(candidate)
        current = _moving_median(recent)
        smoothed.append(float(current) if current is not None else np.nan)
    return clean_speed_series(smoothed)


def write_reference_csv(values: np.ndarray, output: str | Path, fps: float = 30.0) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "frame_index": np.arange(len(values), dtype=int),
            "timestamp_s": np.arange(len(values), dtype=float) / fps,
            "reference_speed_kmh": values,
        }
    )
    table.to_csv(output_path, index=False, float_format="%.6f", lineterminator="\n")
    return output_path
