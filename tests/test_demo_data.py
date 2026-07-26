import csv
import math
from pathlib import Path

import cv2
import torch

from monocular_speed.video import decode_video, frames_to_tensor

ROOT = Path(__file__).resolve().parents[1]


def test_public_demo_video_contract() -> None:
    capture = cv2.VideoCapture(str(ROOT / "sample_data" / "anonymized_demo.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 300
    assert capture.get(cv2.CAP_PROP_FPS) == 30.0
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 960
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 540
    capture.release()


def test_public_demo_csv_contract() -> None:
    path = ROOT / "sample_data" / "demo_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "frame_index",
        "timestamp_s",
        "reference_speed_kmh",
        "predicted_speed_kmh",
    ]
    assert len(rows) == 300
    assert [int(row["frame_index"]) for row in rows] == list(range(300))
    times = [float(row["timestamp_s"]) for row in rows]
    values = [
        float(row[column])
        for row in rows
        for column in ("reference_speed_kmh", "predicted_speed_kmh")
    ]
    assert all(later > earlier for earlier, later in zip(times, times[1:], strict=False))
    assert all(math.isfinite(value) and 0 <= value <= 120 for value in values)


def test_public_demo_decodes_to_model_input_tensor() -> None:
    frames = decode_video(ROOT / "sample_data" / "anonymized_demo.mp4")
    tensor = frames_to_tensor(frames[:30])
    assert len(frames) == 300
    assert tensor.shape == (3, 30, 224, 224)
    assert torch.isfinite(tensor).all()
    assert tensor.min() >= 0
    assert tensor.max() <= 1
