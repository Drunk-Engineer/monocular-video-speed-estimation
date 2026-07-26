import json
from pathlib import Path

import pandas as pd

from monocular_speed.evaluation import evaluate_files


def test_evaluate_files_writes_metrics(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "output"
    pd.DataFrame(
        {"frame_index": [0, 1, 2], "reference_speed_kmh": [10.0, 11.0, 12.0]}
    ).to_csv(labels, index=False)
    pd.DataFrame(
        {"frame_index": [0, 1, 2], "predicted_speed_kmh": [10.0, 12.0, 11.0]}
    ).to_csv(predictions, index=False)

    result = evaluate_files(predictions, labels, output)

    assert result["n_frames"] == 3
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert stored == result
