from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import regression_metrics


def evaluate_files(
    predictions: str | Path,
    labels: str | Path,
    output: str | Path,
) -> dict[str, float]:
    prediction_table = pd.read_csv(predictions)
    label_table = pd.read_csv(labels)
    prediction_column = (
        "predicted_speed_kmh" if "predicted_speed_kmh" in prediction_table else "y_pred_kmh"
    )
    reference_column = (
        "reference_speed_kmh" if "reference_speed_kmh" in label_table else "y_true_kmh"
    )
    if "frame_index" in prediction_table and "frame_index" in label_table:
        merged = label_table[["frame_index", reference_column]].merge(
            prediction_table[["frame_index", prediction_column]],
            on="frame_index",
            how="inner",
            validate="one_to_one",
        )
    else:
        count = min(len(prediction_table), len(label_table))
        merged = pd.DataFrame(
            {
                reference_column: label_table[reference_column].iloc[:count].to_numpy(),
                prediction_column: prediction_table[prediction_column].iloc[:count].to_numpy(),
            }
        )
    metrics = regression_metrics(
        merged[reference_column].to_numpy(),
        merged[prediction_column].to_numpy(),
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return metrics
