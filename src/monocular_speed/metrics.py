from __future__ import annotations

import numpy as np


def regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(mask):
        raise ValueError("No aligned finite values are available for evaluation.")
    reference = reference[mask]
    prediction = prediction[mask]
    residual = prediction - reference
    denominator = np.square(reference - reference.mean()).sum()
    r2 = 1.0 - np.square(residual).sum() / denominator if denominator > 0 else float("nan")
    pearson = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if len(reference) > 1 and reference.std() > 0 and prediction.std() > 0
        else float("nan")
    )
    return {
        "mae_kmh": float(np.abs(residual).mean()),
        "rmse_kmh": float(np.sqrt(np.square(residual).mean())),
        "bias_kmh": float(residual.mean()),
        "r2": float(r2),
        "pearson_r": pearson,
        "n_frames": int(len(reference)),
    }
