# Reproducibility

This page records the public, decision-relevant settings for v1.0.0. It does
not turn the private recordings into a public dataset, and the release remains
validation-only.

## Formal configuration

| Setting | Value |
| --- | --- |
| Frames per clip | 30 |
| Frame rate | 30 fps |
| Frame size | 224×224 |
| Input range | `[0, 1]` |
| View | Complete frame, directly resized to a square |
| Target | Per-frame speed |
| Target range | 0–120 km/h |
| Backbone | R3D-18, random initialization |
| Temporal head | First/second-order differences + dilated TCN |
| Inference stride | 15 frames |
| Window fusion | Triangular temporal weights |
| Training seed | 42 |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Maximum learning rate | `6e-4` |
| Learning-rate schedule | Cosine with 5-epoch warm-up |
| Batch size | 6 |
| Maximum epochs | 120 |
| Formal checkpoint | Epoch 8 |
| Calibration | Disabled by default |

The machine-readable configuration is
`configs/thesis_r3d18_tcn.yaml`. If a value here and the configuration ever
differ, the tagged release configuration and model metadata are authoritative.

## Thesis environment

The formal experiment was run on Windows 11 with an AMD Ryzen 7 5800X3D,
64 GB RAM, and an NVIDIA RTX 3080 10 GB. The recorded software environment
included Python 3.13, PyTorch 2.6.0 with CUDA 12.4, torchvision 0.21.0, OpenCV
4.12.0.88, and FFmpeg 4.2.11.

The public package targets Python 3.11–3.13 and supports CPU inference. Minor
floating-point differences across hardware, codecs, and library builds are
expected.

## Checkpoint procedure

1. Download the v1.0.0 model, metadata, and `SHA256SUMS.txt` from the same
   GitHub Release.
2. Verify SHA-256 before loading.
3. Load only the state dictionary into the declared architecture with strict
   key checking.
4. Keep calibration disabled when reproducing the primary result.

The release checkpoint is converted to CPU storage without changing tensor
values. Its metadata records both the source-checkpoint and release-checkpoint
digests.

## Inference contract

- Decode or resample input at 30 fps.
- Reject videos with fewer than 30 decodable frames.
- Resize the complete RGB frame directly to 224×224 and scale to `[0, 1]`.
- Form 30-frame windows at a 15-frame stride.
- Pad the final incomplete window with its last frame, then trim predictions to
  the original decoded length.
- Fuse overlap with triangular temporal weights.
- Emit one finite speed value per decoded frame.

The public prediction schema is:

```text
frame_index,timestamp_s,predicted_speed_kmh
```

When reference labels are combined with predictions for the demo or evaluation,
the schema is:

```text
frame_index,timestamp_s,reference_speed_kmh,predicted_speed_kmh
```

## Metric convention

Metrics are computed over aligned finite frame pairs in km/h:

- MAE: mean absolute error;
- RMSE: square root of mean squared error;
- Bias: mean of `prediction - reference`;
- R²: coefficient of determination; and
- Pearson r: linear correlation.

The primary validation result is uncalibrated:

```text
MAE=1.809, RMSE=2.923, Bias=-1.003, R2=0.991, Pearson=0.996
```

The optional affine calibration produced by the training workflow is fitted on
the same validation set and is supplementary only. It is never enabled by
default.

## Verification levels

- **Unit tests:** preprocessing, scaling, temporal differences, fusion, label
  cleaning, and metrics.
- **Model tests:** input/output shape, strict keys, CPU loading, and finite
  output.
- **Demo regression:** 300 rows, monotonic relative timestamps, no missing
  values, and prediction agreement within `atol=rtol=1e-4`.
- **Private research check:** recompute aggregate validation metrics without
  publishing the full trajectory.
- **Privacy check:** confirm no audio or sensitive metadata and manually review
  every public demo frame.
