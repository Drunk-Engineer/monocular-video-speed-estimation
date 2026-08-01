# Model Card: R3D-18 + Dilated TCN v1.0.0

## Summary

This model estimates one speed value in kilometres per hour for each frame in a
full-view monocular riding video. It combines a randomly initialized R3D-18
video backbone, first- and second-order temporal differences, and a dilated
temporal convolutional head.

This is a validation-only research artifact. It is released to support
inspection and reproducibility of the associated master's thesis, not as a
production speedometer.

## Model details

| Field | Value |
| --- | --- |
| Author | Jie-Chen Ke |
| Version | 1.0.0 |
| Architecture | Randomly initialized R3D-18 + dilated TCN |
| Input | RGB tensor `[B, 3, 30, 224, 224]`, values in `[0, 1]` |
| Output | Per-frame speed tensor `[B, 30]`, km/h |
| Temporal rate | 30 fps |
| Supported range | 0–120 km/h |
| Primary checkpoint | Epoch 8 |
| License | [CC BY 4.0](MODEL_LICENSE.md) |

Frames are resized from the complete view directly to 224×224. Numerical GPS
speed is not passed to the network. The physical dashboard remains visible.
Overlapping inference windows use a 15-frame stride and triangular temporal
fusion.

## Intended use

Appropriate uses include:

- reproducing the thesis model and validation result;
- research on monocular video speed regression;
- studying temporal video models and dashboard shortcut behaviour; and
- running the included training-source demo as a software smoke test.

## Out-of-scope use

Do not use the model:

- for law enforcement, legal evidence, or certified speed measurement;
- to control a vehicle or any safety-critical system;
- as a replacement for calibrated GPS, OBD, IMU, or vehicle sensors;
- to claim generalization across routes, countries, devices, mounting
  positions, lighting, or weather; or
- to evaluate accuracy using the included demo, which is training-source
  material.

## Training data

The private research corpus consists of 12 recordings from one fixed,
approximately 15 km urban commute in northern Taiwan, captured with one camera
and mounting configuration. Reference speed labels were derived from a
synchronized GPS display by OCR followed by temporal cleaning. The numerical
GPS values were labels only and were not model inputs.

The full recordings, precise route, GPS-display videos, and full label series
are not distributed. See [DATA_CARD.md](DATA_CARD.md).

## Evaluation

The primary reported metrics are uncalibrated per-frame results on the thesis
validation split:

| Result | MAE (km/h) | RMSE (km/h) | Bias (km/h) | R² | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated, primary | 1.809 | 2.923 | -1.003 | 0.991 | 0.996 |
| Same-validation affine calibration, supplementary | 1.838 | 2.700 | -0.001 | 0.993 | 0.996 |

The supplementary calibrator was fitted on the same validation set. Its row is
not an independent estimate of performance. The public training workflow can
emit a validation-fitted calibrator, but no calibrator is enabled or downloaded
by default. A complete independent test output is unavailable for this release,
so v1.0.0 must not be described as test-validated.

## Limitations and risks

- **Dashboard shortcut:** the central physical instrument remains visible. The
  network may read or correlate with the dashboard instead of learning only
  scene motion.
- **Restricted domain:** data come from one fixed route, one device, one
  mounting setup, and a limited set of conditions.
- **Reference-label uncertainty:** labels depend on GPS-display updates, OCR,
  cleaning, and synchronization. The estimated relative lag is approximately
  eight frames.
- **No independent public test:** validation metrics and same-validation
  calibration cannot establish generalization.
- **Single training seed:** the release does not quantify variance across
  random seeds.
- **Full-frame distortion:** frames are resized directly to a square, changing
  their original aspect ratio.
- **Temporal and codec sensitivity:** frame-rate conversion, dropped frames,
  video stabilization, compression, and camera latency may alter predictions.
- **Bias may differ by speed regime:** aggregate metrics can hide larger errors
  in uncommon or rapidly changing conditions.

## Calibration

Calibration is disabled by default. The training command can optionally write
an affine calibrator marked `fitted_on: val` for supplementary analysis. Any
calibrated result must state where the calibrator was fitted and must not be
presented as independent evidence.

## Model integrity

The trained model is distributed through the v1.0.0 GitHub Release rather than
Git history. Each release includes a CPU-compatible state dictionary, model
metadata, and `SHA256SUMS.txt`. Load weights into the exact declared
architecture with strict key checking and verify the checksum first.

## Contact

Jie-Chen Ke — [odinswim1990@icloud.com](mailto:odinswim1990@icloud.com)
