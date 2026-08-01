# v1.0.0 — Validation-Only Research Release

> **Metadata correction (2026-08-01):** The author name and thesis month below
> were corrected to match the final thesis cover. The software, model weights,
> validation results, release date, and DOI are unchanged.

This is the first public release of the code and trained model associated with:

> Jie-Chen Ke, *Per-Frame Vehicle Speed Estimation from Monocular Dashcam
> Videos Using 3D Convolutional Neural Networks*, master's thesis, National
> Taipei University of Education, August 2026.

The archival record for this version is
[https://doi.org/10.5281/zenodo.21604583](https://doi.org/10.5281/zenodo.21604583).

## Included

- installable preprocessing, training, inference, evaluation, and OCR commands;
- the R3D-18 plus dilated TCN architecture and formal thesis configuration;
- a CPU-compatible epoch 8 state dictionary, model metadata, and SHA-256
  checksums as Release assets;
- one anonymized 10-second training-source demo MP4;
- a 300-row demo result CSV and matching GIF; and
- model, data, licensing, citation, and reproducibility documentation.

## Primary validation result

The uncalibrated per-frame validation result is:

| MAE (km/h) | RMSE (km/h) | Bias (km/h) | R² | Pearson r |
| ---: | ---: | ---: | ---: | ---: |
| 1.809 | 2.923 | -1.003 | 0.991 | 0.996 |

Same-validation affine calibration is included only as a supplementary
reproduction option and is disabled by default.

## Important limitations

- This release is validation-only and has no complete independent public test
  output.
- Data were collected on one fixed approximately 15 km urban route in northern
  Taiwan with one capture setup.
- The full-frame input retains the physical dashboard, so dashboard shortcut
  learning is possible.
- Reference labels are GPS-display OCR values with cleaning and an estimated
  relative lag of approximately eight frames.
- The demo comes from training-source footage and is not evaluation evidence.

Do not use this release for law enforcement, certified speed measurement,
vehicle control, or any other safety-critical application.

## Licenses

- Code and documentation: MIT
- Trained weights: CC BY 4.0
- Public demo MP4, CSV, and GIF: CC BY 4.0

Verify every downloaded model file with the `SHA256SUMS.txt` attached to this
Release.
