# Monocular Video Speed Estimation

[![CI](https://github.com/Drunk-Engineer/monocular-video-speed-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/Drunk-Engineer/monocular-video-speed-estimation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Model and demo: CC BY 4.0](https://img.shields.io/badge/model%20%26%20demo-CC%20BY%204.0-lightgrey.svg)](MODEL_LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21604583.svg)](https://doi.org/10.5281/zenodo.21604583)

Per-frame vehicle speed estimation from full-view monocular riding video using a
randomly initialized R3D-18 backbone and a dilated temporal convolutional
network.

> **Research release:** v1.0.0 is validation-only. It does not provide evidence
> of independent route, camera, weather, or device generalization. Do not use it
> for law enforcement, certified measurement, or safety-critical control.

[繁體中文說明](README_zh-TW.md)

![Ten-second anonymized training-source demonstration](assets/demo.gif)

## What this repository provides

- preprocessing of 30 fps full-view video into 30-frame, 224×224 clips;
- GPS-display OCR and reference-label cleaning for privately paired recordings;
- training and evaluation for the R3D-18 plus temporal-convolution model;
- overlapping-window inference with triangular temporal fusion;
- a CPU-compatible trained checkpoint distributed as a GitHub Release asset;
- one anonymized 10-second demo video, its result CSV, and a GIF.

The model receives RGB video only; numerical GPS speed is never an input.
However, the full frame retains the physical dashboard. The network may
therefore exploit the visible instrument as a shortcut. This limitation is
central to interpreting the results.

## Validation result

The primary result is the uncalibrated per-frame result from the thesis
validation split.

| Result | MAE (km/h) | RMSE (km/h) | Bias (km/h) | R² | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated, primary | 1.809 | 2.923 | -1.003 | 0.991 | 0.996 |
| Calibrated on the same validation set, supplementary | 1.838 | 2.700 | -0.001 | 0.993 | 0.996 |

The calibrated row is not an independent evaluation: the affine calibrator was
fitted on the same validation set. No complete independent test outputs are
included in this release. See [MODEL_CARD.md](MODEL_CARD.md) for the intended
use and limitations.

## Install

Python 3.11–3.13 and FFmpeg are required. A clean environment is recommended.

```bash
git clone https://github.com/Drunk-Engineer/monocular-video-speed-estimation.git
cd monocular-video-speed-estimation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Download the v1.0.0 checkpoint:

```bash
monocular-speed download-weights \
  --version v1.0.0 \
  --output checkpoints
```

The command downloads the checkpoint, its model metadata, and
`SHA256SUMS.txt`, then verifies both listed files. The checkpoint is
intentionally excluded from Git history because it is larger than GitHub's
normal per-file limit.

## Run the public demo

```bash
monocular-speed infer \
  --input sample_data/anonymized_demo.mp4 \
  --checkpoint checkpoints/r3d18_tcn_thesis_v1.0.0.pt \
  --output outputs/demo
```

The demo is derived from training-source footage and is only an installation
and inference check. It is not validation or test evidence. Its published CSV
uses only relative fields:

```text
frame_index,timestamp_s,reference_speed_kmh,predicted_speed_kmh
```

## Commands

```bash
monocular-speed extract-labels --video-gps VIDEO --output LABELS
monocular-speed prepare --video VIDEO --labels LABELS --output DATASET
monocular-speed train --config configs/thesis_r3d18_tcn.yaml
monocular-speed infer --input VIDEO --checkpoint MODEL --output OUTPUT
monocular-speed evaluate --predictions PREDICTIONS --labels LABELS --output OUTPUT
monocular-speed download-weights --version v1.0.0 --output checkpoints
```

Calibration is disabled by default. It should be enabled only when the
same-validation calibration limitation is acceptable and explicitly reported.
OCR labels likewise require manual review; the command uses multiple Tesseract
image variants, temporal filtering, and bidirectional slew limiting but does
not make an automated ground-truth claim. `extract-labels` additionally
requires `python -m pip install -e ".[ocr]"` and a system Tesseract executable
available on `PATH`.
The reproducibility settings and output contracts are documented in
[docs/reproducibility.md](docs/reproducibility.md).

## Data and privacy

The complete recordings, paired GPS-display videos, precise route, full label
series, and thesis PDF are not distributed. The public demo has no audio,
capture timestamp, GPS coordinates, original filename, or absolute source
timestamp. Visible plates, faces, readable location text, and route clues were
reviewed for anonymization.

The research corpus contains 12 videos from one fixed approximately 15 km urban
commute in northern Taiwan, captured with one setup. Reference speeds were
derived from a GPS display using OCR and cleaning, with an estimated relative
lag of approximately eight frames. See [DATA_CARD.md](DATA_CARD.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The preferred
academic reference is:

> Ke, Jie-Chen. *Per-Frame Vehicle Speed Estimation from Monocular Dashcam
> Videos Using 3D Convolutional Neural Networks*. Master's thesis, Department
> of Computer Science, National Taipei University of Education, August 2026.

The archived v1.0.0 software release is available at
[https://doi.org/10.5281/zenodo.21604583](https://doi.org/10.5281/zenodo.21604583).

## Licenses and contact

- Source code and documentation: [MIT](LICENSE)
- Trained weights: [CC BY 4.0](MODEL_LICENSE.md)
- Public demo MP4, CSV, and GIF: [CC BY 4.0](DATA_LICENSE.md)
- Dependencies: [third-party notices](THIRD_PARTY_NOTICES.md)

Contact: [Jie-Chen Ke](mailto:odinswim1990@icloud.com)
