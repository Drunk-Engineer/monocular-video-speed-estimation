# Data Card

## Overview

The complete research dataset is private. This repository publishes only one
anonymized 10-second training-source demonstration and its per-frame result
table.

The data support a master's thesis on per-frame speed estimation from full-view
monocular riding video. They do not constitute a public benchmark.

## Private research corpus

| Field | Description |
| --- | --- |
| Recordings | 12 paired riding videos |
| Collection setting | One fixed approximately 15 km urban commute in northern Taiwan |
| Capture setup | One camera and mounting configuration |
| Video rate | Approximately 30 fps |
| Label source | Synchronized GPS speed display, OCR, and temporal cleaning |
| Model input | Full-view RGB video without the GPS overlay |
| Availability | Not distributed |

The corpus covers a limited collection period and route. It is not designed to
represent different countries, road geometries, devices, mounting positions,
seasons, night-time conditions, or severe weather.

## Label generation

The paired GPS-display recording was used privately to derive reference speed:

1. align the full-view and GPS-display versions;
2. read the displayed speed with multiple Tesseract image variants;
3. apply confidence filtering, a moving median, spike rejection, interpolation,
   and bidirectional slew limiting; and
4. manually review the result before using it as a training or evaluation
   target.

The public OCR command implements this reproducible helper pipeline. OCR output
is not treated as ground truth without visual review; the released demo's 300
reference frames and all speed transitions were checked against the private
synchronized display.

Numerical GPS speed is never included in the model input. The label stream may
still contain uncertainty from display refresh rate, GPS latency, OCR, and
temporal alignment. The estimated relative lag is approximately eight frames.

## Public demonstration

The public subset contains exactly:

- `sample_data/anonymized_demo.mp4`;
- `sample_data/demo_results.csv`; and
- `assets/demo.gif`.

The video is 10 seconds, 960×540, 30 fps, H.264, and has no audio. It retains
the complete riding view and central physical dashboard. Plates, faces,
readable location text, and route clues were reviewed for anonymization.
Capture metadata, absolute source time, original filename, and GPS coordinates
are not included.

The CSV contains 300 rows with this schema:

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `frame_index` | integer | frame | Zero-based index within the public clip |
| `timestamp_s` | float | seconds | Relative time within the public clip |
| `reference_speed_kmh` | float | km/h | Cleaned GPS-derived reference |
| `predicted_speed_kmh` | float | km/h | Uncalibrated v1.0.0 model output |

The demo is derived from training-source footage. It is provided only to check
decoding, preprocessing, inference, and output generation. It must not be used
as validation or test evidence.

## Privacy and exclusions

Not published:

- complete source recordings or GPS-display recordings;
- audio, absolute timestamps, GPS coordinates, or exact route information;
- full validation or test label and prediction series;
- intermediate frames or OCR crops; and
- the thesis PDF.

Anonymization reduces disclosure risk but cannot guarantee that every scene is
unrecognizable. Please report a suspected privacy problem privately through
[SECURITY.md](SECURITY.md).

## Appropriate use

The public demo may be used to exercise this repository, demonstrate its output
format, or create derivative examples under [CC BY 4.0](DATA_LICENSE.md). It is
not suitable for training a new model, measuring generalization, location
inference, identifying people or vehicles, or reconstructing a route.

## Known limitations

- a single route and setup produce strong domain correlation;
- GPS-derived OCR labels are reference measurements, not frame-synchronous
  ground truth;
- the physical dashboard remains in the input;
- the public demo is training-source material; and
- no public demographic, geographic, or independent test coverage is claimed.

## Contact

Ke-Jie Chen — [odinswim1990@icloud.com](mailto:odinswim1990@icloud.com)
