# Third-Party Notices

This project depends on third-party software. Those components remain under
their own licenses; the repository's MIT License does not replace them.

Declared runtime and optional dependencies are:

| Component | Typical license | Purpose |
| --- | --- | --- |
| PyTorch | BSD-3-Clause | Tensor operations and model training |
| torchvision | BSD-3-Clause | R3D-18 video architecture |
| NumPy | BSD-3-Clause | Numerical arrays |
| pandas | BSD-3-Clause | Tabular labels and predictions |
| OpenCV | Apache-2.0 | Video and image processing |
| pytesseract | Apache-2.0 | Optional OCR integration |
| PyYAML | MIT | Configuration files |
| tqdm | MPL-2.0 and MIT | Progress reporting |
| Tesseract OCR | Apache-2.0 | System OCR executable used by the optional command |
| FFmpeg | LGPL-2.1-or-later or GPL, depending on build | Video decoding and encoding |

Development and test tools have their own licenses as declared by their
distributions. Python dependency constraints are recorded in `pyproject.toml`;
system FFmpeg and Tesseract versions depend on the user's installation.

No third-party pretrained model weights are claimed as part of the released
checkpoint: the thesis R3D-18 backbone was initialized without torchvision
pretrained weights. Users remain responsible for reviewing the license of
their installed FFmpeg build and any optional OCR model files.
