# Release Checklist

Use this checklist for v1.0.0 and later releases. Never copy the private
research repository or its `.git` directory into the public repository.

## Source and privacy

- [ ] Build from the explicit public-file allowlist.
- [ ] Confirm no complete source video, GPS-display video, thesis PDF, OCR
      crops, raw frames, private labels, or full prediction traces are tracked.
- [ ] Scan tracked text for personal filesystem paths, credentials, exact route
      details, original filenames, and absolute source timestamps.
- [ ] Confirm the demo has exactly 300 frames at 30 fps, no audio, and no
      capture, location, device, or original-name metadata.
- [ ] Manually review every demo frame and every speed transition.
- [ ] Confirm the GIF was generated from the final anonymized MP4.

## Code and model

- [ ] Run unit, model-shape, and end-to-end demo tests in a clean environment.
- [ ] Recompute the primary validation metrics from private data.
- [ ] Convert the checkpoint to CPU storage with safe state-dictionary loading.
- [ ] Strictly verify checkpoint keys and finite output.
- [ ] Record source and release SHA-256 values in model metadata.
- [ ] Confirm model files are absent from Git history.

## Documentation

- [ ] Keep the release labelled validation-only.
- [ ] Identify the dashboard shortcut and same-validation calibration.
- [ ] Mark the public clip as training-source demo, not evaluation.
- [ ] Validate `CITATION.cff`.
- [ ] Check code, model, data, and third-party license links.
- [ ] Do not insert a Zenodo DOI until it resolves publicly.

## GitHub release

- [ ] Push to a private repository and allow CI to finish.
- [ ] Re-clone into a clean directory and run the demo.
- [ ] Inspect Actions logs before changing visibility.
- [ ] Make the repository public only after the history and logs are clean.
- [ ] Enable the repository in Zenodo before publishing the GitHub Release.
- [ ] Publish tag `v1.0.0` with the model, metadata, and checksums.
- [ ] Download assets from the public Release and repeat CPU inference.
- [ ] Add the minted DOI to README and `CITATION.cff`, then publish the metadata
      update.
