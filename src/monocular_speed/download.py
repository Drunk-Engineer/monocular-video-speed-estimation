from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

REPOSITORY = "Drunk-Engineer/monocular-video-speed-estimation"


def _download(url: str, destination: Path) -> None:
    with urlopen(url) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def download_weights(version: str, output: str | Path) -> Path:
    tag = version if version.startswith("v") else f"v{version}"
    plain_version = tag.removeprefix("v")
    filename = f"r3d18_tcn_thesis_v{plain_version}.pt"
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}"
    model_path = output_path / filename
    checksums_path = output_path / "SHA256SUMS.txt"
    try:
        _download(f"{base}/{filename}", model_path)
        _download(f"{base}/SHA256SUMS.txt", checksums_path)
    except HTTPError as error:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to download release {tag}: HTTP {error.code}") from error
    entries: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            entries[parts[-1].lstrip("*")] = parts[0].lower()
    if filename not in entries:
        raise RuntimeError(f"SHA256SUMS.txt does not list {filename}.")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != entries[filename]:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded checkpoint failed SHA-256 verification.")
    return model_path
