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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_weights(version: str, output: str | Path) -> Path:
    tag = version if version.startswith("v") else f"v{version}"
    plain_version = tag.removeprefix("v")
    model_filename = f"r3d18_tcn_thesis_v{plain_version}.pt"
    metadata_filename = f"r3d18_tcn_thesis_v{plain_version}.json"
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}"
    model_path = output_path / model_filename
    metadata_path = output_path / metadata_filename
    checksums_path = output_path / "SHA256SUMS.txt"
    downloaded = (model_path, metadata_path, checksums_path)

    def remove_downloaded() -> None:
        for path in downloaded:
            path.unlink(missing_ok=True)

    try:
        _download(f"{base}/{model_filename}", model_path)
        _download(f"{base}/{metadata_filename}", metadata_path)
        _download(f"{base}/SHA256SUMS.txt", checksums_path)
    except HTTPError as error:
        remove_downloaded()
        raise RuntimeError(f"Unable to download release {tag}: HTTP {error.code}") from error

    entries: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            entries[parts[-1].lstrip("*")] = parts[0].lower()

    for filename, path in (
        (model_filename, model_path),
        (metadata_filename, metadata_path),
    ):
        if filename not in entries:
            remove_downloaded()
            raise RuntimeError(f"SHA256SUMS.txt does not list {filename}.")
        if _sha256(path) != entries[filename]:
            remove_downloaded()
            raise RuntimeError(f"Downloaded asset failed SHA-256 verification: {filename}")

    return model_path
