import hashlib
from pathlib import Path

import pytest

from monocular_speed.cli import main
from monocular_speed.download import download_weights


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "download-weights" in capsys.readouterr().out


def test_download_weights_fetches_and_verifies_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = b"model-state-dict"
    metadata = b'{"version":"1.0.0"}'

    def fake_download(url: str, destination: Path) -> None:
        if url.endswith(".pt"):
            destination.write_bytes(model)
        elif url.endswith(".json"):
            destination.write_bytes(metadata)
        else:
            destination.write_text(
                f"{hashlib.sha256(model).hexdigest()}  r3d18_tcn_thesis_v1.0.0.pt\n"
                f"{hashlib.sha256(metadata).hexdigest()}  r3d18_tcn_thesis_v1.0.0.json\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("monocular_speed.download._download", fake_download)
    model_path = download_weights("v1.0.0", tmp_path)

    assert model_path.read_bytes() == model
    assert (tmp_path / "r3d18_tcn_thesis_v1.0.0.json").read_bytes() == metadata
    assert (tmp_path / "SHA256SUMS.txt").is_file()


def test_download_weights_removes_failed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(url: str, destination: Path) -> None:
        if url.endswith("SHA256SUMS.txt"):
            destination.write_text(
                f"{'0' * 64}  r3d18_tcn_thesis_v1.0.0.pt\n"
                f"{'0' * 64}  r3d18_tcn_thesis_v1.0.0.json\n",
                encoding="utf-8",
            )
        else:
            destination.write_bytes(b"invalid")

    monkeypatch.setattr("monocular_speed.download._download", fake_download)

    with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
        download_weights("v1.0.0", tmp_path)

    assert not any(tmp_path.iterdir())
