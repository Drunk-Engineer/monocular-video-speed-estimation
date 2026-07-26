from pathlib import Path

import pytest
import torch

from monocular_speed.model import R3D18SpeedModel, load_model


def test_model_shape_and_finite_output() -> None:
    torch.manual_seed(0)
    model = R3D18SpeedModel(output_frames=30).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 30, 64, 64))
    assert output["perframe_kmh"].shape == (1, 30)
    assert torch.isfinite(output["perframe_kmh"]).all()
    assert torch.all((output["perframe_kmh"] >= 0) & (output["perframe_kmh"] <= 120))


def test_state_dict_round_trip_uses_strict_loading(tmp_path: Path) -> None:
    original = R3D18SpeedModel(output_frames=30)
    checkpoint = tmp_path / "state_dict.pt"
    torch.save(original.state_dict(), checkpoint)
    loaded = load_model(checkpoint)
    assert loaded.state_dict().keys() == original.state_dict().keys()


def test_state_dict_loading_rejects_missing_keys(tmp_path: Path) -> None:
    model = R3D18SpeedModel(output_frames=30)
    state_dict = model.state_dict()
    state_dict.pop(next(iter(state_dict)))
    checkpoint = tmp_path / "incomplete.pt"
    torch.save(state_dict, checkpoint)
    with pytest.raises(RuntimeError):
        load_model(checkpoint)


def test_state_dict_loading_rejects_unsupported_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsupported.pt"
    torch.save({"message": "not a state dictionary"}, checkpoint)
    with pytest.raises(TypeError):
        load_model(checkpoint)
