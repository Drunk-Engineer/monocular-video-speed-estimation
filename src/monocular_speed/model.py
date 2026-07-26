from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video import r3d_18


class DilatedTCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: int = 256,
        blocks: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, hidden_channels, 1)
        layers: list[nn.Sequential] = []
        for index in range(blocks):
            dilation = 2**index
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.proj_in(values)
        for block in self.blocks:
            hidden = self.activation(hidden + block(hidden))
        return hidden


class R3D18SpeedModel(nn.Module):
    def __init__(
        self,
        output_frames: int = 30,
        dropout: float = 0.1,
        speed_max_kmh: float = 120.0,
    ) -> None:
        super().__init__()
        backbone = r3d_18(weights=None)
        self.features = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.tcn = DilatedTCN()
        self.time_out = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(256, 1, 1))
        self.output_frames = int(output_frames)
        self.speed_max_kmh = float(speed_max_kmh)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        features_5d = self.features(video)
        features = features_5d.mean(dim=(-1, -2))
        first_difference = F.pad(features[:, :, 1:] - features[:, :, :-1], (1, 0))
        second_difference = F.pad(
            first_difference[:, :, 1:] - first_difference[:, :, :-1],
            (1, 0),
        )
        temporal = torch.cat([features, first_difference, second_difference], dim=1)
        per_frame = self.time_out(self.tcn(temporal)).squeeze(1)
        if per_frame.shape[1] != self.output_frames:
            per_frame = F.interpolate(
                per_frame.unsqueeze(1),
                size=self.output_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        scaled = torch.sigmoid(per_frame)
        return {
            "perframe_scaled": scaled,
            "perframe_kmh": scaled * self.speed_max_kmh,
            "clip_kmh": scaled.mean(dim=1) * self.speed_max_kmh,
        }


def _state_dict_from_loaded(value: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint must contain a state dictionary.")
    if "model" in value:
        candidate = value["model"]
    elif "state_dict" in value:
        candidate = value["state_dict"]
    else:
        candidate = value
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint does not contain a non-empty state dictionary.")
    valid_items = all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in candidate.items()
    )
    if not valid_items:
        raise TypeError("State dictionary contains unsupported values.")
    return candidate


def load_model(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    output_frames: int = 30,
    dropout: float = 0.1,
    speed_max_kmh: float = 120.0,
) -> R3D18SpeedModel:
    checkpoint_path = Path(checkpoint)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _state_dict_from_loaded(loaded)
    model = R3D18SpeedModel(
        output_frames=output_frames,
        dropout=dropout,
        speed_max_kmh=speed_max_kmh,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)
