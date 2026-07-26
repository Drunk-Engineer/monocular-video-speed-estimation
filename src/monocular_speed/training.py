from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config, nested_get
from .dataset import ClipDataset, balanced_speed_sampler
from .metrics import regression_metrics
from .model import R3D18SpeedModel, choose_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_two_sided_weights(
    labels_kmh: torch.Tensor,
    *,
    high_threshold_kmh: float,
    high_factor: float,
    high_tau_kmh: float,
    low_center_kmh: float,
    low_factor: float,
    low_tau_kmh: float,
    warmup_alpha: float,
) -> torch.Tensor:
    high_tau_kmh = max(float(high_tau_kmh), 1e-6)
    low_tau_kmh = max(float(low_tau_kmh), 1e-6)
    high = 1.0 + (max(float(high_factor), 1.0) - 1.0) * torch.sigmoid(
        (labels_kmh - high_threshold_kmh) / high_tau_kmh
    )
    low = 1.0 + (max(float(low_factor), 1.0) - 1.0) * torch.exp(
        -0.5 * ((labels_kmh - low_center_kmh) / low_tau_kmh) ** 2
    )
    combined = high * low
    return 1.0 + float(warmup_alpha) * (combined - 1.0)


def std_floor_regularizer(
    predictions_scaled: torch.Tensor,
    labels_scaled: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    finite = torch.isfinite(labels_scaled)
    if not finite.any():
        return predictions_scaled.new_tensor(0.0)
    mask = finite.float()
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (predictions_scaled * mask).sum(dim=1) / count
    variance = (
        ((predictions_scaled - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / count
    )
    standard_deviation = variance.clamp_min(0.0).sqrt()
    return (float(floor) - standard_deviation).clamp_min(0.0).mean()


def _autocast_context(device: torch.device, enabled: bool, dtype: str):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    autocast_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=autocast_dtype)


def _resolve_data_path(config_path: str | Path, configured: str) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent.parent / path
    return path


def _make_loader(
    dataset: ClipDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch: int,
    shuffle: bool,
    sampler=None,
    pin_memory: bool,
) -> DataLoader:
    options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch
        options["persistent_workers"] = True
    return DataLoader(dataset, **options)


def _validation_pass(
    model: R3D18SpeedModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: str,
    speed_max_kmh: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="none")
    total_loss = 0.0
    total_values = 0
    references: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for clips, labels_kmh in loader:
            clips = clips.to(device, non_blocking=True)
            labels_kmh = labels_kmh.to(device, non_blocking=True)
            labels_scaled = labels_kmh / speed_max_kmh
            finite = torch.isfinite(labels_scaled)
            with _autocast_context(device, amp, amp_dtype):
                output = model(clips)
                losses = criterion(output["perframe_scaled"], torch.nan_to_num(labels_scaled))
            count = int(finite.sum())
            if count:
                total_loss += float(losses[finite].sum().cpu())
                total_values += count
                references.append(labels_kmh[finite].float().cpu().numpy())
                predictions.append(output["perframe_kmh"][finite].float().cpu().numpy())
    if not references:
        raise RuntimeError("Validation data contains no finite labels.")
    return (
        total_loss / total_values,
        np.concatenate(references),
        np.concatenate(predictions),
    )


def _fit_affine(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    matrix = np.column_stack([prediction, np.ones_like(prediction)])
    slope, intercept = np.linalg.lstsq(matrix, reference, rcond=None)[0]
    return float(slope), float(intercept)


def train_from_config(config_path: str | Path, output: str | Path, device: str = "auto") -> Path:
    config = load_config(config_path)
    seed = int(nested_get(config, "training.seed", 42))
    set_seed(seed)
    train_index = _resolve_data_path(
        config_path,
        str(nested_get(config, "data.train_index", "data/train_index.csv")),
    )
    validation_index = _resolve_data_path(
        config_path,
        str(nested_get(config, "data.validation_index", "data/validation_index.csv")),
    )
    if not train_index.is_file() or not validation_index.is_file():
        raise FileNotFoundError(
            "Training requires private train and validation index CSV files."
        )
    target_frames = int(nested_get(config, "model.frames", 30))
    smooth_window = int(nested_get(config, "training.smooth_window", 5))
    label_shift = int(nested_get(config, "training.label_shift", 0))
    train_dataset = ClipDataset(
        train_index,
        target_frames=target_frames,
        temporal_crop=bool(nested_get(config, "training.temporal_crop", True)),
        temporal_crop_min_ratio=float(
            nested_get(config, "training.temporal_crop_min_ratio", 0.8)
        ),
        time_mask_probability=float(
            nested_get(config, "training.time_mask_probability", 0.15)
        ),
        time_mask_length=int(nested_get(config, "training.time_mask_length", 3)),
        label_shift=label_shift,
        smooth_window=smooth_window,
    )
    validation_dataset = ClipDataset(
        validation_index,
        target_frames=target_frames,
        label_shift=label_shift,
        smooth_window=smooth_window,
    )
    batch_size = int(nested_get(config, "training.batch_size", 6))
    workers = int(nested_get(config, "training.workers", 0))
    prefetch = int(nested_get(config, "training.prefetch", 2))
    runtime_device = choose_device(device)
    sampler = None
    if nested_get(config, "training.sampler", "balanced_bins") == "balanced_bins":
        sampler = balanced_speed_sampler(train_dataset.records)
    loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        shuffle=True,
        sampler=sampler,
        pin_memory=runtime_device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        shuffle=False,
        pin_memory=runtime_device.type == "cuda",
    )
    speed_max_kmh = float(nested_get(config, "model.speed_max_kmh", 120.0))
    model = R3D18SpeedModel(
        output_frames=int(nested_get(config, "model.output_frames", 30)),
        dropout=float(nested_get(config, "model.dropout", 0.1)),
        speed_max_kmh=speed_max_kmh,
    )
    channels_last = bool(nested_get(config, "training.channels_last_3d", True))
    if channels_last and runtime_device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)
    model = model.to(runtime_device)
    if bool(nested_get(config, "training.compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    learning_rate = float(nested_get(config, "training.learning_rate", 0.0006))
    epochs = int(nested_get(config, "training.epochs", 120))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(nested_get(config, "training.weight_decay", 0.0001)),
    )
    warmup_epochs = int(nested_get(config, "training.warmup_epochs", 5))

    def learning_rate_multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    criterion = nn.SmoothL1Loss(reduction="none")
    high_threshold = float(nested_get(config, "training.speed_weight_kmh_threshold", 45.0))
    high_factor = float(nested_get(config, "training.speed_weight_factor", 1.8))
    high_tau = float(nested_get(config, "training.speed_weight_tau_kmh", 5.0))
    low_center = float(nested_get(config, "training.low_speed_weight_center_kmh", 5.0))
    low_factor = float(nested_get(config, "training.low_speed_weight_factor", 2.5))
    low_tau = float(nested_get(config, "training.low_speed_weight_tau_kmh", 5.0))
    weight_warmup = int(
        nested_get(config, "training.speed_weight_warmup_epochs", 15)
    )
    std_floor_lambda = float(nested_get(config, "training.std_floor_lambda", 0.003))
    std_floor = float(nested_get(config, "training.std_floor", 0.04))
    gradient_clip = float(nested_get(config, "training.gradient_clip", 0.1))
    amp = bool(nested_get(config, "training.amp", True))
    amp_dtype = str(nested_get(config, "training.amp_dtype", "bf16"))
    scaler = torch.amp.GradScaler("cuda", enabled=runtime_device.type == "cuda" and amp)
    minimum_epochs = int(nested_get(config, "training.minimum_epochs", 5))
    patience = int(nested_get(config, "training.patience", 10))
    early_delta = float(nested_get(config, "training.early_stopping_delta", 0.0001))
    history: list[dict[str, float]] = []
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "model_state_dict.pt"
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_reference: np.ndarray | None = None
    best_prediction: np.ndarray | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        warmup_alpha = min(1.0, max(0.0, (epoch - 1) / max(1, weight_warmup)))
        for clips, labels_kmh in loader:
            clips = clips.to(runtime_device, non_blocking=True)
            labels_kmh = labels_kmh.to(runtime_device, non_blocking=True)
            if channels_last and runtime_device.type == "cuda":
                clips = clips.to(memory_format=torch.channels_last_3d)
            labels_scaled = labels_kmh / speed_max_kmh
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(runtime_device, amp, amp_dtype):
                output_values = model(clips)
                predictions_scaled = output_values["perframe_scaled"]
                finite = torch.isfinite(labels_scaled)
                weights = build_two_sided_weights(
                    labels_kmh,
                    high_threshold_kmh=high_threshold,
                    high_factor=high_factor,
                    high_tau_kmh=high_tau,
                    low_center_kmh=low_center,
                    low_factor=low_factor,
                    low_tau_kmh=low_tau,
                    warmup_alpha=warmup_alpha,
                )
                element_loss = criterion(
                    predictions_scaled,
                    torch.nan_to_num(labels_scaled),
                )
                weighted_loss = element_loss[finite] * weights[finite]
                loss = weighted_loss.sum() / weights[finite].sum().clamp_min(1e-8)
                if std_floor_lambda > 0:
                    loss = loss + std_floor_lambda * std_floor_regularizer(
                        predictions_scaled,
                        labels_scaled,
                        std_floor,
                    )
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            count += 1
        validation_loss, references, predictions = _validation_pass(
            model,
            validation_loader,
            runtime_device,
            amp=amp,
            amp_dtype=amp_dtype,
            speed_max_kmh=speed_max_kmh,
        )
        metrics = regression_metrics(references, predictions)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(row)
        if math.isnan(row["train_loss"]):
            raise RuntimeError("Training loss is NaN.")
        improved = validation_loss + 1e-12 < best_validation_loss - early_delta
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            state_dict = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(state_dict, checkpoint_path)
            best_reference = references
            best_prediction = predictions
        else:
            epochs_without_improvement += 1
            if epoch >= minimum_epochs and epochs_without_improvement >= patience:
                break
    if best_reference is None or best_prediction is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    (output_path / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    run_record = {
        "config": config,
        "device": str(runtime_device),
        "train_index": str(Path("data") / train_index.name),
        "validation_index": str(Path("data") / validation_index.name),
        "best_validation_loss": best_validation_loss,
    }
    (output_path / "run_config.json").write_text(
        json.dumps(run_record, indent=2),
        encoding="utf-8",
    )
    if bool(nested_get(config, "training.post_calibrate", True)):
        slope, intercept = _fit_affine(best_reference, best_prediction)
        calibration = {
            "a": slope,
            "b": intercept,
            "space": "kmh",
            "fitted_on": "val",
            "n": int(len(best_reference)),
        }
        (output_path / "calibrator.json").write_text(
            json.dumps(calibration, indent=2),
            encoding="utf-8",
        )
    return checkpoint_path
