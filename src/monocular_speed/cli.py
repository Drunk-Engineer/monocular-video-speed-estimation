from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import prepare_dataset
from .download import download_weights
from .evaluation import evaluate_files
from .inference import infer_video, load_calibration, write_predictions
from .labels import extract_overlay_speeds, write_reference_csv
from .training import train_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monocular-speed",
        description="Per-frame speed estimation from full-view monocular riding video.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run uncalibrated video inference.")
    infer.add_argument("--input", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    infer.add_argument("--calibrator", type=Path)

    evaluate = commands.add_parser("evaluate", help="Evaluate aligned prediction and label CSVs.")
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--labels", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)

    prepare = commands.add_parser("prepare", help="Build private 30-frame training clips.")
    prepare.add_argument("--video", required=True, type=Path)
    prepare.add_argument("--labels", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    extract = commands.add_parser("extract-labels", help="OCR a private GPS-display video.")
    extract.add_argument("--video-gps", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)

    train = commands.add_parser("train", help="Train from a YAML configuration.")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", default=Path("outputs/training"), type=Path)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    download = commands.add_parser(
        "download-weights",
        help="Download and verify a release model bundle.",
    )
    download.add_argument("--version", default="v1.0.0")
    download.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "infer":
        calibration = load_calibration(args.calibrator) if args.calibrator else None
        predictions = infer_video(
            args.input,
            args.checkpoint,
            device=args.device,
            calibration=calibration,
        )
        output = write_predictions(predictions, args.output / "predictions.csv")
        print(output)
    elif args.command == "evaluate":
        print(json.dumps(evaluate_files(args.predictions, args.labels, args.output), indent=2))
    elif args.command == "prepare":
        print(prepare_dataset(args.video, args.labels, args.output))
    elif args.command == "extract-labels":
        values = extract_overlay_speeds(args.video_gps)
        print(write_reference_csv(values, args.output))
    elif args.command == "train":
        print(train_from_config(args.config, args.output, device=args.device))
    elif args.command == "download-weights":
        print(download_weights(args.version, args.output))


if __name__ == "__main__":
    main()
