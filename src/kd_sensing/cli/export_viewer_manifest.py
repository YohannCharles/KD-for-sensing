from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.diagnostics.viewer_manifest import export_viewer_manifest
from kd_sensing.diagnostics.viewer_predictions import (
    export_viewer_model_predictions,
    parse_key_value_paths,
    parse_modalities,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a manifest for the Gradio multimodal viewer.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML training/evaluation config.")
    parser.add_argument(
        "--output",
        "-o",
        help="Explicit samples.json path. Defaults to cache-dir/<fingerprint>/samples.json.",
    )
    parser.add_argument("--cache-dir", help="Directory for reusable processed viewer cache.")
    parser.add_argument(
        "--scenes",
        "--scene",
        dest="scenes",
        help=(
            "Scene ids/aliases to export, comma-separated. "
            "Use --scenes 9,32 or --scenes all to include multiple scenes in one manifest."
        ),
    )
    parser.add_argument("--predictions", help="Optional prediction JSON to merge into manifest records.")
    parser.add_argument("--quality", help="Optional quality-score JSON to merge into manifest records.")
    parser.add_argument("--gate", help="Optional gate-weight JSON to merge into manifest records.")
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run single-modality checkpoints first and merge their per-beam confidence curves.",
    )
    parser.add_argument(
        "--prediction-modalities",
        help="Comma-separated modalities for --run-models. Defaults to diagnostics.visualization.modalities.",
    )
    parser.add_argument(
        "--model-config",
        action="append",
        default=[],
        help="Override a modality model config as modality=path. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        default=[],
        help="Override a modality checkpoint as modality=path. Can be repeated or comma-separated.",
    )
    parser.add_argument("--model-devices", default="auto", help="Devices for model inference, e.g. auto, cpu, cuda:0,cuda:1.")
    parser.add_argument("--model-workers", type=int, help="Number of modality inference workers. Defaults to parallel.")
    parser.add_argument("--model-batch-size", type=int, default=32, help="Batch size for model prediction export.")
    parser.add_argument("--model-num-workers", type=int, default=0, help="DataLoader workers per model prediction worker.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the requested manifest path.")
    parser.add_argument("--force-rebuild", action="store_true", help="Reprocess the dataset even if a valid cache exists.")
    parser.add_argument("--sample-limit", type=int, help="Optional cap for quick debugging. Defaults to all samples.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    prediction_result = None
    predictions = args.predictions
    if args.run_models:
        prediction_result = export_viewer_model_predictions(
            cfg,
            output_path=args.predictions,
            cache_dir=args.cache_dir,
            modalities=parse_modalities(args.prediction_modalities),
            model_config_paths=parse_key_value_paths(args.model_config),
            checkpoint_paths=parse_key_value_paths(args.model_checkpoint),
            devices=args.model_devices,
            workers=args.model_workers,
            batch_size=args.model_batch_size,
            num_workers=args.model_num_workers,
            force_rebuild=bool(args.force_rebuild),
            sample_limit=args.sample_limit,
        )
        predictions = prediction_result["prediction_path"]
    result = export_viewer_manifest(
        cfg,
        output_path=args.output,
        cache_dir=args.cache_dir,
        predictions=predictions,
        quality=args.quality,
        gate=args.gate,
        overwrite=bool(args.overwrite),
        force_rebuild=bool(args.force_rebuild),
        sample_limit=args.sample_limit,
    )
    if prediction_result is not None:
        result["model_predictions"] = prediction_result
    print_result(result)
    return result


if __name__ == "__main__":
    main()
