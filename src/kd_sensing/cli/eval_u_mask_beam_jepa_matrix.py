import argparse
from pathlib import Path

from kd_sensing.cli.common import (
    add_temporal_window_missing_args,
    apply_temporal_window_missing_cli_args,
    load_cli_config,
)
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import (
    evaluate_missing_matrix,
    evaluate_oracle_gate_matrix,
    format_results_markdown,
    save_results_csv,
    save_results_json,
    save_results_markdown,
)
from kd_sensing.eval.missing_patterns import resolve_missing_patterns
from kd_sensing.utils.checkpoint import load_model_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate U-MaskBeamJEPA across missing-modality mask patterns.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for eval_matrix CSV/JSON/Markdown outputs.")
    parser.add_argument("--split", choices=("val", "validation", "test"), default=None)
    parser.add_argument("--patterns", nargs="*", default=None, help="Pattern names to evaluate, or omitted/default.")
    parser.add_argument("--random-missing", nargs="*", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prediction-index", default=None)
    parser.add_argument("--eval-oracle-gate", "--eval_oracle_gate", action="store_true")
    parser.add_argument("--device", default=None, help="Override experiment.device, e.g. cuda or cpu.")
    parser.add_argument("--override", "-o", action="append", default=[])
    add_temporal_window_missing_args(parser)
    return parser


def run(argv: list[str] | None = None) -> list[dict]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    apply_temporal_window_missing_cli_args(cfg, args)
    eval_cfg = cfg.get("eval_matrix", {}) if isinstance(cfg.get("eval_matrix"), dict) else {}
    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device
    patterns = args.patterns if args.patterns is not None else eval_cfg.get("patterns", "default")
    device = build_device(cfg)
    dataloaders = build_dataloaders(cfg)
    split = args.split or eval_cfg.get("split", "val")
    split_key = _resolve_split(dataloaders, split)
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(
        args.checkpoint,
        model,
        role="u-mask eval matrix",
        map_location=device,
        strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
    )
    model_cfg = cfg["model"]["primary"]
    modalities = list(model_cfg.get("modalities") or ["image", "radar", "lidar", "gps"])
    fixed_patterns = resolve_missing_patterns(patterns, modalities)
    configured_random = _random_patterns(patterns)
    random_missing = args.random_missing if args.random_missing is not None else eval_cfg.get("random_missing", configured_random)
    if args.random_missing is None and configured_random:
        random_missing = configured_random
    prediction_index = args.prediction_index if args.prediction_index is not None else eval_cfg.get("prediction_index", "last")
    max_batches = args.max_batches if args.max_batches is not None else eval_cfg.get("max_batches")
    results = evaluate_missing_matrix(
        model,
        dataloaders[split_key],
        device,
        modalities,
        patterns=fixed_patterns,
        random_missing=random_missing,
        prediction_index=prediction_index,
        max_batches=max_batches,
        cfg=cfg,
    )
    output_dir = Path(args.output_dir or eval_cfg.get("output_dir", "outputs/eval/u_mask_beam_jepa_matrix"))
    save_results_csv(results, output_dir / "eval_matrix.csv")
    save_results_json(results, output_dir / "eval_matrix.json")
    save_results_markdown(results, output_dir / "eval_matrix.md")
    if args.eval_oracle_gate:
        oracle_results = evaluate_oracle_gate_matrix(
            model,
            dataloaders[split_key],
            device,
            modalities,
            patterns=fixed_patterns,
            random_missing=random_missing,
            prediction_index=prediction_index,
            max_batches=max_batches,
            cfg=cfg,
        )
        save_results_csv(oracle_results, output_dir / "oracle_eval_matrix.csv")
        save_results_json(oracle_results, output_dir / "oracle_eval_matrix.json")
        save_results_markdown(oracle_results, output_dir / "oracle_eval_matrix.md")
    print(format_results_markdown(results))
    return results


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


def _resolve_split(dataloaders: dict, split: str) -> str:
    candidates = ("validation", "val", "test") if split in {"val", "validation"} else (split,)
    for candidate in candidates:
        if candidate in dataloaders:
            return candidate
    raise ValueError(f"Requested split '{split}' is unavailable. Available: {sorted(dataloaders)}")


def _random_patterns(patterns) -> list[float]:
    if patterns is None or patterns == "default":
        return []
    if isinstance(patterns, str):
        values = [item for item in patterns.replace(",", " ").split() if item]
    else:
        values = [str(item) for item in patterns]
    return [float(item.split("_", 1)[1]) for item in values if item.startswith("random_")]


if __name__ == "__main__":
    raise SystemExit(main())
