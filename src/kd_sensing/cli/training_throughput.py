import argparse
import json
from pathlib import Path

from kd_sensing.config import load_config
from kd_sensing.engine.throughput_recommendations import recommend_parallel_training
from kd_sensing.engine.training_io_profile import profile_training_io


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile training IO or recommend parallel training overrides.")
    parser.add_argument("--mode", choices=("profile", "recommend"), default="profile")
    parser.add_argument("--config", "-c", required=True, help="Training YAML config.")
    parser.add_argument("--output", help="Write JSON output to this path.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples", type=int, default=32, help="Approximate number of samples to profile.")
    parser.add_argument("--warmup", type=int, default=1, help="Initial batches excluded from profile timing.")
    parser.add_argument("--device", help="Override experiment.device for profile mode.")
    parser.add_argument("--csv-output", help="Write flat CSV profile summary to this path.")
    parser.add_argument("--parallel-runs", type=int, default=4, help="Concurrent training runs for recommendation mode.")
    parser.add_argument("--cpu-count", type=int, help="CPU cores available to concurrent runs.")
    parser.add_argument("--cache-min-coverage", type=float, default=0.95, help="Coverage required before read_only cache.")
    parser.add_argument("--no-cache-check", action="store_true", help="Skip LiDAR cache coverage checks.")
    parser.add_argument("--profile-json", help="Optional profile JSON used to tune recommendation mode.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(argv)
    print(json.dumps(result, indent=2))
    return 0


def run(argv: list[str] | None = None) -> dict:
    args, unknown = build_parser().parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    if args.mode == "recommend":
        cfg = load_config(args.config, overrides)
        result = recommend_parallel_training(
            cfg,
            config_path=args.config,
            parallel_runs=args.parallel_runs,
            cpu_count=args.cpu_count,
            cache_min_coverage=args.cache_min_coverage,
            check_cache=not args.no_cache_check,
            profile=_load_profile(args.profile_json),
        )
    else:
        result = profile_training_io(
            config_path=args.config,
            split=args.split,
            samples=args.samples,
            warmup=args.warmup,
            device_override=args.device,
            output=args.output,
            csv_output=args.csv_output,
            overrides=overrides,
        )
    if args.output and args.mode == "recommend":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_profile(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
