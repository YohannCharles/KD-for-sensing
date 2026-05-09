from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.diagnostics import export_viewer_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Gradio viewer manifest. The static PNG modality visualizer has been retired."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output", help="Explicit samples.json path. Defaults to cache-dir/<fingerprint>/samples.json.")
    parser.add_argument("--cache-dir", help="Directory for reusable processed viewer cache.")
    parser.add_argument("--predictions", help="Optional prediction JSON to merge into manifest records.")
    parser.add_argument("--quality", help="Optional quality-score JSON to merge into manifest records.")
    parser.add_argument("--gate", help="Optional gate-weight JSON to merge into manifest records.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the requested manifest path.")
    parser.add_argument("--force-rebuild", action="store_true", help="Reprocess the dataset even if a valid cache exists.")
    parser.add_argument("--sample-limit", type=int, help="Optional cap for quick debugging. Defaults to all samples.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    result = export_viewer_manifest(
        cfg,
        output_path=args.output,
        cache_dir=args.cache_dir,
        predictions=args.predictions,
        quality=args.quality,
        gate=args.gate,
        overwrite=bool(args.overwrite),
        force_rebuild=bool(args.force_rebuild),
        sample_limit=args.sample_limit,
    )
    print_result(result)
    return result


if __name__ == "__main__":
    main()
