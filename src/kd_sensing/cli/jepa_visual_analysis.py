from __future__ import annotations

import argparse
import json
import sys

from kd_sensing.diagnostics.jepa_visual_analysis import run_jepa_visual_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export offline JEPA visual analysis figures, tables, and report.")
    parser.add_argument("--analysis-config", required=True, help="Path to the JEPA visual analysis YAML config.")
    parser.add_argument("--output-dir", required=False, help="Directory for analysis outputs.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override analysis config value using dotted key=value syntax. Can be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty analysis output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and write manifest/report without model forward.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    command = ["kd-sensing-jepa-visual-analysis", *(argv if argv is not None else sys.argv[1:])]
    result = run_jepa_visual_analysis(
        analysis_config=args.analysis_config,
        output_dir=args.output_dir,
        overrides=overrides,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        command=command,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
