from __future__ import annotations

import argparse
import json

from kd_sensing.diagnostics.predictive_gps_query_visualizations import (
    run_predictive_gps_query_visualizations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate explanatory Predictive GPS-query++ diagnostics.")
    parser.add_argument("--manifest", required=True, help="Path to a benchmark runner manifest JSON.")
    parser.add_argument("--output-dir", required=False, help="Directory for diagnostics outputs.")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty diagnostics directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_predictive_gps_query_visualizations(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
