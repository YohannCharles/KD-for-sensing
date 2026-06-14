from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.baselines.amr_net_gps_image.report import DEFAULT_OUTPUT_ROOT, run_amr_net_gps_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AMR-Net_gps_image source-audit report and mock/local-substitute manifest."
    )
    parser.add_argument("--config", "-c", default="configs/baselines/amr_net_gps_image.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Report output directory; default: {DEFAULT_OUTPUT_ROOT}/<run_id>.")
    parser.add_argument("--claim-status", default=None, help="Requested claim status; official_reproduction is rejected unless audit is complete.")
    parser.add_argument("--mock", dest="mock", action="store_true", default=True, help="Write deterministic synthetic smoke metrics.")
    parser.add_argument("--no-mock", dest="mock", action="store_false", help="Write a blocked/local manifest without synthetic metrics.")
    parser.add_argument("--no-write", action="store_true", help="Return the manifest without writing report files.")
    return parser


def run_main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = ["kd-sensing-run-amr-net-gps-image", *(argv or [])]
    return run_amr_net_gps_image(
        config_path=args.config,
        output_dir=args.output_dir,
        mock=bool(args.mock),
        claim_status=args.claim_status,
        command=command,
        write=not bool(args.no_write),
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
