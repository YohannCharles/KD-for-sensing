#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.mmw.preparation import load_preparation_config, prepare_town10_skybridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local MMW Town10 skybridge data for KD sensing.")
    parser.add_argument("--config", "-c", required=True, help="Path to configs/preprocess/mmw_town10_skybridge.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and layout without extraction or writes.")
    parser.add_argument("--force", action="store_true", help="Re-extract zip inputs and rebuild derived artifacts.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value with dotted key=value syntax, e.g. mmw.sensor_zip=/path/file.zip.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override)
    for item in unknown:
        if "=" in item:
            overrides.append(item)
        else:
            parser.error(f"Unknown argument: {item}")
    config = load_preparation_config(args.config, overrides)
    result = prepare_town10_skybridge(config, dry_run=args.dry_run, force=args.force)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
