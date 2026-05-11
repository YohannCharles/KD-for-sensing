#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.phase_1_5_utility_validation import (  # noqa: E402
    run_phase_1_5_utility_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1.5 utility validation summaries.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/analysis/phase_1_5_utility_validation.yaml",
        help="Phase 1.5 manifest YAML.",
    )
    parser.add_argument("--output-dir", help="Override Phase 1.5 output directory.")
    parser.add_argument("--num-bootstrap", type=int, help="Override bootstrap sample count.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    result = run_phase_1_5_utility_validation(
        args.config,
        output_dir=args.output_dir,
        num_bootstrap=args.num_bootstrap,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
