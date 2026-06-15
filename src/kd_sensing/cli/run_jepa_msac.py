from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.baselines.jepa_msac.workflow import STAGES, run_jepa_msac


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run JEPA-MSAC paper/workflow reproduction stages.")
    parser.add_argument("--config", "-c", default="configs/pretraining/jepa_msac_s32_smoke.yaml")
    parser.add_argument("--stage", choices=STAGES, default="report")
    parser.add_argument("--dry-run", action="store_true", help="Audit config/data protocol and run only metadata/smoke checks.")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Stage 1 checkpoint for frozen Stage 2 heads.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory, default from config or outputs/analysis/jepa_msac.")
    parser.add_argument("--no-write", action="store_true", help="Return JSON without writing report artifacts.")
    return parser


def run_main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_jepa_msac(
        config_path=args.config,
        stage=args.stage,
        dry_run=bool(args.dry_run),
        pretrained_checkpoint=args.pretrained_checkpoint,
        output_dir=args.output_dir,
        write=not bool(args.no_write),
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
