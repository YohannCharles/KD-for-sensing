import argparse
import json
import sys
from pathlib import Path

from kd_sensing.baselines.tii_vlrg_transformer import run_reproduction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or import a TII VLRG Transformer reproduction manifest.")
    parser.add_argument("--config", "-c", default="configs/baselines/tii_vlrg_transformer_reproduction.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-repo", type=Path, default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--prediction-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="execute", action="store_false", help="Record external commands without executing them.")
    mode.add_argument("--execute", dest="execute", action="store_true", help="Run the recorded external commands and write logs.")
    parser.set_defaults(execute=False)
    return parser


def run_main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_reproduction(
        args.config,
        overrides={
            "source_repo": args.source_repo,
            "source_commit": args.source_commit,
            "checkpoint_path": args.checkpoint_path,
            "prediction_path": args.prediction_path,
            "metrics_path": args.metrics_path,
        },
        output_root=args.output_root,
        dry_run=not args.execute,
        execute=args.execute,
        manifest_output=args.manifest_output,
        summary_output=args.summary_output,
        command_args=tuple(sys.argv[1:] if argv is None else argv),
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
