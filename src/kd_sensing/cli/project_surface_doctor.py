"""CLI for the read-only project surface doctor."""

import argparse
import sys
from pathlib import Path

from kd_sensing.diagnostics.project_surface_doctor import (
    AVAILABLE_SCOPES,
    DEFAULT_SCOPES,
    build_project_surface_report,
    doctor_should_fail,
    render_project_surface_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only project surface diagnostics for scripts, configs, and hotspots.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Repository root. Defaults to the current directory.")
    parser.add_argument(
        "--scope",
        action="append",
        choices=AVAILABLE_SCOPES,
        help="Doctor scope to run. Can be repeated. Defaults to scripts, configs, and hotspots.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report format printed to stdout. Defaults to issue-only summaries unless --dump-inventory is set.",
    )
    parser.add_argument(
        "--dump-inventory",
        action="store_true",
        help="Include full pass inventory sections and machine-readable entries for audit workflows.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("none", "info", "warning", "error"),
        default="error",
        help="Return non-zero when issues at or above this severity are present.",
    )
    return parser


def run(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = build_project_surface_report(
        args.project_root,
        scopes=args.scope,
        fail_on=args.fail_on,
    )
    rendered = render_project_surface_report(report, format=args.format, dump_inventory=args.dump_inventory)
    sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(argv)
    return 1 if doctor_should_fail(report) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
