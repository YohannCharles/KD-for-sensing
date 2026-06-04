from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

from kd_sensing.diagnostics.runtime_artifact_cleanup import (
    apply_cleanup_manifest,
    build_cleanup_manifest,
    render_cleanup_summary,
    write_cleanup_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or apply a local runtime artifact cleanup manifest.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Scan root for dry-run manifest generation. Can be repeated. Defaults to outputs, logs, cache, and .pytest_cache.",
    )
    parser.add_argument("--project-root", default=".", help="Project root used for path and git tracked-file checks.")
    parser.add_argument("--output", "-o", help="Manifest output path. Defaults to outputs/cleanup_manifests/.")
    parser.add_argument(
        "--stale-after-hours",
        type=float,
        default=12.0,
        help="Classify unfinished runs as stale after this many hours.",
    )
    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Skip live process, GPU, memory, and swap resource collection during dry-run scanning.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Apply/delete candidates from an existing manifest instead of generating a dry-run manifest.",
    )
    parser.add_argument(
        "--manifest",
        help="Manifest path required for --delete mode; accepted as the dry-run output path when --delete is not set.",
    )
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Required with --delete. Confirms that the manifest has been reviewed.",
    )
    parser.add_argument("--report", help="Delete report output path. Defaults next to the manifest.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the manifest or delete report as JSON after completion.",
    )
    return parser


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delete:
        if not args.manifest:
            parser.error("--delete requires --manifest")
        report = apply_cleanup_manifest(
            args.manifest,
            project_root=args.project_root,
            confirm_delete=args.confirm_delete,
            report_path=args.report,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = report.get("summary", {})
            print(
                "runtime cleanup delete report\n"
                f"deleted: {summary.get('deleted_count', 0)}\n"
                f"skipped: {summary.get('skipped_count', 0)}\n"
                f"failed: {summary.get('failed_count', 0)}\n"
                f"report: {report.get('metadata', {}).get('report_path')}"
            )
        return report

    command_args = tuple(sys.argv[1:] if argv is None else argv)
    manifest = build_cleanup_manifest(
        project_root=Path(args.project_root),
        scan_roots=args.root,
        include_resources=not args.no_resources,
        stale_after=dt.timedelta(hours=float(args.stale_after_hours)),
        command_args=command_args,
    )
    target = write_cleanup_manifest(manifest, output_path=args.output or args.manifest)
    manifest.setdefault("metadata", {})["manifest_path"] = str(target)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(render_cleanup_summary(manifest))
        print(f"manifest: {target}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
