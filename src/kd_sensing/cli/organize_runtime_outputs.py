from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from kd_sensing.diagnostics.runtime_artifact_cleanup import (
    apply_runtime_output_organize_manifest,
    build_runtime_output_organize_manifest,
    render_organize_summary,
    write_runtime_output_organize_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or apply a local runtime output organize manifest.",
    )
    parser.add_argument("--project-root", default=".", help="Project root used for path and git tracked-file checks.")
    parser.add_argument("--outputs-root", default="outputs", help="Outputs root to scan. Defaults to outputs.")
    parser.add_argument("--output", "-o", help="Manifest output path. Defaults to outputs/cleanup_manifests/.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply move/archive actions from an existing organize manifest.",
    )
    parser.add_argument("--manifest", help="Manifest path required for --execute mode.")
    parser.add_argument(
        "--confirm-organize",
        action="store_true",
        help="Required with --execute. Confirms that the organize manifest has been reviewed.",
    )
    parser.add_argument("--report", help="Execution report output path. Defaults next to the manifest.")
    parser.add_argument("--json", action="store_true", help="Print the manifest or execution report as JSON.")
    return parser


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute:
        if not args.manifest:
            parser.error("--execute requires --manifest")
        report = apply_runtime_output_organize_manifest(
            args.manifest,
            project_root=args.project_root,
            confirm_organize=args.confirm_organize,
            report_path=args.report,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            summary = report.get("summary", {})
            print(
                "runtime output organize execution report\n"
                f"moved: {summary.get('moved_count', 0)}\n"
                f"skipped: {summary.get('skipped_count', 0)}\n"
                f"failed: {summary.get('failed_count', 0)}\n"
                f"report: {report.get('metadata', {}).get('report_path')}"
            )
        return report

    command_args = tuple(sys.argv[1:] if argv is None else argv)
    manifest = build_runtime_output_organize_manifest(
        project_root=Path(args.project_root),
        outputs_root=Path(args.outputs_root),
        command_args=command_args,
    )
    target = write_runtime_output_organize_manifest(manifest, output_path=args.output or args.manifest)
    manifest.setdefault("metadata", {})["manifest_path"] = str(target)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(render_organize_summary(manifest))
        print(f"manifest: {target}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
