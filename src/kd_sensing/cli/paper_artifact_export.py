import argparse
import json
from pathlib import Path

from kd_sensing.diagnostics.paper_artifact_export import DEFAULT_OUTPUT_DIR, export_paper_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reviewed claim rows into paper table and figure-data drafts.")
    parser.add_argument("--input", action="append", required=True, type=Path, help="Claim/ledger/summary JSON, CSV, or Markdown table.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Ignored output directory for paper drafts.")
    parser.add_argument("--table-name", default="main_results", help="Base name for the main table drafts.")
    parser.add_argument(
        "--include-status",
        action="append",
        default=[],
        help="Status marker to also write into explicit diagnostic appendix outputs. Can be repeated or comma-separated.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full export manifest JSON.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    manifest = export_paper_artifacts(
        args.input,
        output_dir=args.output_dir,
        include_statuses=_split_markers(args.include_status),
        table_name=args.table_name,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            "paper export complete\n"
            f"manifest: {manifest['manifest_path']}\n"
            f"main rows: {manifest['filter']['main_row_count']}\n"
            f"appendix rows: {manifest['filter']['appendix_row_count']}"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


def _split_markers(values: list[str]) -> list[str]:
    markers: list[str] = []
    for value in values:
        markers.extend(item.strip() for item in str(value).split(",") if item.strip())
    return list(dict.fromkeys(markers))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
