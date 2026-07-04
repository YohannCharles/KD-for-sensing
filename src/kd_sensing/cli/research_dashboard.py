
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from kd_sensing.diagnostics.research_claim_harvester import (
    DEFAULT_LEDGER_DIR,
    build_dashboard_summary,
    ledger_records_from_candidates,
    render_dashboard_summary,
    write_jsonl_ledger,
    write_ledger_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only research claim dashboard.")
    parser.add_argument(
        "--outputs",
        action="append",
        default=None,
        help="Output root to index. Can be repeated. Defaults to outputs.",
    )
    parser.add_argument(
        "--logs",
        action="append",
        default=None,
        help="Log root or file to scan. Can be repeated. Defaults to logs.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        help="Artifact root for claim harvesting. Can be repeated. Defaults to --outputs.",
    )
    parser.add_argument("--project-root", default=".", help="Project root used for OpenSpec status.")
    parser.add_argument("--json", action="store_true", help="Print dashboard JSON instead of the text summary.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write dashboard JSON.")
    parser.add_argument("--write-ledger", action="store_true", help="Write a JSONL experiment ledger.")
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR, help="JSONL ledger directory.")
    parser.add_argument("--ledger-csv", type=Path, help="Optional CSV ledger export path.")
    parser.add_argument(
        "--stale-after-hours",
        type=float,
        default=12.0,
        help="Classify unfinished runs as stale after this many hours.",
    )
    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Skip process, GPU, memory, and swap resource snapshot collection.",
    )
    return parser


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = build_dashboard_summary(
        project_root=args.project_root,
        outputs=args.outputs or ["outputs"],
        logs=args.logs or ["logs"],
        scan_roots=args.scan_root or args.outputs or ["outputs"],
        include_resources=not args.no_resources,
        stale_after=dt.timedelta(hours=float(args.stale_after_hours)),
        now=None,
    )
    ledger_path = None
    ledger_csv_path = None
    records = ledger_records_from_candidates(summary.get("candidates", []))
    if args.write_ledger:
        ledger_path = write_jsonl_ledger(records, ledger_dir=args.ledger_dir, now=dt.datetime.now(dt.timezone.utc))
        summary.setdefault("metadata", {})["ledger_path"] = str(ledger_path)
    if args.ledger_csv:
        ledger_csv_path = write_ledger_csv(records, output_path=args.ledger_csv)
        summary.setdefault("metadata", {})["ledger_csv_path"] = str(ledger_csv_path)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_dashboard_summary(summary))
        if ledger_path:
            print(f"ledger: {ledger_path}")
        if ledger_csv_path:
            print(f"ledger_csv: {ledger_csv_path}")
        if args.output_json:
            print(f"dashboard_json: {args.output_json}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
