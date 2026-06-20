import argparse
import datetime as dt
import json
from pathlib import Path

from kd_sensing.diagnostics.run_index import (
    RUN_STATES,
    RunIndexFilters,
    build_run_index,
    render_run_csv,
    render_run_table,
    write_run_index_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local KD sensing experiment runs.")
    parser.add_argument(
        "--outputs",
        action="append",
        default=None,
        help="Output root to scan. Can be repeated. Defaults to outputs.",
    )
    parser.add_argument(
        "--logs",
        action="append",
        default=None,
        help="Log root or file to scan. Can be repeated. Defaults to logs.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Output format: table, json, or csv.",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        help="Filter by run state. Can be repeated or comma-separated.",
    )
    parser.add_argument("--output", "-o", help="Optional path to write the rendered run index.")
    parser.add_argument("--dataset-family", help="Filter by dataset family.")
    parser.add_argument("--objective", help="Filter by experiment objective.")
    parser.add_argument("--run-name", help="Filter by run name substring.")
    parser.add_argument("--since", help="Only include runs updated at or after this ISO timestamp.")
    parser.add_argument("--until", help="Only include runs updated at or before this ISO timestamp.")
    parser.add_argument(
        "--stale-after-hours",
        type=float,
        default=12.0,
        help="Classify started runs without metrics as stale after this many hours.",
    )
    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Skip process, GPU, memory, and swap resource snapshot collection.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    states = _parse_states(args.state)
    filters = RunIndexFilters(
        states=states,
        dataset_family=args.dataset_family,
        objective=args.objective,
        run_name=args.run_name,
        since=_parse_iso_datetime(args.since),
        until=_parse_iso_datetime(args.until),
    )
    index = build_run_index(
        outputs=args.outputs or ["outputs"],
        logs=args.logs or ["logs"],
        filters=filters,
        include_resources=not args.no_resources,
        stale_after=dt.timedelta(hours=float(args.stale_after_hours)),
    )
    if args.output:
        target = write_run_index_output(index, path=Path(args.output), format=args.format)
        print(f"Wrote run index to {target}")
    elif args.format == "json":
        print(json.dumps(index, indent=2, sort_keys=True))
    elif args.format == "csv":
        print(render_run_csv(index), end="")
    else:
        print(render_run_table(index))
    return index


def console_main(argv: list[str] | None = None) -> None:
    main(argv)


def _parse_states(values: list[str]) -> tuple[str, ...]:
    states: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            state = item.strip()
            if not state:
                continue
            if state not in RUN_STATES:
                raise ValueError(f"Unknown run state {state!r}. Expected one of: {', '.join(sorted(RUN_STATES))}")
            states.append(state)
    return tuple(dict.fromkeys(states))


def _parse_iso_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


if __name__ == "__main__":
    main()
