import datetime as dt
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.utils.runtime_output_layout import is_default_skipped_partition

from kd_sensing.diagnostics.run_index_artifacts import (
    infer_run_state,
    summarize_artifacts,
    summarize_checkpoints,
    summarize_claim_harvester_fields,
    summarize_cleanup_context,
    summarize_config,
    summarize_metrics,
    summarize_run_dir,
    summarize_tensorboard,
    summarize_timestamps,
)
from kd_sensing.diagnostics.run_index_base import (
    DEFAULT_STALE_AFTER,
    RUN_STATES,
    RunIndexFilters,
    _as_list,
    _ensure_utc,
    _filters_to_dict,
    _format_dt,
    _matches_filters,
    _resolve_existing_or_requested,
    _utc_now,
)
from kd_sensing.diagnostics.run_index_render import (
    render_run_csv,
    render_run_table,
    write_run_index_output,
)
from kd_sensing.diagnostics.run_index_resources import (
    _empty_resources,
    collect_python_processes,
    collect_resource_snapshot,
)
from kd_sensing.diagnostics.run_index_scanner import (
    collect_log_records,
    discover_run_dirs,
    parse_failure_patterns,
)


def build_run_index(
    *,
    outputs: str | Path | Iterable[str | Path] = "outputs",
    logs: str | Path | Iterable[str | Path] | None = "logs",
    filters: RunIndexFilters | None = None,
    include_resources: bool = True,
    stale_after: dt.timedelta = DEFAULT_STALE_AFTER,
    now: dt.datetime | None = None,
    processes: list[dict[str, Any]] | None = None,
    resources: dict[str, Any] | None = None,
    include_legacy_containers: bool = False,
) -> dict[str, Any]:
    """Build a read-only index of local experiment run directories."""

    generated_at = _utc_now() if now is None else _ensure_utc(now)
    output_roots = [_resolve_existing_or_requested(path) for path in _as_list(outputs)]
    log_roots = [_resolve_existing_or_requested(path) for path in _as_list(logs)] if logs is not None else []
    filter_spec = filters or RunIndexFilters()
    process_records = processes
    if process_records is None:
        process_records = collect_python_processes() if include_resources else []
    resource_snapshot = resources
    if resource_snapshot is None:
        resource_snapshot = collect_resource_snapshot(process_records) if include_resources else _empty_resources(process_records)
    else:
        resource_snapshot = dict(resource_snapshot)
        resource_snapshot.setdefault("processes", process_records)
    process_records = list(resource_snapshot.get("processes", process_records))

    warnings: list[str] = []
    log_records = collect_log_records(log_roots, warnings=warnings)
    run_dirs: list[Path] = []
    for root in output_roots:
        if root.exists():
            run_dirs.extend(discover_run_dirs(root, warnings=warnings, include_legacy_containers=include_legacy_containers))
        else:
            warnings.append(f"outputs root does not exist: {root}")

    runs = [
        summarize_run_dir(
            run_dir,
            logs=log_records,
            processes=process_records,
            now=generated_at,
            stale_after=stale_after,
            warnings=warnings,
        )
        for run_dir in sorted(set(run_dirs), key=lambda path: path.as_posix())
    ]
    runs = [run for run in runs if _matches_filters(run, filter_spec)]
    return {
        "generated_at": _format_dt(generated_at),
        "roots": {
            "outputs": [str(path) for path in output_roots],
            "logs": [str(path) for path in log_roots],
            "explicit_non_run_partitions": [
                str(path) for path in output_roots if is_default_skipped_partition(path)
            ],
        },
        "filters": _filters_to_dict(filter_spec),
        "runs": runs,
        "resources": resource_snapshot,
        "warnings": warnings,
    }
