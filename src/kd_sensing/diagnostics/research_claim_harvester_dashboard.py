from collections import Counter
import datetime as dt
import html
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from kd_sensing.diagnostics.run_index import build_run_index
from kd_sensing.diagnostics.paper_artifact_export import DEFAULT_EXCLUDED_STATUS_MARKERS
from kd_sensing.diagnostics.research_claim_harvester_base import (
    SCHEMA_VERSION,
    DashboardSummary,
    _ensure_utc,
    _format_dt,
    _utc_now,
)
from kd_sensing.diagnostics.research_claim_harvester_collectors import (
    build_claim_doctor_report,
    harvest_research_claims,
)


def build_dashboard_summary(
    *,
    project_root: str | Path = ".",
    outputs: str | Path | Iterable[str | Path] = "outputs",
    logs: str | Path | Iterable[str | Path] | None = "logs",
    scan_roots: str | Path | Iterable[str | Path] | None = None,
    include_resources: bool = True,
    stale_after: dt.timedelta | None = None,
    now: dt.datetime | None = None,
    run_index: dict[str, Any] | None = None,
    active_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generated = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    index = run_index or build_run_index(
        outputs=outputs,
        logs=logs,
        include_resources=include_resources,
        stale_after=stale_after or dt.timedelta(hours=12),
        now=now,
    )
    harvest = harvest_research_claims(scan_roots or outputs, run_index=index, now=now)
    candidates = harvest["candidates"]
    state_counts = Counter(str(run.get("state") or "unknown") for run in index.get("runs", []))
    claim_counts = Counter(str(candidate.get("comparability_status") or "needs_review") for candidate in candidates)
    changes = active_changes if active_changes is not None else collect_active_openspec_changes(project_root)
    doctor = build_claim_doctor_report(candidates=candidates, now=now)
    upgradable = [
        _candidate_brief(candidate)
        for candidate in candidates
        if candidate.get("comparability_status") == "strict"
    ][:10]
    hints = _dashboard_hints(index=index, candidates=candidates)
    summary = DashboardSummary(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "project_root": str(Path(project_root).expanduser().resolve()),
            "candidate_only": True,
            "read_only": True,
            "does_not_update_claim_registry": True,
        },
        active_changes=changes,
        run_state_counts=dict(sorted(state_counts.items())),
        resources=_resource_summary(index.get("resources", {})),
        claim_counts=dict(sorted(claim_counts.items())),
        paper_readiness=_paper_readiness(candidates=candidates, doctor=doctor),
        candidates=candidates,
        upgradable_candidates=upgradable,
        next_action_hints=hints,
        warnings=list(index.get("warnings", [])) + list(harvest.get("warnings", [])),
    )
    return summary.to_dict()

def render_dashboard_summary(summary: dict[str, Any]) -> str:
    active = summary.get("active_changes", [])
    active_text = ", ".join(str(item.get("name") or item.get("changeName")) for item in active) if active else "none"
    run_counts = ", ".join(f"{key}={value}" for key, value in summary.get("run_state_counts", {}).items()) or "none"
    claim_counts = ", ".join(f"{key}={value}" for key, value in summary.get("claim_counts", {}).items()) or "none"
    resources = summary.get("resources", {})
    lines = [
        "daily research dashboard",
        f"generated_at: {summary.get('metadata', {}).get('generated_at')}",
        f"active_changes: {active_text}",
        f"runs: {run_counts}",
        f"resources: gpus={resources.get('gpu_count', 0)} processes={resources.get('process_count', 0)}",
        f"claim_candidates: {claim_counts}",
        f"paper_readiness: {summary.get('paper_readiness', {}).get('status', 'unknown')}",
        f"upgradable_candidates: {len(summary.get('upgradable_candidates', []))}",
        "candidate_only: true",
    ]
    hints = summary.get("next_action_hints", [])
    if hints:
        lines.append("next_actions:")
        lines.extend(f"- {hint}" for hint in hints[:8])
    return "\n".join(lines)

def render_dashboard_html(summary: dict[str, Any] | None) -> str:
    payload = summary or {}
    metadata = _as_mapping(payload.get("metadata"))
    generated_at = metadata.get("generated_at") or "unknown"
    body = [
        _hero(metadata),
        _section(
            "Metadata",
            _definition_list(
                [
                    ("generated_at", generated_at),
                    ("schema_version", metadata.get("schema_version", "unknown")),
                    ("project_root", metadata.get("project_root", "unknown")),
                    ("read_only", metadata.get("read_only", True)),
                    ("candidate_only", metadata.get("candidate_only", True)),
                    ("does_not_update_claim_registry", metadata.get("does_not_update_claim_registry", True)),
                ]
            ),
        ),
        _section("Run States", _key_value_table(payload.get("run_state_counts"), empty="No indexed run states.")),
        _section("Active Changes", _active_changes_table(payload.get("active_changes"))),
        _section("Resources", _resources_block(payload.get("resources"))),
        _section("Claim Readiness", _claim_readiness_block(payload)),
        _section("Paper Readiness", _paper_readiness_block(payload.get("paper_readiness"))),
        _section("Warnings", _list_block(payload.get("warnings"), empty="No dashboard warnings.")),
        _section("Next Actions", _list_block(payload.get("next_action_hints"), empty="No next actions detected.")),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Daily Research Dashboard</title>",
            "<style>",
            _DASHBOARD_CSS,
            "</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            *body,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )

def escape_dashboard_html(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value)
    return html.escape(text, quote=True)

def collect_active_openspec_changes(project_root: str | Path = ".") -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["openspec", "list", "--json"],
            cwd=Path(project_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    raw_changes = payload.get("changes", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_changes, list):
        return []
    changes = []
    for item in raw_changes:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("changeName") or item.get("id")
        if not name:
            continue
        if item.get("status") in {"archived", "complete"}:
            continue
        changes.append({"name": name, "status": item.get("status") or item.get("state") or "active"})
    return changes

def _hero(metadata: dict[str, Any]) -> str:
    return (
        '<section class="hero">'
        "<div>"
        "<p>candidate-only evidence dashboard</p>"
        "<h1>Daily Research Dashboard</h1>"
        f"<span>generated_at: {escape_dashboard_html(metadata.get('generated_at') or 'unknown')}</span>"
        "</div>"
        '<strong class="badge">read-only</strong>'
        "</section>"
    )

def _section(title: str, content: str) -> str:
    return f'<section class="section"><h2>{escape_dashboard_html(title)}</h2>{content}</section>'

def _definition_list(items: Iterable[tuple[str, Any]]) -> str:
    rows = [
        f"<div><dt>{escape_dashboard_html(label)}</dt><dd>{escape_dashboard_html(value)}</dd></div>"
        for label, value in items
    ]
    return '<dl class="defs">' + "".join(rows) + "</dl>"

def _key_value_table(value: Any, *, empty: str) -> str:
    mapping = _as_mapping(value)
    rows = [(key, item) for key, item in sorted(mapping.items(), key=lambda item: str(item[0]))]
    return _table(("field", "value"), rows, empty=empty)

def _active_changes_table(value: Any) -> str:
    rows: list[tuple[Any, ...]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            rows.append((item.get("name") or item.get("changeName") or item.get("id") or "unknown", item.get("status") or item.get("state") or "active"))
        else:
            rows.append((item, "active"))
    return _table(("change", "status"), rows, empty="No active OpenSpec changes.")

def _resources_block(value: Any) -> str:
    resources = _as_mapping(value)
    return _definition_list(
        [
            ("gpu_available", resources.get("gpu_available", False)),
            ("gpu_count", resources.get("gpu_count", 0)),
            ("process_count", resources.get("process_count", 0)),
            ("memory", resources.get("memory", {})),
        ]
    )

def _claim_readiness_block(summary: dict[str, Any]) -> str:
    candidates = _as_list(summary.get("candidates"))
    upgradable = _as_list(summary.get("upgradable_candidates"))
    candidate_rows = [
        (
            candidate.get("candidate_id"),
            candidate.get("run_name"),
            candidate.get("method"),
            candidate.get("seed"),
            candidate.get("comparability_status"),
            candidate.get("claim_status"),
            candidate.get("candidate_only", True),
            _artifact_paths(candidate.get("artifact_paths")),
            "; ".join(str(item) for item in _as_list(candidate.get("next_action_hints"))),
        )
        for candidate in candidates[:20]
        if isinstance(candidate, dict)
    ]
    upgradable_rows = [
        (
            candidate.get("candidate_id") or candidate.get("claim_id"),
            candidate.get("run_name"),
            candidate.get("method"),
            candidate.get("comparability_status"),
            candidate.get("claim_status"),
            candidate.get("candidate_only", True),
            "; ".join(str(item) for item in _as_list(candidate.get("next_action_hints"))),
        )
        for candidate in upgradable[:10]
        if isinstance(candidate, dict)
    ]
    parts = [
        '<p class="caveat">All rows are candidate-only readiness signals. They are not reviewed paper claims and do not update docs/result_claims_registry.md.</p>',
        "<h3>Claim Status Counts</h3>",
        _key_value_table(summary.get("claim_counts"), empty="No claim candidates."),
        "<h3>Candidate Sample</h3>",
        _table(
            (
                "candidate_id",
                "run_name",
                "method",
                "seed",
                "comparability",
                "claim_status",
                "candidate_only",
                "artifact_paths",
                "next_actions",
            ),
            candidate_rows,
            empty="No claim candidates in this summary.",
        ),
        "<h3>Upgradable Candidates</h3>",
        _table(
            ("candidate_id", "run_name", "method", "comparability", "claim_status", "candidate_only", "next_actions"),
            upgradable_rows,
            empty="No upgradable candidates detected.",
        ),
    ]
    if len(candidates) > 20:
        parts.append(f'<p class="note">Showing 20 of {escape_dashboard_html(len(candidates))} candidates.</p>')
    return "".join(parts)

def _paper_readiness_block(value: Any) -> str:
    readiness = _as_mapping(value)
    gate = _as_mapping(readiness.get("paper_export_gate"))
    return "".join(
        [
            _definition_list(
                [
                    ("status", readiness.get("status", "unknown")),
                    ("pending_or_unverified_count", readiness.get("pending_or_unverified_count", 0)),
                    ("candidate_only_count", readiness.get("candidate_only_count", 0)),
                    ("upgradable_candidate_count", readiness.get("upgradable_candidate_count", 0)),
                    ("candidate_only_excluded", gate.get("candidate_only_excluded", True)),
                    ("main_table_hard_exclude_status_markers", gate.get("main_table_hard_exclude_status_markers", [])),
                ]
            ),
            "<h3>Missing Evidence</h3>",
            _key_value_table(readiness.get("missing_field_counts"), empty="No missing evidence counts."),
            "<h3>Paper Gate Next Actions</h3>",
            _list_block(readiness.get("next_action_hints"), empty="No paper readiness hints."),
        ]
    )

def _list_block(value: Any, *, empty: str) -> str:
    values = _as_list(value)
    if not values:
        return f'<p class="empty">{escape_dashboard_html(empty)}</p>'
    rows = "".join(f"<li>{escape_dashboard_html(item)}</li>" for item in values)
    return f'<ul class="items">{rows}</ul>'

def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]], *, empty: str) -> str:
    row_list = [tuple(row) for row in rows]
    if not row_list:
        return f'<p class="empty">{escape_dashboard_html(empty)}</p>'
    head = "".join(f"<th>{escape_dashboard_html(header)}</th>" for header in headers)
    body_rows = []
    for row in row_list:
        cells = "".join(f"<td>{escape_dashboard_html(item)}</td>" for item in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'

def _artifact_paths(value: Any) -> str:
    mapping = _as_mapping(value)
    if not mapping:
        return ""
    return "; ".join(f"{key}={item}" for key, item in sorted(mapping.items(), key=lambda item: str(item[0])))

def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []

def _dashboard_hints(*, index: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for state in ("failed", "killed", "waiting", "stale", "partial"):
        count = sum(1 for run in index.get("runs", []) if run.get("state") == state)
        if count:
            hints.append(f"review {count} {state} run(s) in the run index")
    for candidate in candidates:
        for hint in candidate.get("next_action_hints", []):
            if hint not in hints:
                hints.append(hint)
    if not any(candidate.get("comparability_status") == "strict" for candidate in candidates) and candidates:
        hints.append("no strict candidates yet; review missing strict fields before promoting claims")
    return hints

def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics", {})
    primary_name, primary_value = next(iter(metrics.items()), (None, None)) if metrics else (None, None)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "run_name": candidate.get("run_name"),
        "method": candidate.get("method"),
        "seed": candidate.get("seed"),
        "pattern": candidate.get("pattern"),
        "comparability_status": candidate.get("comparability_status"),
        "claim_status": candidate.get("claim_status"),
        "candidate_only": candidate.get("candidate_only", True),
        "primary_metric": {"name": primary_name, "value": primary_value},
        "next_action_hints": list(candidate.get("next_action_hints", [])),
    }

def _resource_summary(resources: dict[str, Any]) -> dict[str, Any]:
    gpus = resources.get("gpus", {}) if isinstance(resources.get("gpus"), dict) else {}
    return {
        "gpu_available": bool(gpus.get("available")),
        "gpu_count": len(gpus.get("devices", []) or []),
        "process_count": len(resources.get("processes", []) or []),
        "memory": resources.get("memory", {}),
    }

def _paper_readiness(*, candidates: list[dict[str, Any]], doctor: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(str(candidate.get("claim_status") or "draft").lower() for candidate in candidates)
    review_count = sum(
        count
        for status, count in status_counts.items()
        if any(marker in status for marker in ("pending", "unverified", "not_comparable", "draft"))
    )
    candidate_only_count = sum(1 for candidate in candidates if candidate.get("candidate_only", True))
    missing_counts = dict(doctor.get("missing_field_counts", {}))
    upgradable = list(doctor.get("upgradable_candidates", []))
    blocked = bool(review_count or candidate_only_count or missing_counts)
    return {
        "status": "blocked" if blocked else "ready_for_manual_review",
        "pending_or_unverified_count": review_count,
        "candidate_only_count": candidate_only_count,
        "missing_field_counts": missing_counts,
        "upgradable_candidate_count": len(upgradable),
        "upgradable_candidates": upgradable[:10],
        "paper_export_gate": {
            "main_table_hard_exclude_status_markers": list(DEFAULT_EXCLUDED_STATUS_MARKERS),
            "candidate_only_excluded": True,
        },
        "next_action_hints": list(doctor.get("next_action_hints", [])),
    }

_DASHBOARD_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f4ef;
  --ink: #1f2933;
  --muted: #5f6b7a;
  --line: #d8d4cb;
  --panel: #ffffff;
  --accent: #0f766e;
  --accent-weak: #e0f2f1;
  --warn: #8a4b00;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
}
.page {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 24px 0 40px;
}
.hero,
.section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 22px 24px;
}
.hero p,
.hero h1,
.hero span {
  margin: 0;
}
.hero p {
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 700;
  text-transform: uppercase;
}
.hero h1 {
  margin-top: 6px;
  font-size: clamp(1.75rem, 4vw, 2.5rem);
}
.hero span,
.note,
.empty {
  color: var(--muted);
}
.badge {
  border-radius: 999px;
  background: var(--accent-weak);
  color: var(--accent);
  padding: 6px 10px;
  white-space: nowrap;
}
.section {
  margin-top: 16px;
  padding: 18px 20px;
}
h2 {
  margin: 0 0 14px;
  font-size: 1.08rem;
}
h3 {
  margin: 18px 0 10px;
  font-size: 0.95rem;
}
.defs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin: 0;
}
.defs div {
  border-left: 3px solid var(--accent);
  padding-left: 10px;
}
dt {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
}
dd {
  margin: 3px 0 0;
  overflow-wrap: anywhere;
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}
th,
td {
  border-top: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
}
.items {
  margin: 0;
  padding-left: 20px;
}
.items li {
  margin: 6px 0;
}
.caveat {
  margin: 0 0 12px;
  border-left: 3px solid var(--warn);
  background: #fff7ed;
  color: var(--warn);
  padding: 10px 12px;
}
@media (max-width: 640px) {
  .page { width: min(100% - 20px, 1180px); padding-top: 10px; }
  .hero { display: block; }
  .badge { display: inline-block; margin-top: 12px; }
  .section { padding: 14px; }
}
"""
