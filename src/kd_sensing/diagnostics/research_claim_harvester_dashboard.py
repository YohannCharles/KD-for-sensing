from collections import Counter
import datetime as dt
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from kd_sensing.diagnostics.run_index import build_run_index
from kd_sensing.diagnostics.research_claim_harvester_base import (
    SCHEMA_VERSION,
    DashboardSummary,
    _ensure_utc,
    _format_dt,
    _utc_now,
)
from kd_sensing.diagnostics.research_claim_harvester_collectors import harvest_research_claims


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
        f"upgradable_candidates: {len(summary.get('upgradable_candidates', []))}",
        "candidate_only: true",
    ]
    hints = summary.get("next_action_hints", [])
    if hints:
        lines.append("next_actions:")
        lines.extend(f"- {hint}" for hint in hints[:8])
    return "\n".join(lines)

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
