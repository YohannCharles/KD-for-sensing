import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from kd_sensing.diagnostics.research_claim_harvester import (
    build_dashboard_summary,
    render_dashboard_html,
)


DEFAULT_PREVIEW_DIR = Path("outputs/analysis/research_preview")
SCHEMA_VERSION = 1

REVIEWED_STATUS_MARKERS = ("reviewed", "accepted", "promoted", "official")
EXCLUDED_STATUS_MARKERS = (
    "candidate",
    "pending",
    "mock",
    "smoke",
    "historical",
    "upper-bound",
    "upper_bound",
    "not_comparable",
    "unverified",
    "blocked",
)

FIELD_ALIASES = {
    "method": ("method", "model", "model line", "method_name"),
    "claim_status": ("claim_status", "claim status", "status"),
    "caveat": ("caveat", "note", "notes", "warning", "warnings"),
    "comparability": ("comparability_status", "comparability", "comparable", "strict_comparability"),
    "reference": ("reference", "provenance", "source", "artifact_paths", "source_artifact"),
    "metric": ("metric", "metric_field", "target_metric"),
    "value": ("value", "mean", "score", "result"),
    "item": ("item", "check", "name", "requirement"),
}


def build_research_run_preview(
    *,
    project_root: str | Path = ".",
    outputs: Iterable[str | Path] = ("outputs",),
    logs: Iterable[str | Path] | None = ("logs",),
    output_dir: str | Path = DEFAULT_PREVIEW_DIR,
    evidence_inputs: dict[str, Iterable[str | Path]] | None = None,
    dashboard_summary: dict[str, Any] | None = None,
    include_resources: bool = True,
    run_checks: bool = False,
    check_runner: Callable[[list[str], Path], dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build a no-training research preview manifest and static dashboard."""

    generated_at = _format_dt(now or dt.datetime.now(dt.timezone.utc))
    root = Path(project_root).expanduser().resolve()
    preview_dir = Path(output_dir).expanduser()
    if not preview_dir.is_absolute():
        preview_dir = root / preview_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    summary = dashboard_summary or build_dashboard_summary(
        project_root=root,
        outputs=list(outputs),
        logs=list(logs) if logs is not None else None,
        scan_roots=list(outputs),
        include_resources=include_resources,
        now=now,
    )
    dashboard_html_path = preview_dir / "dashboard.html"
    dashboard_json_path = preview_dir / "dashboard_summary.json"
    dashboard_html_path.write_text(render_dashboard_html(summary), encoding="utf-8")
    dashboard_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    qa_inputs: dict[str, list[str | Path]] = {"html": [dashboard_html_path]}
    for kind, paths in (evidence_inputs or {}).items():
        qa_inputs.setdefault(kind, []).extend(paths)
    preview_qa = validate_evidence_preview(qa_inputs)

    budget_manifest = build_budget_manifest(
        budget or {},
        output_root=str(preview_dir),
        generated_at=generated_at,
    )
    budget_path = preview_dir / "budget_manifest.json"
    budget_path.write_text(json.dumps(budget_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    check_plan = default_check_plan(outputs=outputs, logs=logs)
    checks = [_planned_check(item) for item in check_plan]
    if run_checks:
        runner = check_runner or _run_check
        checks = []
        for item in check_plan:
            result = runner(item["command"], root)
            checks.append({**item, **result})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "metadata": {
            "workflow": "research_run_preview_loop",
            "project_root": str(root),
            "read_only": True,
            "candidate_only": True,
            "does_not_start_training": True,
            "does_not_read_real_dataset": True,
            "does_not_load_checkpoint": True,
            "does_not_write_training_artifacts": True,
        },
        "outputs": {
            "preview_dir": str(preview_dir),
            "dashboard_html": str(dashboard_html_path),
            "dashboard_json": str(dashboard_json_path),
            "budget_manifest": str(budget_path),
        },
        "checks": checks,
        "preview_qa": preview_qa,
        "budget": budget_manifest,
        "run_recipe": run_recipe(),
    }
    manifest_path = preview_dir / "preview_manifest.json"
    manifest["outputs"]["preview_manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_evidence_preview(evidence_inputs: dict[str, Iterable[str | Path]]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    for kind, paths in sorted(evidence_inputs.items()):
        for raw_path in paths:
            path = Path(raw_path)
            checked.append({"kind": kind, "path": str(path)})
            if not path.exists():
                issues.append(_issue("error", path, kind, "evidence file is missing"))
                continue
            text = path.read_text(encoding="utf-8")
            if kind == "html":
                _check_html(path, text, issues)
            elif kind in {"csv", "table"}:
                _check_csv_table(path, text, issues, kind=kind)
            elif kind in {"figure", "figure_data"}:
                _check_figure_data(path, text, issues)
            elif kind == "checklist":
                _check_checklist(path, text, issues)
            elif kind == "conclusion":
                _check_conclusion(path, text, issues)
            else:
                issues.append(_issue("warning", path, kind, f"unknown evidence kind: {kind}"))
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if error_count == 0 else "fail",
        "checked": checked,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def build_budget_manifest(
    values: dict[str, Any] | None = None,
    *,
    output_root: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    data = dict(values or {})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _format_dt(dt.datetime.now(dt.timezone.utc)),
        "workflow_id": data.get("workflow_id") or "research_run_preview_loop",
        "change_id": data.get("change_id") or "add-research-run-preview-loop",
        "mode": data.get("mode") or "smoke_dev_preview",
        "config_path": data.get("config_path"),
        "manifest_path": data.get("manifest_path"),
        "dataset_family": data.get("dataset_family") or "synthetic_or_unspecified",
        "reads_real_dataset": bool(data.get("reads_real_dataset", False)),
        "gpu": data.get("gpu") or "none for preview; explicit for long runs",
        "cpu": data.get("cpu") or "current smoke/dev CPU",
        "estimated_wall_time": data.get("estimated_wall_time") or "under 5 minutes for preview",
        "parallelism": data.get("parallelism") or 1,
        "output_root": str(output_root or data.get("output_root") or DEFAULT_PREVIEW_DIR),
        "checkpoint_plan": data.get("checkpoint_plan") or "none for preview",
        "cache_plan": data.get("cache_plan") or "read-only or none for preview",
        "fresh_eval_plan": data.get("fresh_eval_plan") or "none unless explicitly requested",
        "paper_export_plan": data.get("paper_export_plan") or "dry draft only, ignored output root",
        "stop_conditions": _as_list(data.get("stop_conditions")) or ["preview checks complete"],
        "artifacts_not_committed": True,
        "long_run": bool(data.get("long_run", False)),
    }
    manifest["validation"] = validate_budget_manifest(manifest)
    return manifest


def validate_budget_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    required = (
        "workflow_id",
        "change_id",
        "dataset_family",
        "reads_real_dataset",
        "gpu",
        "estimated_wall_time",
        "output_root",
        "checkpoint_plan",
        "cache_plan",
        "stop_conditions",
    )
    missing = [field for field in required if manifest.get(field) in (None, "", [])]
    if manifest.get("long_run"):
        for field in ("config_path", "manifest_path"):
            if not manifest.get(field):
                missing.append(f"{field} (one required for long_run)")
                break
        if str(manifest.get("checkpoint_plan", "")).lower() == "none for preview":
            missing.append("checkpoint_plan (long_run must declare write/read policy)")
        if str(manifest.get("gpu", "")).lower().startswith("none for preview"):
            missing.append("gpu (long_run must declare GPU/CPU plan)")
    return {
        "status": "pass" if not missing else "missing_fields",
        "missing_required_fields": list(dict.fromkeys(missing)),
        "artifacts_not_committed": manifest.get("artifacts_not_committed") is True,
    }


def default_check_plan(
    *,
    outputs: Iterable[str | Path] = ("outputs",),
    logs: Iterable[str | Path] | None = ("logs",),
) -> list[dict[str, Any]]:
    output_args = _repeat_args("--outputs", outputs)
    log_args = _repeat_args("--logs", logs or ())
    return [
        {
            "id": "openspec_validate_all",
            "description": "OpenSpec strict validation",
            "command": ["openspec", "validate", "--all", "--strict"],
            "side_effect": "read_only",
        },
        {
            "id": "architecture_quick_check",
            "description": "Architecture boundary quick check",
            "command": ["conda", "run", "-n", "kd_mm_beam", "pytest", "tests/test_architecture_boundaries.py", "-q"],
            "side_effect": "pytest_tmp_only",
        },
        {
            "id": "surface_doctor",
            "description": "Read-only project surface doctor",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "kd-sensing-project-surface-doctor",
                "--scope",
                "scripts",
                "--scope",
                "configs",
                "--scope",
                "hotspots",
                "--fail-on",
                "none",
            ],
            "side_effect": "read_only_stdout",
        },
        {
            "id": "run_index_json",
            "description": "Read-only run index summary",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "kd-sensing-runs",
                *output_args,
                *log_args,
                "--format",
                "json",
            ],
            "side_effect": "read_only_stdout",
        },
        {
            "id": "research_dashboard",
            "description": "Candidate-only research dashboard",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "kd-sensing-research-dashboard",
                *output_args,
                *log_args,
                "--json",
            ],
            "side_effect": "read_only_stdout",
        },
        {
            "id": "paper_table_consistency",
            "description": "Paper table draft consistency gate",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "kd-sensing-paper-export",
                "--input",
                "docs/result_claims_registry.md",
                "--output-dir",
                "outputs/paper_artifacts/current",
            ],
            "side_effect": "ignored_output_root",
        },
    ]


def run_recipe() -> dict[str, Any]:
    return {
        "smoke_dev": {
            "command": "conda run -n kd_mm_beam kd-sensing-research-preview --no-resources",
            "module_fallback": "conda run -n kd_mm_beam python -m kd_sensing.cli.research_preview --no-resources",
            "boundary": "no training, no real dataset read, no checkpoint load",
        },
        "gpu_full_training": {
            "requires_explicit_opt_in": True,
            "requires_budget_manifest": True,
            "forbidden_recipe_inputs": [
                "password",
                "token",
                "platform startup file edits",
                "checkpoint committed to source",
                "absolute private dataset path",
            ],
        },
        "package_cli_fallbacks": {
            "kd-sensing-runs": "python -m kd_sensing.cli.runs",
            "kd-sensing-research-dashboard": "python -m kd_sensing.cli.research_dashboard",
            "kd-sensing-paper-export": "python -m kd_sensing.cli.paper_artifact_export",
            "kd-sensing-project-surface-doctor": "python -m kd_sensing.cli.project_surface_doctor",
            "kd-sensing-train": "python -m kd_sensing.cli.train",
            "kd-sensing-evaluate": "python -m kd_sensing.cli.evaluate",
            "kd-sensing-preprocess": "python -m kd_sensing.cli.preprocess",
        },
    }


def render_preview_summary(manifest: dict[str, Any]) -> str:
    qa = manifest.get("preview_qa", {})
    budget = manifest.get("budget", {})
    outputs = manifest.get("outputs", {})
    planned = sum(1 for check in manifest.get("checks", []) if check.get("status") == "planned")
    failed = sum(1 for check in manifest.get("checks", []) if check.get("status") == "failed")
    return "\n".join(
        [
            "research run preview",
            f"preview_manifest: {outputs.get('preview_manifest')}",
            f"dashboard_html: {outputs.get('dashboard_html')}",
            f"qa_status: {qa.get('status')} errors={qa.get('error_count', 0)} warnings={qa.get('warning_count', 0)}",
            f"budget_status: {budget.get('validation', {}).get('status')}",
            f"checks: planned={planned} failed={failed}",
            "training_side_effects: false",
        ]
    )


def _check_html(path: Path, text: str, issues: list[dict[str, str]]) -> None:
    lower = text.lower()
    for marker in ("metadata", "claim readiness", "paper readiness"):
        if marker not in lower:
            issues.append(_issue("error", path, "html", f"missing dashboard section: {marker}"))
    if "candidate" not in lower and "pending" not in lower:
        issues.append(_issue("error", path, "html", "missing candidate/pending caveat"))
    if re.search(r"https?://|//cdn\.|<script\b", lower):
        issues.append(_issue("error", path, "html", "remote dependency or script tag is not allowed"))


def _check_csv_table(path: Path, text: str, issues: list[dict[str, str]], *, kind: str) -> None:
    rows = _csv_rows(text)
    if not rows:
        issues.append(_issue("error", path, kind, "table has no data rows"))
        return
    headers = _headers(rows)
    for field in ("method", "claim_status", "caveat", "comparability", "reference"):
        if _field_name(headers, field) is None:
            issues.append(_issue("error", path, kind, f"missing required field: {field}", field=field))
    for index, row in enumerate(rows, start=2):
        status = _cell(row, "claim_status").lower()
        caveat = _cell(row, "caveat")
        candidate_only = _truthy(_cell(row, "candidate_only")) or "candidate" in status
        comparable = _cell(row, "comparability").lower()
        if candidate_only and any(marker in status for marker in REVIEWED_STATUS_MARKERS):
            issues.append(_issue("error", path, kind, f"candidate-only row marked as reviewed at row {index}"))
        if any(marker in status for marker in EXCLUDED_STATUS_MARKERS) and not caveat:
            issues.append(_issue("error", path, kind, f"excluded-status row missing caveat at row {index}"))
        if "not_comparable" in comparable and any(marker in status for marker in REVIEWED_STATUS_MARKERS):
            issues.append(_issue("error", path, kind, f"not-comparable row marked as reviewed at row {index}"))


def _check_figure_data(path: Path, text: str, issues: list[dict[str, str]]) -> None:
    rows = _structured_rows(path, text)
    if not rows:
        issues.append(_issue("error", path, "figure_data", "figure data has no rows"))
        return
    headers = _headers(rows)
    for field in ("method", "metric", "caveat"):
        if _field_name(headers, field) is None:
            issues.append(_issue("error", path, "figure_data", f"missing required field: {field}", field=field))


def _check_checklist(path: Path, text: str, issues: list[dict[str, str]]) -> None:
    rows = _csv_rows(text) if path.suffix.lower() == ".csv" else []
    if rows:
        headers = _headers(rows)
        for field in ("item", "claim_status", "caveat"):
            if _field_name(headers, field) is None:
                issues.append(_issue("error", path, "checklist", f"missing required field: {field}", field=field))
        for index, row in enumerate(rows, start=2):
            status = _cell(row, "claim_status").lower()
            if any(marker in status for marker in ("pending", "incomplete")) and not _cell(row, "caveat"):
                issues.append(_issue("error", path, "checklist", f"pending checklist item missing caveat at row {index}"))
        return
    lower = text.lower()
    if any(marker in lower for marker in ("pending", "incomplete")) and "caveat" not in lower:
        issues.append(_issue("error", path, "checklist", "pending checklist text must include caveat"))


def _check_conclusion(path: Path, text: str, issues: list[dict[str, str]]) -> None:
    lower = text.lower()
    if "candidate" in lower and "candidate-only" not in lower:
        issues.append(_issue("warning", path, "conclusion", "candidate mention should preserve candidate-only caveat"))
    if any(marker in lower for marker in ("pending", "incomplete")) and "caveat" not in lower:
        issues.append(_issue("error", path, "conclusion", "pending/incomplete conclusion must include caveat"))
    if "candidate" in lower and "reviewed claim" in lower:
        issues.append(_issue("error", path, "conclusion", "candidate evidence cannot be called reviewed claim"))


def _structured_rows(path: Path, text: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("rows", "data", "figure_data", "conditions", "summary"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, dict)]
            return [payload]
    return _csv_rows(text)


def _csv_rows(text: str) -> list[dict[str, str]]:
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def _headers(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return tuple(headers)


def _field_name(headers: Iterable[str], field: str) -> str | None:
    aliases = {alias.lower() for alias in FIELD_ALIASES.get(field, (field,))}
    for header in headers:
        if str(header).strip().lower() in aliases:
            return header
    return None


def _cell(row: dict[str, Any], field: str) -> str:
    name = _field_name(row.keys(), field)
    value = row.get(name, "") if name else ""
    return "" if value is None else str(value).strip()


def _issue(severity: str, path: Path, check: str, message: str, *, field: str | None = None) -> dict[str, str]:
    issue = {
        "severity": severity,
        "path": str(path),
        "check": check,
        "message": message,
    }
    if field:
        issue["field"] = field
    return issue


def _planned_check(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "status": "planned", "returncode": None}


def _run_check(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False, timeout=300)
    except Exception as exc:
        return {"status": "failed", "returncode": -1, "stdout": "", "stderr": str(exc)}
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _repeat_args(flag: str, values: Iterable[str | Path]) -> list[str]:
    args: list[str] = []
    for value in values:
        args.extend([flag, str(value)])
    return args


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_dt(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat()
