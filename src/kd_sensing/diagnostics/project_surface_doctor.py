"""Read-only project surface diagnostics for scripts, configs, and hotspots."""

import ast
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from kd_sensing.config.canonical import (
    CANONICAL_SINGLE_MODALITIES,
    SNAPSHOT_MODE,
    VISION_POSITION_BASELINE_PRESETS,
)
from kd_sensing.config.io import load_config_source
from kd_sensing.utils.paths import project_root as resolve_project_root

DEFAULT_SCOPES = ("scripts", "configs", "hotspots")
SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
DOC_AUTHORITY_PATHS = (
    "README.md",
    "docs/agent_navigation.md",
    "docs/project_surface_inventory.md",
    "docs/maintainer_context_index.yaml",
    "pyproject.toml",
)
TRACKED_SCRIPT_ROOTS = ("scripts/", "tools/analysis/")
RUNTIME_OUTPUT_MARKERS = ("outputs/", "logs/", "outputs", "logs")
HIGH_RISK_CONFIG_TOKENS = {
    "hist_beam",
    "top8",
    "logits_kd",
    "rkd",
    "craf",
    "marf",
    "g2d",
    "multimodal_nf",
    "raymobtime",
    "raymobtime_s008",
    "bgam",
    "gps_lidar_bgam",
    "viewer_manifest",
    "amr_net_gps_image",
    "jepa_msac",
    "run_amr_net_gps_image",
    "run_jepa_msac",
    "gps_circular_soft_label",
    "mmw_sunny_modal15",
}


def build_project_surface_report(
    project_root: str | Path = ".",
    *,
    scopes: Iterable[str] | None = None,
    fail_on: str = "error",
) -> dict[str, Any]:
    root = resolve_project_root(Path(project_root).expanduser())
    selected_scopes = tuple(scopes or DEFAULT_SCOPES)
    invalid = sorted(set(selected_scopes) - set(DEFAULT_SCOPES))
    if invalid:
        raise ValueError(f"Unknown doctor scope(s): {', '.join(invalid)}")
    if fail_on not in {"none", *SEVERITY_ORDER}:
        raise ValueError("fail_on must be one of: none, info, warning, error")

    tracked = _git_ls_files(root)
    authority = _load_authority_sources(root)
    issues: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}

    if "scripts" in selected_scopes:
        sections["scripts"] = _doctor_scripts(root, tracked, authority, issues)
    if "configs" in selected_scopes:
        sections["configs"] = _doctor_configs(root, tracked, authority, issues)
    if "hotspots" in selected_scopes:
        sections["hotspots"] = _doctor_hotspots(root, tracked, authority, issues)

    summary = _summarize_issues(issues, fail_on=fail_on)
    return {
        "metadata": {
            "project_root": str(root),
            "scopes": list(selected_scopes),
            "read_only": True,
            "tracked_file_count": len(tracked),
            "authority_sources": sorted(authority),
            "scan_policy": {
                "tracked_files_only_for_surfaces": True,
                "excluded_roots": ["dataset/", "outputs/", "logs/", "cache/", "outputs/cache/"],
                "default_failure_level": fail_on,
            },
        },
        "summary": summary,
        "issues": issues,
        "sections": sections,
    }


def render_project_surface_report(report: dict[str, Any], *, format: str = "markdown") -> str:
    if format == "json":
        return json.dumps(report, indent=2, sort_keys=True)
    if format != "markdown":
        raise ValueError("format must be 'markdown' or 'json'")
    summary = report["summary"]
    lines = [
        "# Project Surface Doctor",
        "",
        f"- status: {summary['status']}",
        f"- scopes: {', '.join(report['metadata']['scopes'])}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        f"- infos: {summary['infos']}",
        "",
    ]
    sections = report.get("sections", {})
    if "scripts" in sections:
        scripts = sections["scripts"]
        lines.extend(
            [
                "## Scripts",
                "",
                f"- tracked entries: {scripts['tracked_count']}",
                f"- documented entries: {scripts['documented_count']}",
                f"- config references: {scripts['config_reference_count']}",
                "",
            ]
        )
    if "configs" in sections:
        configs = sections["configs"]
        family_counts = configs.get("family_counts", {})
        lines.extend(["## Configs", "", f"- tracked YAML: {configs['tracked_count']}"])
        for family, count in sorted(family_counts.items()):
            lines.append(f"- {family}: {count}")
        if configs.get("recipe_migration_candidates"):
            lines.append(f"- recipe migration candidates: {len(configs['recipe_migration_candidates'])}")
        lines.append(f"- virtual route patterns: {len(configs.get('virtual_routes', []))}")
        lines.append("")
    if "hotspots" in sections:
        hotspots = sections["hotspots"]
        action_counts = Counter(item.get("recommended_action", "unknown") for item in hotspots.get("entries", []))
        lines.extend(["## Hotspots", "", f"- registered entries: {hotspots['registered_count']}"])
        for action, count in sorted(action_counts.items()):
            lines.append(f"- {action}: {count}")
        lines.append("")
    if report.get("issues"):
        lines.extend(
            [
                "## Issues",
                "",
                "| severity | scope | kind | path | source | recommendation |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for issue in report["issues"]:
            lines.append(
                "| {severity} | {scope} | {kind} | `{path}` | {source} | {recommendation} |".format(
                    severity=issue["severity"],
                    scope=issue["scope"],
                    kind=issue["kind"],
                    path=issue.get("path", ""),
                    source=_markdown_cell(issue.get("source", "unavailable")),
                    recommendation=_markdown_cell(issue.get("recommendation", "")),
                )
            )
        lines.append("")
    else:
        lines.extend(["## Issues", "", "No issues at the selected failure/reporting levels.", ""])
    return "\n".join(lines).rstrip() + "\n"


def doctor_should_fail(report: dict[str, Any]) -> bool:
    return str(report.get("summary", {}).get("status")) == "fail"


def _doctor_scripts(
    root: Path,
    tracked: list[str],
    authority: dict[str, str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    script_paths = [
        rel
        for rel in tracked
        if rel.startswith(TRACKED_SCRIPT_ROOTS)
        and Path(rel).suffix in {".py", ".sh"}
        and not rel.startswith(("dataset/", "outputs/", "logs/"))
    ]
    entries = []
    config_reference_count = 0
    for rel in sorted(script_paths):
        path = root / rel
        text = _read_text(path)
        doc = _source_for_fragment(authority, f"`{rel}`")
        lifecycle = _lifecycle_for_fragment(authority, rel)
        config_refs = sorted(set(_config_references(text)))
        config_reference_count += len(config_refs)
        invalid_refs = []
        for config_ref in config_refs:
            status = _config_route_status(root, config_ref)
            if status["status"] == "missing":
                invalid_refs.append(config_ref)
                issues.append(
                    _issue(
                        scope="scripts",
                        severity="warning",
                        kind="missing_config_reference",
                        path=rel,
                        message=f"Script references missing config path {config_ref}.",
                        source=doc or "script literal",
                        recommendation="Update the local/manual runner reference or document why the generated config is expected.",
                        validation="conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope scripts --format json",
                    )
                )
        if doc is None:
            issues.append(
                _issue(
                    scope="scripts",
                    severity="error",
                    kind="unclassified_script",
                    path=rel,
                    message="Tracked script is not classified in inventory, docs, or OpenSpec authority text.",
                    source="docs/project_surface_inventory.md",
                    recommendation="Add lifecycle, owner, output boundary, and focused validation, or remove the duplicate entry.",
                    validation="conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope scripts",
                )
            )
        elif _script_may_write_outputs(text) and not _source_window_has_output_boundary(authority, rel):
            issues.append(
                _issue(
                    scope="scripts",
                    severity="warning",
                    kind="missing_output_boundary",
                    path=rel,
                    message="Script appears to write artifacts but its source classification lacks an explicit output boundary.",
                    source=doc,
                    recommendation="Document the ignored output/log root in the inventory or script lifecycle note.",
                    validation="conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q",
                )
            )
        if path.suffix == ".py" and _is_duplicate_cli_wrapper(text):
            issues.append(
                _issue(
                    scope="scripts",
                    severity="error",
                    kind="duplicate_thin_wrapper",
                    path=rel,
                    message="Python script looks like a thin wrapper around a package CLI.",
                    source=doc or "script source",
                    recommendation="Use the package console script directly unless a current spec explicitly allows the wrapper.",
                    validation="conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
                )
            )
        entries.append(
            {
                "path": rel,
                "lifecycle": lifecycle or "undocumented",
                "source": doc or "missing",
                "config_references": config_refs,
                "invalid_config_references": invalid_refs,
                "output_boundary_documented": _source_window_has_output_boundary(authority, rel),
            }
        )
    return {
        "tracked_count": len(script_paths),
        "documented_count": sum(1 for item in entries if item["source"] != "missing"),
        "config_reference_count": config_reference_count,
        "entries": entries,
    }


def _doctor_configs(
    root: Path,
    tracked: list[str],
    authority: dict[str, str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    config_paths = [rel for rel in tracked if rel.startswith("configs/") and Path(rel).suffix in {".yaml", ".yml"}]
    entries = []
    payloads: dict[str, dict[str, Any]] = {}
    for rel in sorted(config_paths):
        path = root / rel
        payload = _safe_load_yaml(path)
        payloads[rel] = payload
        classification = _classify_config(rel, authority)
        source = classification["source"]
        base_refs = _base_config_references(root, rel, payload)
        for ref in base_refs:
            if _config_route_status(root, ref)["status"] == "missing":
                issues.append(
                    _issue(
                        scope="configs",
                        severity="error",
                        kind="missing_base_config",
                        path=rel,
                        message=f"Config _base_ references missing path {ref}.",
                        source=source,
                        recommendation="Fix the _base_ path or migrate the config to a documented recipe without restoring retired routes.",
                        validation="conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q",
                    )
                )
        retired_hits = _retired_config_hits(rel, _read_text(path))
        for token in retired_hits:
            issues.append(
                _issue(
                    scope="configs",
                    severity="error",
                    kind="retired_token",
                    path=rel,
                    message=f"Config path or content contains retired token {token!r}.",
                    source="docs/maintainer_context_index.yaml",
                    recommendation="Remove the retired route reference; do not restore it as a virtual alias or entity YAML.",
                    validation="conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q",
                )
            )
        if classification["lifecycle"] == "unclassified":
            issues.append(
                _issue(
                    scope="configs",
                    severity="warning",
                    kind="unclassified_config",
                    path=rel,
                    message="Tracked config is not covered by a known family or direct documentation source.",
                    source="docs/project_surface_inventory.md",
                    recommendation="Add the config family/lifecycle and focused validation to inventory or OpenSpec.",
                    validation="conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope configs --format json",
                )
            )
        entries.append(
            {
                "path": rel,
                "family": classification["family"],
                "lifecycle": classification["lifecycle"],
                "run_class": classification["run_class"],
                "requires_real_data": classification["requires_real_data"],
                "default_output_boundary": classification["default_output_boundary"],
                "focused_validation": classification["focused_validation"],
                "source": source,
                "base_references": base_refs,
            }
        )
    candidates = _recipe_migration_candidates(payloads)
    return {
        "tracked_count": len(config_paths),
        "family_counts": dict(Counter(item["family"] for item in entries)),
        "entries": entries,
        "virtual_routes": _virtual_config_routes(),
        "recipe_migration_candidates": candidates,
    }


def _doctor_hotspots(
    root: Path,
    tracked: list[str],
    authority: dict[str, str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory = authority.get("docs/project_surface_inventory.md", "")
    registered = sorted(set(re.findall(r"`(src/kd_sensing/[^`]+?\.py)`", inventory)))
    entries = []
    for rel in registered:
        path = root / rel
        if not path.exists():
            continue
        text = _read_text(path)
        line_count = len(text.splitlines())
        window = _source_window(authority, rel)
        action = _hotspot_action(window)
        focused_tests = _validation_commands(window)
        budget = _hotspot_budget(window)
        if budget is not None and line_count > budget and action == "hard-budget":
            issues.append(
                _issue(
                    scope="hotspots",
                    severity="error",
                    kind="hard_budget_exceeded",
                    path=rel,
                    message=f"Public facade/hard-budget file has {line_count} lines, above budget {budget}.",
                    source="docs/project_surface_inventory.md",
                    recommendation="Move implementation back to the narrow owner or delete the low-value facade growth.",
                    validation="conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q",
                )
            )
        entries.append(
            {
                "path": rel,
                "line_count": line_count,
                "max_function_lines": _max_function_lines(path),
                "recommended_action": action,
                "reason": _hotspot_reason(window),
                "focused_tests": focused_tests,
                "line_count_is_trend_signal": True,
                "decision_uses_line_count_only": False,
                "source": "docs/project_surface_inventory.md",
            }
        )
    tracked_src = [rel for rel in tracked if rel.startswith("src/kd_sensing/") and rel.endswith(".py")]
    registered_set = set(registered)
    for rel in tracked_src:
        if rel in registered_set or not (root / rel).exists():
            continue
        line_count = len(_read_text(root / rel).splitlines())
        if line_count >= 900:
            issues.append(
                _issue(
                    scope="hotspots",
                    severity="warning",
                    kind="large_unregistered_owner",
                    path=rel,
                    message=f"Tracked source file has {line_count} lines and is not registered in hotspot inventory.",
                    source="docs/project_surface_inventory.md",
                    recommendation="Classify the owner as split/keep/monitor/accepted with focused validation before expanding it further.",
                    validation="conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q",
                )
            )
    return {
        "registered_count": len(entries),
        "entries": entries,
    }


def _issue(
    *,
    scope: str,
    severity: str,
    kind: str,
    path: str,
    message: str,
    source: str,
    recommendation: str,
    validation: str,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "severity": severity,
        "kind": kind,
        "path": path,
        "message": message,
        "source": source,
        "recommendation": recommendation,
        "validation": validation,
    }


def _summarize_issues(issues: list[dict[str, Any]], *, fail_on: str) -> dict[str, Any]:
    counts = Counter(issue["severity"] for issue in issues)
    threshold = None if fail_on == "none" else SEVERITY_ORDER[fail_on]
    failing = threshold is not None and any(SEVERITY_ORDER[issue["severity"]] >= threshold for issue in issues)
    return {
        "status": "fail" if failing else "pass",
        "fail_on": fail_on,
        "errors": counts.get("error", 0),
        "warnings": counts.get("warning", 0),
        "infos": counts.get("info", 0),
        "total_issues": len(issues),
    }


def _git_ls_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _load_authority_sources(root: Path) -> dict[str, str]:
    candidates = [root / rel for rel in DOC_AUTHORITY_PATHS]
    candidates.extend(sorted((root / "openspec/specs").glob("*/spec.md")))
    sources: dict[str, str] = {}
    for path in candidates:
        if path.exists() and path.is_file():
            rel = path.relative_to(root).as_posix()
            sources[rel] = _read_text(path)
    return sources


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _source_for_fragment(authority: dict[str, str], fragment: str) -> str | None:
    for rel, text in authority.items():
        index = text.find(fragment)
        if index < 0:
            continue
        line = text.count("\n", 0, index) + 1
        return f"{rel}:{line}"
    return None


def _source_window(authority: dict[str, str], fragment: str, *, radius: int = 3) -> str:
    windows = []
    for text in authority.values():
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if fragment in line:
                windows.append("\n".join(lines[max(0, index - radius) : index + radius + 1]))
    return "\n".join(windows)


def _source_window_has_output_boundary(authority: dict[str, str], rel_path: str) -> bool:
    window = _source_window(authority, rel_path, radius=4)
    return "输出" in window or any(marker in window for marker in RUNTIME_OUTPUT_MARKERS)


def _lifecycle_for_fragment(authority: dict[str, str], rel_path: str) -> str | None:
    window = _source_window(authority, rel_path, radius=2)
    match = re.search(r"属于\s+([^。\n]+)", window)
    if match:
        return match.group(1).strip()
    for marker in ("package_cli", "research_diagnostic", "local/manual", "dataset_preparation", "figure_helper"):
        if marker in window:
            return marker
    return "documented" if window else None


def _config_references(text: str) -> list[str]:
    pattern = re.compile(r"configs/[A-Za-z0-9_./-]+\.ya?ml")
    return [match.group(0).rstrip(".,)'\"") for match in pattern.finditer(text)]


def _config_route_status(root: Path, rel_path: str) -> dict[str, str]:
    path = root / rel_path
    if path.exists():
        return {"status": "file", "path": rel_path}
    try:
        source = load_config_source(path)
    except Exception as exc:  # noqa: BLE001 - report route resolution failures, do not mask them.
        return {"status": "missing", "path": rel_path, "error": str(exc)}
    return {"status": source.source_type, "path": rel_path}


def _script_may_write_outputs(text: str) -> bool:
    markers = (".write_text(", ".mkdir(", "csv.DictWriter", "json.dump", "torch.save", "fig.savefig", "output_dir")
    return any(marker in text for marker in markers)


def _is_duplicate_cli_wrapper(text: str) -> bool:
    if "kd_sensing.cli" not in text:
        return False
    nonblank = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if len(nonblank) > 45:
        return False
    tree = ast.parse(text)
    imports_cli = any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kd_sensing.cli.")
        for node in ast.walk(tree)
    )
    calls_main = "main(" in text or "console_main(" in text
    return imports_cli and calls_main


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(_read_text(path)) or {}
    return data if isinstance(data, dict) else {}


def _base_config_references(root: Path, rel_path: str, payload: dict[str, Any]) -> list[str]:
    base = payload.get("_base_")
    if base is None:
        return []
    values = base if isinstance(base, list) else [base]
    refs = []
    for item in values:
        base_path = Path(str(item))
        if not base_path.is_absolute():
            base_path = (root / rel_path).parent / base_path
        try:
            refs.append(base_path.resolve().relative_to(root).as_posix())
        except ValueError:
            refs.append(str(base_path))
    return refs


def _classify_config(rel_path: str, authority: dict[str, str]) -> dict[str, Any]:
    family = "unclassified"
    lifecycle = "unclassified"
    run_class = "unknown"
    requires_real_data = True
    output_boundary = "outputs/"
    validation = "conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q"
    if re.match(r"configs/(image|radar|gps|lidar|mmwave|csi)/(strong|lightweight|supervised)\.ya?ml$", rel_path):
        family, lifecycle, run_class = "canonical root", "current", "formal"
    elif rel_path.startswith("configs/fusion/experiments/"):
        family, lifecycle, run_class = "experiment reproduction", "local/manual", "local/manual"
    elif rel_path.startswith("configs/fusion/"):
        family, lifecycle, run_class = "canonical fusion", "current", "formal"
    elif rel_path.startswith("configs/diagnostics/"):
        family, lifecycle, run_class = "diagnostics", "current", "diagnostic"
        requires_real_data = False
        output_boundary = "outputs/analysis/"
    elif rel_path.startswith("configs/preprocess/"):
        family, lifecycle, run_class = "dataset preparation", "current", "local/manual"
        output_boundary = "dataset/ or outputs/cache/"
    elif rel_path.startswith("configs/baselines/") or rel_path.startswith("configs/pretraining/"):
        family, lifecycle, run_class = "baseline reproduction", "current", "local/manual"
    elif rel_path.startswith("configs/scene31/"):
        family, lifecycle, run_class = "scene31 local/manual", "local/manual", "local/manual"
    elif "hardening_matrix" in rel_path:
        family, lifecycle, run_class = "CSI experiment matrix", "current", "smoke/local"
    elif rel_path in {"configs/deepsense6g_gps_adapter_v2.yaml", "configs/mmw_town_gps_adapter_v2.yaml"}:
        family, lifecycle, run_class = "adapter workflow", "current", "diagnostic"
        output_boundary = "outputs/analysis/"
    direct_source = _source_for_fragment(authority, f"`{rel_path}`")
    if direct_source and lifecycle == "unclassified":
        lifecycle = "documented"
    return {
        "family": family,
        "lifecycle": lifecycle,
        "run_class": run_class,
        "requires_real_data": requires_real_data,
        "default_output_boundary": output_boundary,
        "focused_validation": validation,
        "source": direct_source or "docs/project_surface_inventory.md#配置生命周期分类",
    }


def _retired_config_hits(rel_path: str, text: str) -> list[str]:
    lowered_path = rel_path.lower()
    lowered_text = text.lower()
    hits = []
    for token in sorted(HIGH_RISK_CONFIG_TOKENS):
        lowered = token.lower()
        if lowered in lowered_path or lowered in lowered_text:
            hits.append(token)
    return hits


def _recipe_migration_candidates(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for rel, payload in payloads.items():
        if not (
            rel.startswith("configs/fusion/experiments/")
            or rel.startswith("configs/scene31/")
            or "hardening_matrix" in rel
        ):
            continue
        fingerprint = json.dumps(_semantic_config_fingerprint(payload), sort_keys=True, default=str)
        groups[fingerprint].append(rel)
    candidates = []
    for paths in groups.values():
        if len(paths) < 3:
            continue
        candidates.append(
            {
                "paths": sorted(paths),
                "reason": "Multiple tracked YAML files share the same high-level dataset/model/objective/training semantics.",
                "required_before_deletion": [
                    "preserve experiment name",
                    "preserve objective and dataset split",
                    "preserve model/loss/training/output/checkpoint semantics",
                    "cover with focused tests",
                ],
            }
        )
    return candidates


def _semantic_config_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    dataset = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
    model = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), dict) else {}
    training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
    loss = payload.get("loss", {}) if isinstance(payload.get("loss"), dict) else {}
    return {
        "dataset_type": dataset.get("type"),
        "dataset_scene": dataset.get("scene"),
        "train_scenes": dataset.get("train_scenes"),
        "eval_scenes": dataset.get("eval_scenes"),
        "model_type": primary.get("type"),
        "modalities": primary.get("modalities") or model.get("modalities"),
        "objective": payload.get("experiment", {}).get("objective") if isinstance(payload.get("experiment"), dict) else None,
        "loss_type": loss.get("type"),
        "epochs": training.get("epochs"),
    }


def _virtual_config_routes() -> list[dict[str, Any]]:
    routes = [
        {
            "pattern": "configs/fusion/<canonical_slug>_<strong|lightweight>.yaml",
            "lifecycle": "current",
            "source": "src/kd_sensing/config/canonical.py",
            "retired_guard": "KD aliases such as logits_kd/rkd are rejected.",
        }
    ]
    routes.extend(
        {
            "path": f"configs/{modality}/{SNAPSHOT_MODE}.yaml",
            "lifecycle": "current",
            "source": "src/kd_sensing/config/canonical.py",
        }
        for modality in CANONICAL_SINGLE_MODALITIES
    )
    routes.extend(
        {
            "path": f"configs/fusion/{preset}.yaml",
            "lifecycle": "current",
            "source": "src/kd_sensing/config/canonical.py",
        }
        for preset in VISION_POSITION_BASELINE_PRESETS
    )
    return routes


def _hotspot_action(window: str) -> str:
    lowered = window.lower()
    if "hard-budget" in lowered or "facade-budget" in lowered or "hard-fail" in lowered:
        return "hard-budget"
    if "split-next" in lowered:
        return "split"
    if "merge-candidate" in lowered or "consolidate" in lowered:
        return "merge"
    if "keep-and-test" in lowered:
        return "keep-and-test"
    if "right-size-accepted" in lowered or "accepted-size" in lowered:
        return "accepted-size"
    if "monitor" in lowered:
        return "monitor"
    return "review"


def _hotspot_reason(window: str) -> str:
    for line in window.splitlines():
        stripped = line.strip()
        if stripped and ("|" in stripped or "登记" in stripped or "预算" in stripped):
            return stripped[:240]
    return "Inventory entry supplies owner, public surface, and focused-test context."


def _hotspot_budget(window: str) -> int | None:
    match = re.search(r"预算\s*(\d+)", window)
    return int(match.group(1)) if match else None


def _validation_commands(text: str) -> list[str]:
    return sorted(set(re.findall(r"conda run -n kd_mm_beam pytest [^`\n]+?-q", text)))


def _max_function_lines(path: Path) -> int:
    try:
        tree = ast.parse(_read_text(path))
    except SyntaxError:
        return 0
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            spans.append(int(end) - int(node.lineno) + 1)
    return max(spans, default=0)


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
