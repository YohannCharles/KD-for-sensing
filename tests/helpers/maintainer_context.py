from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml


MAINTAINER_CONTEXT_INDEX_REL_PATH = Path("docs") / "maintainer_context_index.yaml"
OPENSPEC_LIFECYCLE_ALLOWED = {"current", "supporting", "retired-tombstone"}
RUNTIME_ARTIFACT_PREFIXES = ("dataset/", "outputs/", "logs/", "cache/", ".pytest_cache/")
RUNTIME_ARTIFACT_SUFFIXES = (".pth", ".pt", ".ckpt")
ARCHITECTURE_BASELINE_REQUIRED_FIELDS = {
    "captured_at",
    "purpose",
    "measurement_scope",
    "codegraph",
    "ast",
    "interpretation",
}
HOTSPOT_ACTION_REQUIRED_FIELDS = {
    "priority",
    "status",
    "enforcement",
    "planned_action",
    "public_surface_policy",
    "rationale",
    "rollback_note",
    "validation_commands",
}
HOTSPOT_ACTION_CLUE_FIELDS = {
    "split_targets",
    "consolidation_targets",
    "accepted_size_rationale",
    "next_change",
}
HOTSPOT_WAVE_REQUIRED_FIELDS = {
    "id",
    "target_paths",
    "owner_module",
    "planned_action",
    "public_surface_policy",
    "rollback_note",
    "validation_commands",
}
ENTRYPOINT_OUTPUT_BOUNDARY_MARKERS = (
    "read-only",
    "stdout",
    "outputs/",
    "logs/",
    "cache",
    "checkpoint",
    "dataset",
    "docs/figures/",
    "explicit user path",
    "explicit local",
)


def _index_fail(message: str) -> None:
    raise AssertionError(
        f"{MAINTAINER_CONTEXT_INDEX_REL_PATH.as_posix()}: {message}. "
        "Fix: update the named index field or the referenced project source so they stay synchronized."
    )


def _index_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _index_fail(f"{label} must be a mapping")
    return value


def _index_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _index_fail(f"{label} must be a list")
    return value


def _index_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        _index_fail(f"{label} contains duplicate values: {duplicates}")


def _check_index_path(root: Path, rel_path: str, label: str) -> None:
    if rel_path.startswith(RUNTIME_ARTIFACT_PREFIXES) or rel_path.endswith(RUNTIME_ARTIFACT_SUFFIXES):
        _index_fail(f"{label} references local runtime artifact path {rel_path}")
    if "*" in rel_path or "<" in rel_path or rel_path == "retired history":
        return
    if not (root / rel_path).exists():
        _index_fail(f"{label} references missing path {rel_path}")


def _check_index_command(command: str, label: str) -> None:
    python_or_project_markers = ("pytest", "python ", "python -m ", "kd-sensing-")
    if any(marker in command for marker in python_or_project_markers):
        expected = "conda run -n kd_mm_beam "
        if not command.startswith(expected):
            _index_fail(f"{label} command must use kd_mm_beam: {command}")


def _check_owner_module(root: Path, module: str, label: str) -> None:
    if module != "kd_sensing" and not module.startswith("kd_sensing."):
        _index_fail(f"{label}.owner_module must be inside kd_sensing: {module}")
    module_path = Path("src") / Path(*module.split("."))
    module_file = (root / module_path).with_suffix(".py")
    package_file = root / module_path / "__init__.py"
    if not module_file.exists() and not package_file.exists():
        _index_fail(f"{label}.owner_module references missing module {module}")


def _check_entrypoint_metadata(
    root: Path,
    entry: dict[str, Any],
    label: str,
    *,
    require_owner_module: bool = False,
) -> None:
    responsibility = entry.get("responsibility")
    if not isinstance(responsibility, str) or not responsibility.strip():
        _index_fail(f"{label}.responsibility must be a non-empty string")

    output_boundary = entry.get("output_boundary")
    if not isinstance(output_boundary, str) or not output_boundary.strip():
        _index_fail(f"{label}.output_boundary must be a non-empty string")
    if not any(marker in output_boundary for marker in ENTRYPOINT_OUTPUT_BOUNDARY_MARKERS):
        _index_fail(f"{label}.output_boundary is not auditable: {output_boundary}")

    owner_module = entry.get("owner_module")
    owner_script = entry.get("owner_script")
    if bool(owner_module) == bool(owner_script):
        _index_fail(f"{label} must define exactly one of owner_module or owner_script")
    if require_owner_module and not owner_module:
        _index_fail(f"{label}.owner_module is required")
    if owner_module is not None:
        if not isinstance(owner_module, str) or not owner_module.strip():
            _index_fail(f"{label}.owner_module must be a non-empty string")
        _check_owner_module(root, owner_module, label)
    if owner_script is not None:
        if not isinstance(owner_script, str) or not owner_script.strip():
            _index_fail(f"{label}.owner_script must be a non-empty string")
        _check_index_path(root, owner_script, f"{label}.owner_script")

    retired_route_guard = entry.get("retired_route_guard")
    if retired_route_guard is not None and (
        not isinstance(retired_route_guard, str) or not retired_route_guard.strip()
    ):
        _index_fail(f"{label}.retired_route_guard must be a non-empty string when present")


def _check_positive_int(value: object, label: str) -> None:
    if not isinstance(value, int) or value <= 0:
        _index_fail(f"{label} must be a positive integer")


def _check_architecture_sizing_baseline(root: Path, value: object) -> None:
    baseline = _index_mapping(value, "governance.architecture_sizing_baseline")
    missing = sorted(ARCHITECTURE_BASELINE_REQUIRED_FIELDS - set(baseline))
    if missing:
        _index_fail(f"governance.architecture_sizing_baseline missing fields: {missing}")

    captured_at = baseline.get("captured_at")
    if not isinstance(captured_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", captured_at):
        _index_fail("governance.architecture_sizing_baseline.captured_at must be YYYY-MM-DD")

    purpose = baseline.get("purpose")
    if not isinstance(purpose, str) or "not standalone" not in purpose:
        _index_fail("governance.architecture_sizing_baseline.purpose must say counts are not standalone KPIs")

    measurement_scope = _index_mapping(
        baseline.get("measurement_scope"),
        "governance.architecture_sizing_baseline.measurement_scope",
    )
    includes = [str(path) for path in _index_list(measurement_scope.get("includes"), "architecture baseline includes")]
    excludes = [str(path) for path in _index_list(measurement_scope.get("excludes"), "architecture baseline excludes")]
    for rel_path in includes:
        _check_index_path(root, rel_path, "governance.architecture_sizing_baseline.measurement_scope.includes")
    required_excludes = {
        "dataset/",
        "outputs/",
        "logs/",
        "cache/",
        ".pytest_cache/",
        "*.pth",
        "*.pt",
        "*.ckpt",
    }
    missing_excludes = sorted(required_excludes - set(excludes))
    if missing_excludes:
        _index_fail(
            "governance.architecture_sizing_baseline.measurement_scope.excludes missing "
            f"local artifact patterns: {missing_excludes}"
        )

    codegraph = _index_mapping(baseline.get("codegraph"), "governance.architecture_sizing_baseline.codegraph")
    for key in ("files_indexed", "python_files", "total_nodes", "total_edges", "function_nodes", "import_nodes"):
        _check_positive_int(codegraph.get(key), f"governance.architecture_sizing_baseline.codegraph.{key}")

    ast_baseline = _index_mapping(baseline.get("ast"), "governance.architecture_sizing_baseline.ast")
    for key in ("python_files", "function_defs", "import_statements"):
        _check_positive_int(ast_baseline.get(key), f"governance.architecture_sizing_baseline.ast.{key}")
    if codegraph["python_files"] != ast_baseline["python_files"]:
        _index_fail("architecture baseline CodeGraph and AST python file counts must match")

    distribution = _index_list(
        ast_baseline.get("source_tree_distribution"),
        "governance.architecture_sizing_baseline.ast.source_tree_distribution",
    )
    roots = {str(item.get("root")) for item in distribution if isinstance(item, dict)}
    if {"src", "tests", "scripts"} - roots:
        _index_fail("architecture baseline source_tree_distribution must include src, tests, and scripts")
    for item in distribution:
        entry = _index_mapping(item, "architecture baseline source_tree_distribution[]")
        root_name = str(entry.get("root"))
        _check_index_path(root, f"{root_name}/", "architecture baseline source_tree_distribution.root")
        for key in ("python_files", "function_defs", "import_statements"):
            _check_positive_int(entry.get(key), f"architecture baseline source_tree_distribution.{root_name}.{key}")

    major_subpackages = _index_list(
        ast_baseline.get("major_subpackages"),
        "governance.architecture_sizing_baseline.ast.major_subpackages",
    )
    if len(major_subpackages) < 6:
        _index_fail("architecture baseline must list major src/kd_sensing subpackages")
    for item in major_subpackages:
        entry = _index_mapping(item, "architecture baseline major_subpackages[]")
        rel_path = str(entry.get("path"))
        if not rel_path.startswith("src/kd_sensing/"):
            _index_fail(f"architecture baseline major subpackage must be under src/kd_sensing: {rel_path}")
        _check_index_path(root, rel_path, "architecture baseline major_subpackages.path")
        for key in ("python_files", "function_defs", "import_statements"):
            _check_positive_int(entry.get(key), f"architecture baseline major_subpackages.{rel_path}.{key}")

    interpretation = _index_list(
        baseline.get("interpretation"),
        "governance.architecture_sizing_baseline.interpretation",
    )
    interpretation_text = "\n".join(str(item) for item in interpretation)
    if "trend signals only" not in interpretation_text or "Local datasets" not in interpretation_text:
        _index_fail("architecture baseline interpretation must explain trend-only counts and local artifact exclusion")


def _check_hotspot_action_metadata(
    entry: dict[str, Any],
    label: str,
    *,
    priority_values: set[str],
    status_values: set[str],
    enforcement_values: set[str],
    planned_action_values: set[str],
    public_surface_policy_values: set[str],
) -> None:
    missing = sorted(HOTSPOT_ACTION_REQUIRED_FIELDS - set(entry))
    if missing:
        _index_fail(f"{label} missing hotspot action metadata fields: {missing}")

    priority = str(entry.get("priority"))
    if priority not in priority_values:
        _index_fail(f"{label}.priority has illegal value {priority!r}")
    status = str(entry.get("status"))
    if status not in status_values:
        _index_fail(f"{label}.status has illegal value {status!r}")

    enforcement = str(entry.get("enforcement"))
    if enforcement not in enforcement_values:
        _index_fail(f"{label}.enforcement has illegal value {enforcement!r}")

    planned_action = str(entry.get("planned_action"))
    if planned_action not in planned_action_values:
        _index_fail(f"{label}.planned_action has illegal value {planned_action!r}")

    public_surface_policy = str(entry.get("public_surface_policy"))
    if public_surface_policy not in public_surface_policy_values:
        _index_fail(f"{label}.public_surface_policy has illegal value {public_surface_policy!r}")

    if not any(field in entry and entry[field] not in (None, "", []) for field in HOTSPOT_ACTION_CLUE_FIELDS):
        _index_fail(
            f"{label} must include one of {sorted(HOTSPOT_ACTION_CLUE_FIELDS)} "
            "so agents can choose split, consolidate, accepted-size, or next-change action"
        )

    for field in ("split_targets", "consolidation_targets"):
        if field not in entry or entry[field] is None:
            continue
        targets = _index_list(entry.get(field), f"{label}.{field}")
        if not targets:
            _index_fail(f"{label}.{field} must not be empty when present")
        for target in targets:
            if not isinstance(target, str) or not target.strip():
                _index_fail(f"{label}.{field} must contain non-empty strings")

    accepted_size_rationale = entry.get("accepted_size_rationale")
    if accepted_size_rationale is not None and (
        not isinstance(accepted_size_rationale, str) or not accepted_size_rationale.strip()
    ):
        _index_fail(f"{label}.accepted_size_rationale must be a non-empty string when present")
    if status == "right-size-accepted" and accepted_size_rationale is None:
        _index_fail(f"{label}.accepted_size_rationale is required for right-size-accepted owner")
    if planned_action == "accepted-size" and accepted_size_rationale is None:
        _index_fail(f"{label}.accepted_size_rationale is required for accepted-size action")
    if status == "merge-candidate" or planned_action == "consolidate":
        if not entry.get("consolidation_targets"):
            _index_fail(f"{label}.consolidation_targets is required for merge/consolidate action")

    rollback_note = entry.get("rollback_note")
    if not isinstance(rollback_note, str) or not rollback_note.strip():
        _index_fail(f"{label}.rollback_note must be a non-empty string")

    headroom = entry.get("headroom_lines", 0)
    if not isinstance(headroom, int) or headroom < 0:
        _index_fail(f"{label}.headroom_lines must be a non-negative integer when present")
    if enforcement == "hard-fail" and headroom != 0:
        _index_fail(f"{label}.headroom_lines must be 0 for hard-fail enforcement")
    if enforcement == "headroom" and headroom <= 0:
        _index_fail(f"{label}.headroom_lines must be positive for headroom enforcement")

    rationale = entry.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        _index_fail(f"{label}.rationale must be a non-empty string")

    commands = _index_list(entry.get("validation_commands"), f"{label}.validation_commands")
    if not commands:
        _index_fail(f"{label}.validation_commands must not be empty")
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            _index_fail(f"{label}.validation_commands must contain non-empty strings")
        _check_index_command(command, f"{label}.validation_commands")

    if "next_change" in entry and entry["next_change"] is not None:
        next_change = entry["next_change"]
        if not isinstance(next_change, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", next_change):
            _index_fail(f"{label}.next_change must be a kebab-case change id")


def _check_hotspot_remediation_wave(
    root: Path,
    entry: dict[str, Any],
    label: str,
    *,
    planned_action_values: set[str],
    public_surface_policy_values: set[str],
) -> None:
    missing = sorted(HOTSPOT_WAVE_REQUIRED_FIELDS - set(entry))
    if missing:
        _index_fail(f"{label} missing remediation wave fields: {missing}")

    wave_id = entry.get("id")
    if not isinstance(wave_id, str) or not re.fullmatch(r"wave-[a-z0-9][a-z0-9-]*", wave_id):
        _index_fail(f"{label}.id must be a wave-* kebab-case id")

    target_paths = _index_list(entry.get("target_paths"), f"{label}.target_paths")
    if not target_paths:
        _index_fail(f"{label}.target_paths must not be empty")
    for rel_path in target_paths:
        _check_index_path(root, str(rel_path), f"{label}.target_paths")

    owner_module = entry.get("owner_module")
    if not isinstance(owner_module, str) or not owner_module.strip():
        _index_fail(f"{label}.owner_module must be a non-empty string")
    _check_owner_module(root, owner_module, label)

    planned_action = str(entry.get("planned_action"))
    if planned_action not in planned_action_values:
        _index_fail(f"{label}.planned_action has illegal value {planned_action!r}")

    public_surface_policy = str(entry.get("public_surface_policy"))
    if public_surface_policy not in public_surface_policy_values:
        _index_fail(f"{label}.public_surface_policy has illegal value {public_surface_policy!r}")

    rollback_note = entry.get("rollback_note")
    if not isinstance(rollback_note, str) or not rollback_note.strip():
        _index_fail(f"{label}.rollback_note must be a non-empty string")

    commands = _index_list(entry.get("validation_commands"), f"{label}.validation_commands")
    if not commands:
        _index_fail(f"{label}.validation_commands must not be empty")
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            _index_fail(f"{label}.validation_commands must contain non-empty strings")
        _check_index_command(command, f"{label}.validation_commands")

    if "consolidation_targets" in entry and entry["consolidation_targets"] is not None:
        targets = _index_list(entry["consolidation_targets"], f"{label}.consolidation_targets")
        if not targets:
            _index_fail(f"{label}.consolidation_targets must not be empty when present")
    if "accepted_size_rationale" in entry and (
        not isinstance(entry["accepted_size_rationale"], str) or not entry["accepted_size_rationale"].strip()
    ):
        _index_fail(f"{label}.accepted_size_rationale must be a non-empty string when present")


def _check_hotspot_inventory_marker(markers: set[str], marker: str, label: str) -> None:
    if marker == "__init__":
        if marker in markers or any(value.endswith(".__init__") for value in markers):
            return
    elif marker in markers:
        return
    _index_fail(f"{label} is not listed in governance.hotspots.inventory_markers")


def _strip_toml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_project_scripts(pyproject_text: str) -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    saw_scripts = False
    for line_number, raw_line in enumerate(pyproject_text.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            saw_scripts = saw_scripts or in_scripts
            continue
        if not in_scripts:
            continue
        if "=" not in line:
            _index_fail(f"pyproject.toml:[project.scripts] line {line_number} is not key = value")
        raw_name, raw_target = line.split("=", 1)
        name = raw_name.strip().strip("\"'")
        try:
            target = ast.literal_eval(raw_target.strip())
        except (SyntaxError, ValueError) as exc:
            _index_fail(f"pyproject.toml:[project.scripts].{name} target is not parseable: {exc}")
        if not isinstance(target, str):
            _index_fail(f"pyproject.toml:[project.scripts].{name} target must be a string")
        if name in scripts:
            _index_fail(f"pyproject.toml:[project.scripts].{name} is declared more than once")
        scripts[name] = target
    if not saw_scripts:
        _index_fail("pyproject.toml is missing [project.scripts]")
    return scripts


def load_pyproject_scripts(root: Path) -> dict[str, str]:
    return _parse_project_scripts((root / "pyproject.toml").read_text(encoding="utf-8"))


def assert_pyproject_scripts_match_index(root: Path, index: dict[str, Any]) -> None:
    entrypoints = _index_mapping(
        _index_mapping(index.get("governance"), "governance").get("entrypoints"),
        "governance.entrypoints",
    )
    package_cli = _index_list(entrypoints.get("package_cli"), "governance.entrypoints.package_cli")
    pyproject_scripts = load_pyproject_scripts(root)
    index_scripts: dict[str, str] = {}
    errors: list[str] = []

    for item in package_cli:
        entry = _index_mapping(item, "governance.entrypoints.package_cli[]")
        name = str(entry.get("name"))
        target = str(entry.get("target"))
        index_scripts[name] = target

    for name, target in sorted(index_scripts.items()):
        pyproject_target = pyproject_scripts.get(name)
        if pyproject_target is None:
            errors.append(
                f"governance.entrypoints.package_cli.{name} is stale; restore [project.scripts].{name} "
                "or remove the index entry"
            )
        elif pyproject_target != target:
            errors.append(
                f"governance.entrypoints.package_cli.{name}.target={target!r} does not match "
                f"pyproject.toml [project.scripts].{name}={pyproject_target!r}"
            )

    missing_from_index = sorted(set(pyproject_scripts) - set(index_scripts))
    for name in missing_from_index:
        errors.append(
            f"pyproject.toml [project.scripts].{name} is missing from governance.entrypoints.package_cli; "
            "register name, target, and lifecycle"
        )

    if errors:
        _index_fail("package CLI and pyproject.toml are out of sync: " + "; ".join(errors))


def _validate_maintainer_context_index(root: Path, data: dict[str, Any]) -> None:
    required_top_level = {"schema_version", "kind", "purpose", "routing", "governance", "references"}
    missing = sorted(required_top_level - set(data))
    if missing:
        _index_fail(f"missing top-level sections: {missing}")
    if data.get("kind") != "maintainer_context_index":
        _index_fail("kind must be maintainer_context_index")
    if data.get("schema_version") != 1:
        _index_fail("schema_version must be 1")

    purpose = _index_mapping(data["purpose"], "purpose")
    runtime_contract = _index_mapping(purpose.get("runtime_contract"), "purpose.runtime_contract")
    if runtime_contract.get("imported_by_runtime") is not False:
        _index_fail("purpose.runtime_contract.imported_by_runtime must be false")
    if runtime_contract.get("no_runtime_side_effects") is not True:
        _index_fail("purpose.runtime_contract.no_runtime_side_effects must be true")

    routing = _index_mapping(data["routing"], "routing")
    task_types = _index_list(routing.get("task_types"), "routing.task_types")
    required_task_ids = {
        "model_forward_registry",
        "data_batch_contract",
        "config_virtual_config",
        "cli_scripts",
        "diagnostics_viewer",
        "runtime_outputs_cache_cleanup",
        "openspec_artifact",
        "documentation_lifecycle",
    }
    task_ids = [str(item.get("id")) for item in task_types if isinstance(item, dict)]
    _index_unique(task_ids, "routing.task_types.id")
    missing_task_ids = sorted(required_task_ids - set(task_ids))
    if missing_task_ids:
        _index_fail(f"routing.task_types missing required ids: {missing_task_ids}")
    for item in task_types:
        task = _index_mapping(item, "routing.task_types[]")
        task_id = str(task.get("id"))
        if task.get("lifecycle") not in OPENSPEC_LIFECYCLE_ALLOWED:
            _index_fail(f"routing.task_types.{task_id}.lifecycle has illegal value {task.get('lifecycle')!r}")
        for key in ("read_first", "primary_paths"):
            for rel_path in _index_list(task.get(key), f"routing.task_types.{task_id}.{key}"):
                _check_index_path(root, str(rel_path), f"routing.task_types.{task_id}.{key}")
        for command in _index_list(task.get("validation_commands"), f"routing.task_types.{task_id}.validation_commands"):
            _check_index_command(str(command), f"routing.task_types.{task_id}.validation_commands")

    governance = _index_mapping(data["governance"], "governance")
    _check_architecture_sizing_baseline(root, governance.get("architecture_sizing_baseline"))

    entrypoints = _index_mapping(governance.get("entrypoints"), "governance.entrypoints")
    lifecycle_values = set(_index_list(entrypoints.get("lifecycle_values"), "governance.entrypoints.lifecycle_values"))
    for section in ("python_allowlist", "shell_allowlist"):
        entries = _index_list(entrypoints.get(section), f"governance.entrypoints.{section}")
        paths = []
        for item in entries:
            entry = _index_mapping(item, f"governance.entrypoints.{section}[]")
            path = str(entry.get("path"))
            lifecycle = entry.get("lifecycle")
            paths.append(path)
            if lifecycle not in lifecycle_values:
                _index_fail(f"governance.entrypoints.{section}.{path}.lifecycle has unknown value {lifecycle!r}")
            _check_index_path(root, path, f"governance.entrypoints.{section}.{path}")
            _check_entrypoint_metadata(
                root,
                entry,
                f"governance.entrypoints.{section}.{path}",
                require_owner_module=lifecycle == "thin_cli_alias",
            )
        _index_unique(paths, f"governance.entrypoints.{section}.path")

    package_cli = _index_list(entrypoints.get("package_cli"), "governance.entrypoints.package_cli")
    package_names = []
    for item in package_cli:
        entry = _index_mapping(item, "governance.entrypoints.package_cli[]")
        name = str(entry.get("name"))
        target = str(entry.get("target"))
        package_names.append(name)
        if entry.get("lifecycle") not in lifecycle_values:
            _index_fail(f"governance.entrypoints.package_cli.{name}.lifecycle has unknown value {entry.get('lifecycle')!r}")
        _check_entrypoint_metadata(
            root,
            entry,
            f"governance.entrypoints.package_cli.{name}",
            require_owner_module=True,
        )
        module = target.split(":", 1)[0]
        _check_index_path(root, f"src/{module.replace('.', '/')}.py", f"governance.entrypoints.package_cli.{name}")
    _index_unique(package_names, "governance.entrypoints.package_cli.name")
    assert_pyproject_scripts_match_index(root, data)

    configs = _index_mapping(governance.get("configs"), "governance.configs")
    fusion_paths = [str(path) for path in _index_list(configs.get("fusion_root_allowlist"), "governance.configs.fusion_root_allowlist")]
    for rel_path in fusion_paths:
        _check_index_path(root, rel_path, "governance.configs.fusion_root_allowlist")
    _index_unique(fusion_paths, "governance.configs.fusion_root_allowlist")
    retired_paths = [
        str(path)
        for path in _index_list(
            configs.get("retired_generated_fusion_configs"),
            "governance.configs.retired_generated_fusion_configs",
        )
    ]
    _index_unique(retired_paths, "governance.configs.retired_generated_fusion_configs")
    for rel_path in retired_paths:
        if (root / rel_path).exists():
            _index_fail(f"governance.configs.retired_generated_fusion_configs reintroduced {rel_path}")
    _index_unique(
        [str(marker) for marker in _index_list(configs.get("lifecycle_markers"), "governance.configs.lifecycle_markers")],
        "governance.configs.lifecycle_markers",
    )

    models = _index_mapping(governance.get("models"), "governance.models")
    _index_unique(
        [str(name) for name in _index_list(models.get("registration_allowlist"), "governance.models.registration_allowlist")],
        "governance.models.registration_allowlist",
    )

    batch_runtime = _index_mapping(governance.get("batch_runtime"), "governance.batch_runtime")
    batch_entries = _index_list(batch_runtime.get("function_allowlist"), "governance.batch_runtime.function_allowlist")
    batch_paths = []
    for item in batch_entries:
        entry = _index_mapping(item, "governance.batch_runtime.function_allowlist[]")
        path = str(entry.get("path"))
        batch_paths.append(path)
        _check_index_path(root, path, "governance.batch_runtime.function_allowlist.path")
        _index_unique(
            [str(name) for name in _index_list(entry.get("functions"), f"governance.batch_runtime.{path}.functions")],
            f"governance.batch_runtime.{path}.functions",
        )
    _index_unique(batch_paths, "governance.batch_runtime.function_allowlist.path")

    hotspots = _index_mapping(governance.get("hotspots"), "governance.hotspots")
    global_limits = _index_mapping(hotspots.get("global_limits"), "governance.hotspots.global_limits")
    for key in ("long_function_lines", "long_class_lines"):
        if not isinstance(global_limits.get(key), int) or int(global_limits[key]) <= 0:
            _index_fail(f"governance.hotspots.global_limits.{key} must be a positive integer")
    action_metadata = _index_mapping(hotspots.get("action_metadata"), "governance.hotspots.action_metadata")
    priority_value_list = [
        str(value)
        for value in _index_list(
            action_metadata.get("priority_values"),
            "governance.hotspots.action_metadata.priority_values",
        )
    ]
    status_value_list = [
        str(value)
        for value in _index_list(
            action_metadata.get("status_values"),
            "governance.hotspots.action_metadata.status_values",
        )
    ]
    enforcement_value_list = [
        str(value)
        for value in _index_list(
            action_metadata.get("enforcement_values"),
            "governance.hotspots.action_metadata.enforcement_values",
        )
    ]
    planned_action_value_list = [
        str(value)
        for value in _index_list(
            action_metadata.get("planned_action_values"),
            "governance.hotspots.action_metadata.planned_action_values",
        )
    ]
    public_surface_policy_value_list = [
        str(value)
        for value in _index_list(
            action_metadata.get("public_surface_policy_values"),
            "governance.hotspots.action_metadata.public_surface_policy_values",
        )
    ]
    _index_unique(priority_value_list, "governance.hotspots.action_metadata.priority_values")
    _index_unique(status_value_list, "governance.hotspots.action_metadata.status_values")
    _index_unique(enforcement_value_list, "governance.hotspots.action_metadata.enforcement_values")
    _index_unique(planned_action_value_list, "governance.hotspots.action_metadata.planned_action_values")
    _index_unique(
        public_surface_policy_value_list,
        "governance.hotspots.action_metadata.public_surface_policy_values",
    )
    priority_values = set(priority_value_list)
    status_values = set(status_value_list)
    enforcement_values = set(enforcement_value_list)
    planned_action_values = set(planned_action_value_list)
    public_surface_policy_values = set(public_surface_policy_value_list)
    if not priority_values:
        _index_fail("governance.hotspots.action_metadata.priority_values must not be empty")
    if not status_values:
        _index_fail("governance.hotspots.action_metadata.status_values must not be empty")
    if not enforcement_values:
        _index_fail("governance.hotspots.action_metadata.enforcement_values must not be empty")
    if not planned_action_values:
        _index_fail("governance.hotspots.action_metadata.planned_action_values must not be empty")
    if not public_surface_policy_values:
        _index_fail("governance.hotspots.action_metadata.public_surface_policy_values must not be empty")
    for value in priority_values:
        if not re.fullmatch(r"P[0-9]", value):
            _index_fail(f"governance.hotspots.action_metadata.priority_values has illegal value {value!r}")
    for label, values in (
        ("status_values", status_values),
        ("enforcement_values", enforcement_values),
        ("planned_action_values", planned_action_values),
        ("public_surface_policy_values", public_surface_policy_values),
    ):
        for value in values:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
                _index_fail(f"governance.hotspots.action_metadata.{label} has illegal value {value!r}")
    for required_status in ("facade-budget", "right-size-accepted", "merge-candidate"):
        if required_status not in status_values:
            _index_fail(f"governance.hotspots.action_metadata.status_values missing {required_status}")
    for required_action in ("split", "consolidate", "keep-and-test", "owner-facade", "hard-budget", "accepted-size"):
        if required_action not in planned_action_values:
            _index_fail(f"governance.hotspots.action_metadata.planned_action_values missing {required_action}")
    for required_policy in ("keep-public-import", "thin-owner", "no-public-surface", "remove-internal-only"):
        if required_policy not in public_surface_policy_values:
            _index_fail(
                f"governance.hotspots.action_metadata.public_surface_policy_values missing {required_policy}"
            )
    inventory_marker_list = [
        str(marker)
        for marker in _index_list(
            hotspots.get("inventory_markers"),
            "governance.hotspots.inventory_markers",
        )
    ]
    _index_unique(inventory_marker_list, "governance.hotspots.inventory_markers")
    inventory_markers = set(inventory_marker_list)
    symbol_budget_keys = []
    for item in _index_list(hotspots.get("symbol_budgets"), "governance.hotspots.symbol_budgets"):
        entry = _index_mapping(item, "governance.hotspots.symbol_budgets[]")
        path = str(entry.get("path"))
        symbol = str(entry.get("symbol"))
        kind = str(entry.get("kind"))
        label = f"governance.hotspots.symbol_budgets {path}:{symbol}"
        symbol_budget_keys.append(f"{path}:{symbol}:{kind}")
        _check_index_path(root, path, "governance.hotspots.symbol_budgets.path")
        if kind not in {"class", "function"}:
            _index_fail(f"governance.hotspots.symbol_budgets.{path}:{symbol}.kind has illegal value {kind!r}")
        if not isinstance(entry.get("max_lines"), int) or int(entry["max_lines"]) <= 0:
            _index_fail(f"governance.hotspots.symbol_budgets.{path}:{symbol}.max_lines must be positive")
        _check_hotspot_action_metadata(
            entry,
            label,
            priority_values=priority_values,
            status_values=status_values,
            enforcement_values=enforcement_values,
            planned_action_values=planned_action_values,
            public_surface_policy_values=public_surface_policy_values,
        )
        _check_hotspot_inventory_marker(inventory_markers, path, f"{label}.path")
        _check_hotspot_inventory_marker(inventory_markers, symbol, f"{label}.symbol")
    _index_unique(symbol_budget_keys, "governance.hotspots.symbol_budgets")
    file_budget_paths = []
    for item in _index_list(hotspots.get("file_budgets"), "governance.hotspots.file_budgets"):
        entry = _index_mapping(item, "governance.hotspots.file_budgets[]")
        path = str(entry.get("path"))
        label = f"governance.hotspots.file_budgets {path}"
        file_budget_paths.append(path)
        _check_index_path(root, path, "governance.hotspots.file_budgets.path")
        if not isinstance(entry.get("max_lines"), int) or int(entry["max_lines"]) <= 0:
            _index_fail(f"governance.hotspots.file_budgets.{path}.max_lines must be positive")
        _check_hotspot_action_metadata(
            entry,
            label,
            priority_values=priority_values,
            status_values=status_values,
            enforcement_values=enforcement_values,
            planned_action_values=planned_action_values,
            public_surface_policy_values=public_surface_policy_values,
        )
        _check_hotspot_inventory_marker(inventory_markers, path, f"{label}.path")
    _index_unique(file_budget_paths, "governance.hotspots.file_budgets.path")

    wave_ids = []
    for item in _index_list(hotspots.get("remediation_waves"), "governance.hotspots.remediation_waves"):
        entry = _index_mapping(item, "governance.hotspots.remediation_waves[]")
        wave_id = str(entry.get("id"))
        wave_ids.append(wave_id)
        _check_hotspot_remediation_wave(
            root,
            entry,
            f"governance.hotspots.remediation_waves.{wave_id}",
            planned_action_values=planned_action_values,
            public_surface_policy_values=public_surface_policy_values,
        )
    _index_unique(wave_ids, "governance.hotspots.remediation_waves.id")

    health_checks = _index_mapping(governance.get("health_checks"), "governance.health_checks")
    commands = [str(command) for command in _index_list(health_checks.get("quick_commands"), "governance.health_checks.quick_commands")]
    _index_unique(commands, "governance.health_checks.quick_commands")
    for command in commands:
        _check_index_command(command, "governance.health_checks.quick_commands")

    retired_routes = _index_mapping(governance.get("retired_routes"), "governance.retired_routes")
    for key in ("config_tokens", "text_markers"):
        values = [str(value) for value in _index_list(retired_routes.get(key), f"governance.retired_routes.{key}")]
        _index_unique(values, f"governance.retired_routes.{key}")

    references = _index_mapping(data["references"], "references")
    for key, values in references.items():
        for rel_path in _index_list(values, f"references.{key}"):
            _check_index_path(root, str(rel_path), f"references.{key}")


def load_maintainer_context_index(root: Path) -> dict[str, Any]:
    index_path = root / MAINTAINER_CONTEXT_INDEX_REL_PATH
    if not index_path.exists():
        _index_fail("file is missing; create docs/maintainer_context_index.yaml")
    try:
        data = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _index_fail(f"YAML is not parseable: {exc}")
    data = _index_mapping(data, "root")
    _validate_maintainer_context_index(root, data)
    return data


def entrypoint_allowlists(index: dict[str, Any]) -> dict[str, object]:
    entrypoints = index["governance"]["entrypoints"]
    return {
        "python": {item["path"]: item["lifecycle"] for item in entrypoints["python_allowlist"]},
        "shell": {item["path"]: item["lifecycle"] for item in entrypoints["shell_allowlist"]},
        "python_entries": {item["path"]: item for item in entrypoints["python_allowlist"]},
        "shell_entries": {item["path"]: item for item in entrypoints["shell_allowlist"]},
        "package_cli": {item["name"]: item for item in entrypoints["package_cli"]},
        "lifecycles": set(entrypoints["lifecycle_values"]),
    }


def config_allowlists(index: dict[str, Any]) -> dict[str, object]:
    configs = index["governance"]["configs"]
    return {
        "fusion_root_yaml": set(configs["fusion_root_allowlist"]),
        "retired_generated_fusion": set(configs["retired_generated_fusion_configs"]),
        "lifecycle_markers": tuple(configs["lifecycle_markers"]),
    }


def model_registration_allowlist(index: dict[str, Any]) -> set[str]:
    return set(index["governance"]["models"]["registration_allowlist"])


def batch_runtime_function_allowlist(index: dict[str, Any]) -> dict[str, set[str]]:
    return {
        item["path"]: set(item["functions"])
        for item in index["governance"]["batch_runtime"]["function_allowlist"]
    }


def hotspot_budgets(index: dict[str, Any]) -> dict[str, object]:
    hotspots = index["governance"]["hotspots"]
    return {
        "symbol": {
            (item["path"], item["symbol"], item["kind"]): item["max_lines"]
            for item in hotspots["symbol_budgets"]
        },
        "file": {item["path"]: item["max_lines"] for item in hotspots["file_budgets"]},
        "symbol_metadata": {
            (item["path"], item["symbol"], item["kind"]): item
            for item in hotspots["symbol_budgets"]
        },
        "file_metadata": {item["path"]: item for item in hotspots["file_budgets"]},
        "inventory_markers": tuple(hotspots["inventory_markers"]),
        "long_function_limit": hotspots["global_limits"]["long_function_lines"],
        "long_class_limit": hotspots["global_limits"]["long_class_lines"],
        "action_metadata": hotspots["action_metadata"],
        "remediation_waves": tuple(hotspots["remediation_waves"]),
    }


def health_check_commands(index: dict[str, Any]) -> tuple[str, ...]:
    return tuple(index["governance"]["health_checks"]["quick_commands"])


def retired_route_tokens(index: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    retired_routes = index["governance"]["retired_routes"]
    return {
        "config": tuple(retired_routes["config_tokens"]),
        "text": tuple(retired_routes["text_markers"]),
    }
