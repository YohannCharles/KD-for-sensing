from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from tests.helpers.maintainer_context import (  # noqa: E402
    MAINTAINER_CONTEXT_INDEX_REL_PATH,
    OPENSPEC_LIFECYCLE_ALLOWED,
    assert_pyproject_scripts_match_index,
    batch_runtime_function_allowlist,
    config_allowlists,
    entrypoint_allowlists,
    health_check_commands,
    hotspot_budgets,
    load_maintainer_context_index,
    model_registration_allowlist,
    retired_route_tokens,
)
from kd_sensing.modalities import (  # noqa: E402
    MODALITY_ORDER,
    batch_input_keys_for_modalities,
    dataset_flags_for_modalities,
    normalize_modalities,
)
from kd_sensing.config.defaults import DEFAULT_CONFIG  # noqa: E402
from kd_sensing.utils.runtime_output_layout import canonical_runtime_partitions  # noqa: E402

MAINTAINER_CONTEXT_INDEX_PATH = ROOT / MAINTAINER_CONTEXT_INDEX_REL_PATH
LEGACY_MODEL_RETIREMENT_FIXTURE = ROOT / "tests" / "fixtures" / "legacy_model_registry_retirement.yaml"
MAINTAINER_CONTEXT_INDEX = load_maintainer_context_index(ROOT)
ENTRYPOINT_ALLOWLISTS = entrypoint_allowlists(MAINTAINER_CONTEXT_INDEX)
CONFIG_ALLOWLISTS = config_allowlists(MAINTAINER_CONTEXT_INDEX)
HOTSPOT_BUDGETS = hotspot_budgets(MAINTAINER_CONTEXT_INDEX)
RETIRED_ROUTE_TOKENS = retired_route_tokens(MAINTAINER_CONTEXT_INDEX)

PYTHON_ENTRYPOINT_ALLOWLIST = ENTRYPOINT_ALLOWLISTS["python"]
SHELL_ORCHESTRATION_ALLOWLIST = ENTRYPOINT_ALLOWLISTS["shell"]
PYTHON_ENTRYPOINT_METADATA = ENTRYPOINT_ALLOWLISTS["python_entries"]
PACKAGE_CLI_METADATA = ENTRYPOINT_ALLOWLISTS["package_cli"]
ENTRYPOINT_LIFECYCLES = ENTRYPOINT_ALLOWLISTS["lifecycles"]
RETIRED_GENERATED_FUSION_CONFIGS = CONFIG_ALLOWLISTS["retired_generated_fusion"]
FUSION_ROOT_YAML_ALLOWLIST = CONFIG_ALLOWLISTS["fusion_root_yaml"]
EXISTING_MODEL_REGISTRATION_ALLOWLIST = model_registration_allowlist(MAINTAINER_CONTEXT_INDEX)
BATCH_RUNTIME_FUNCTION_ALLOWLIST = batch_runtime_function_allowlist(MAINTAINER_CONTEXT_INDEX)
GOVERNANCE_SCAN_PREFIXES = (
    "AGENTS.md",
    "README.md",
    "configs/",
    "docs/",
    "openspec/specs/",
    "pyproject.toml",
    "scripts/",
    "src/",
    "tests/",
    "tools/analysis/",
)
GOVERNANCE_SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".sh"}

HEALTH_CHECK_COMMANDS = health_check_commands(MAINTAINER_CONTEXT_INDEX)
CONFIG_LIFECYCLE_MARKERS = CONFIG_ALLOWLISTS["lifecycle_markers"]
HOTSPOT_SYMBOL_BUDGETS = HOTSPOT_BUDGETS["symbol"]
HOTSPOT_FILE_BUDGETS = HOTSPOT_BUDGETS["file"]
HOTSPOT_SYMBOL_METADATA = HOTSPOT_BUDGETS["symbol_metadata"]
HOTSPOT_FILE_METADATA = HOTSPOT_BUDGETS["file_metadata"]
HOTSPOT_ACTION_METADATA = HOTSPOT_BUDGETS["action_metadata"]
HOTSPOT_REMEDIATION_WAVES = HOTSPOT_BUDGETS["remediation_waves"]
REQUIRED_HOTSPOT_INVENTORY_MARKERS = HOTSPOT_BUDGETS["inventory_markers"]
LONG_FUNCTION_LIMIT = HOTSPOT_BUDGETS["long_function_limit"]
LONG_CLASS_LIMIT = HOTSPOT_BUDGETS["long_class_limit"]
RETIRED_CONFIG_TOKENS = RETIRED_ROUTE_TOKENS["config"]
CONFIG_REFERENCE_RE = re.compile(r"configs/[A-Za-z0-9_./-]+\.yaml")
CONFIG_REFERENCE_SCAN_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "openspec" / "specs",
    ROOT / "openspec" / "changes" / "strengthen-project-health-guardrails" / "specs",
)
HISTORICAL_DOCS_WITH_ARCHIVED_CONFIGS = {
    "docs/p3_v7_multisource_crossroad_analysis.md",
}
NON_CURRENT_CONFIG_CONTEXT_MARKERS = (
    "旧",
    "退役",
    "不再",
    "不存在",
    "缺失",
    "拒绝",
    "失败",
    "历史",
    "migration",
    "retired",
    "removed",
    "no longer",
    "MUST not",
)
ROOT_DOCUMENT_LIFECYCLE_EXCLUSIONS = set()
CURRENT_DOCS_TO_CHECK_FOR_RETIRED_RECOMMENDATIONS = (
    ROOT / "README.md",
    ROOT / "docs" / "experiment_matrix.md",
    ROOT / "docs" / "extension_guide.md",
    ROOT / "docs" / "training_throughput.md",
)
RETIRED_ROUTE_TEXT_MARKERS = RETIRED_ROUTE_TOKENS["text"]
RETIRED_ROUTE_CLASSIFICATION_MARKERS = (
    "退役",
    "旧",
    "不再",
    "拒绝",
    "历史",
    "fail",
    "removed",
    "retired",
    "Retired",
    "migration",
)
AGENT_NAVIGATION_MARKERS = (
    "权威来源",
    "任务路由",
    "docs/maintainer_context_index.yaml",
    "机器可读",
    "OpenSpec capability lifecycle",
    "current",
    "supporting",
    "retired-tombstone",
    "generated metadata",
    "ignored runtime artifacts",
    ".pytest_cache/v/cache/lastfailed",
    "OpenSpec archive",
    "active change",
    "virtual config",
    "virtual configs",
    "retired research lines",
    "kd_mm_beam",
    "src/kd_sensing.egg-info/SOURCES.txt",
    "entry_points.txt",
    "dataset/",
    "outputs/",
    "logs/",
    "checkpoint",
    "openspec list --json",
    "openspec status --change <change>",
)
OPENSPEC_LIFECYCLE_HEADING = "## OpenSpec capability lifecycle 分类"
RETIREMENT_WORDING_MARKERS = (
    "退役",
    "不属于当前支持",
    "不属于当前",
    "拒绝",
    "历史",
    "migration guard",
    "防回流",
    "no longer",
    "retired",
    "Retired",
    "已删除",
)
LEGACY_ROUTE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bHiST(?:-Beam)?\b",
        r"\bHist\b",
        r"Raymobtime s008",
        r"raymobtime_s008",
        r"standalone Top8",
        r"Top8 selector",
        r"GPS residual",
        r"camera residual",
        r"\bBGAM\b",
        r"viewer manifest",
        r"Gradio viewer",
        r"AMR-Net_gps_image",
        r"\bamr_net_gps_image\b",
        r"kd-sensing-run-amr-net-gps-image",
        r"JEPA-MSAC",
        r"\bjepa_msac\b",
        r"jepa_msac_pretraining",
        r"kd-sensing-run-jepa-msac",
        r"scripts/mmw/visualize_gps_",
        r"gps_circular_soft_label",
        r"run_mmw_sunny_modal15",
        r"visualize-modalities",
        r"\bCRAF\b",
        r"\bMARF\b",
        r"\bG2D\b",
        r"Multimodal-NF",
        r"legacy KD",
        r"旧 KD",
        r"logits_kd",
        r"\brkd\b",
    )
)
ACTIVE_ENTRY_WORDING_RE = re.compile(
    r"active mainline|当前主线|当前推荐|推荐入口|默认入口|默认 workflow|长期入口|"
    r"当前入口|quickstart|可运行训练|可运行 workflow"
)
LEGACY_ACTIVE_WORDING_RE = re.compile(
    ACTIVE_ENTRY_WORDING_RE.pattern
    + r"|MUST support|MUST provide|系统 MUST 支持|MUST 支持|项目 MUST 提供|"
    r"作为当前(?:入口|能力|workflow|支持能力)"
)
NON_CURRENT_CONTEXT_MARKERS = (
    "退役",
    "不再",
    "不得",
    "不属于",
    "已删除",
    "拒绝",
    "历史",
    "migration",
    "retired",
    "Retired",
    "removed",
    "supporting",
    "支撑",
    "防回流",
    "旧",
    "不可用",
    "不要求",
    "不能",
    "不作为",
    "不构建",
    "只作为",
    "仅作",
    "不会",
    "无关",
    "不把",
    "不声明",
    "不包含",
    "不新增",
    "禁止",
    "删除",
    "fail fast",
    "MUST NOT",
    "已从当前支持面",
)
CURRENT_WORKFLOW_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "agent_navigation.md",
    ROOT / "docs" / "project_surface_inventory.md",
    ROOT / "docs" / "experiment_matrix.md",
    ROOT / "docs" / "mainline_model_catalog.md",
    ROOT / "docs" / "experiment_protocols.md",
    ROOT / "docs" / "result_claims_registry.md",
    ROOT / "docs" / "extension_guide.md",
    ROOT / "docs" / "training_throughput.md",
)

JEPA_VISUAL_SWEEP_FILES = (
    ROOT / "configs" / "diagnostics" / "jepa_visual_architecture_sweep_manifest.yaml",
    ROOT / "configs" / "fusion" / "experiments" / "jepa_image_gps" / "architecture_sweep_smoke.yaml",
    ROOT / "configs" / "fusion" / "experiments" / "jepa_image_gps" / "architecture_sweep_lowmem.yaml",
    ROOT / "configs" / "fusion" / "experiments" / "jepa_image_gps" / "architecture_sweep_strict.yaml",
)
MAINLINE_EXPERIMENT_DOCS = (
    "docs/mainline_model_catalog.md",
    "docs/experiment_protocols.md",
    "docs/result_claims_registry.md",
)
MAINLINE_DOC_INDEX_PATHS = (
    ROOT / "README.md",
    ROOT / "docs" / "experiment_matrix.md",
    ROOT / "docs" / "project_surface_inventory.md",
)
HIGH_RISK_RESULT_WORDING_PATHS = (
    ROOT / "README_REPRODUCE.md",
    ROOT / "BASELINE_REPORT.md",
    ROOT / "results" / "reproduce_baseline.md",
    ROOT / "docs" / "experiment_matrix.md",
    ROOT / "docs" / "experiment_protocols.md",
    ROOT / "docs" / "result_claims_registry.md",
    ROOT / "configs" / "fusion" / "beambench_image_ae_gps_direct.yaml",
)
FUTURE_TARGET_MARKERS = (
    "target-beam-source future",
    "target_beam_source: future",
)
FUTURE_TARGET_CONTEXT_MARKERS = (
    "historical",
    "ablation",
    "sequence-prediction",
    "history",
    "历史",
    "不得",
    "not current",
)
TEST_AS_VALIDATION_CONTEXT_MARKERS = (
    "upper-bound",
    "上界",
    "test CSV",
    "非 official",
    "not official",
    "not strict",
    "不得",
)
MOCK_SMOKE_CONTEXT_MARKERS = (
    "mock",
    "MOCK",
    "smoke",
    "only",
    "只验证",
    "不得",
    "不能",
    "not a result",
    "Validates",
)



def _dotted(*parts: str) -> str:
    return ".".join(parts)


def _run_import_probe(statement: str) -> dict[str, bool]:
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
{statement}
modules = {{
    "deepsense6g": "kd_sensing.data.datasets.deepsense6g" in sys.modules,
    "synthetic": "kd_sensing.data.datasets.synthetic" in sys.modules,
    "models": any(name.startswith("kd_sensing.models.") for name in sys.modules),
    "diagnostics": any(name.startswith("kd_sensing.diagnostics.") for name in sys.modules),
    "artifact_registry": "kd_sensing.utils.artifact_registry" in sys.modules,
}}
print(json.dumps(modules, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _run_module_presence_probe(statement: str, modules: dict[str, str]) -> dict[str, bool]:
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
{statement}
modules = {modules!r}
print(json.dumps({{key: module in sys.modules for key, module in modules.items()}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _run_architecture_boundary_probe(statement: str) -> dict[str, bool]:
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
{statement}
modules = {{
    "torch": "torch" in sys.modules,
    "data_factory": "kd_sensing.engine.data_factory" in sys.modules,
    "deepsense6g": "kd_sensing.data.datasets.deepsense6g" in sys.modules,
    "models": any(name.startswith("kd_sensing.models") for name in sys.modules),
    "visualization_core": "kd_sensing.diagnostics.visualization.core" in sys.modules,
    "trainer": "kd_sensing.engine.trainer" in sys.modules,
}}
print(json.dumps(modules, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        path for path in result.stdout.decode("utf-8").split("\0") if path and (ROOT / path).exists()
    )


def _active_change_artifact_paths() -> list[str]:
    changes_root = ROOT / "openspec" / "changes"
    if not changes_root.exists():
        return []
    paths: list[str] = []
    for path in sorted(changes_root.glob("*")):
        if not path.is_dir() or path.name == "archive":
            continue
        for artifact in sorted(path.rglob("*")):
            if artifact.is_file() and artifact.suffix in {".md", ".yaml", ".yml", ".json"}:
                paths.append(artifact.relative_to(ROOT).as_posix())
    return paths


def _governance_scan_paths() -> list[str]:
    tracked = {
        path
        for path in _tracked_paths()
        if path.endswith(tuple(GOVERNANCE_SCAN_SUFFIXES))
        and any(path == prefix or path.startswith(prefix) for prefix in GOVERNANCE_SCAN_PREFIXES)
    }
    active_artifacts = set(_active_change_artifact_paths())
    return sorted(tracked | active_artifacts)


def _model_registration_names(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    registrations: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register"
                and isinstance(func.value, ast.Name)
                and func.value.id == "MODELS"
            ):
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                registrations.append((str(decorator.args[0].value), decorator.lineno))
            else:
                registrations.append((node.name, decorator.lineno))
    return registrations


def _active_change_spec_capabilities() -> set[str]:
    capabilities: set[str] = set()
    changes_root = ROOT / "openspec" / "changes"
    if not changes_root.exists():
        return capabilities
    for path in sorted(changes_root.glob("*/specs/*/spec.md")):
        if "archive" in path.parts:
            continue
        capabilities.add(path.parent.name)
    return capabilities


def _current_or_active_spec_paths(capability: str) -> list[Path]:
    current = ROOT / "openspec" / "specs" / capability / "spec.md"
    if current.exists():
        return [current]
    changes_root = ROOT / "openspec" / "changes"
    if not changes_root.exists():
        return []
    return [
        path
        for path in sorted(changes_root.glob(f"*/specs/{capability}/spec.md"))
        if "archive" not in path.parts
    ]


def _iter_scan_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in {".md", ".py", ".sh"})


def _symbol_lengths(path: Path) -> dict[tuple[str, str], int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    symbols: dict[tuple[str, str], int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, "end_lineno"):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        symbols[(node.name, kind)] = node.end_lineno - node.lineno + 1
    return symbols


def _is_sys_path_insert_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "insert"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def _is_supported_virtual_config_reference(rel_path: str) -> bool:
    path = Path(rel_path)
    stem = path.stem
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "configs" and parts[1] == "fusion":
        if stem in {"camera_ae_gps", "resnet_gps", "transformer_image_gps", "gps_only_neural"}:
            return True
        if any(token in stem for token in RETIRED_CONFIG_TOKENS):
            return False
        if stem.endswith(("_strong", "_lightweight", "_supervised")):
            return True
        if stem.endswith("_snapshot_next_frame_supervised"):
            return True
        objective_suffixes = (
            "_beam_supervised",
            "_occlusion_supervised",
            "_position_supervised",
            "_multitask_supervised",
        )
        return stem.endswith(objective_suffixes)
    if len(parts) == 3 and parts[0] == "configs" and parts[1] in {"image", "radar", "gps", "lidar", "mmwave"}:
        return stem == "snapshot_next_frame_supervised"
    return False


def _is_classified_non_current_config_reference(rel_path: str, rel_source: str, line: str) -> bool:
    if rel_source in HISTORICAL_DOCS_WITH_ARCHIVED_CONFIGS:
        return True
    if any(marker in line for marker in NON_CURRENT_CONFIG_CONTEXT_MARKERS):
        return True
    if rel_source.startswith("openspec/specs/") and any(token in rel_path for token in RETIRED_CONFIG_TOKENS):
        return True
    return False


def _openspec_lifecycle_inventory() -> tuple[dict[str, str], list[str], list[str]]:
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    assert OPENSPEC_LIFECYCLE_HEADING in inventory
    section = inventory.split(OPENSPEC_LIFECYCLE_HEADING, 1)[1].split("\n## ", 1)[0]
    lifecycles: dict[str, str] = {}
    duplicates: list[str] = []
    invalid: list[str] = []

    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        capability = parts[0].strip("`")
        lifecycle = parts[1].strip("`")
        if lifecycle not in OPENSPEC_LIFECYCLE_ALLOWED:
            invalid.append(f"{capability}: {lifecycle}")
            continue
        if capability in lifecycles:
            duplicates.append(capability)
        lifecycles[capability] = lifecycle

    return lifecycles, duplicates, invalid


def _openspec_first_requirement(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### Requirement:"):
            return line
    return ""


def _has_legacy_route_reference(line: str) -> bool:
    return any(pattern.search(line) for pattern in LEGACY_ROUTE_PATTERNS)


def _has_non_current_context(line: str) -> bool:
    return any(marker in line for marker in NON_CURRENT_CONTEXT_MARKERS)


def _has_allowed_current_jepa_context(line: str, nearby: str = "") -> bool:
    text = f"{line}\n{nearby}"
    if ("GPS-query" in text or "gps_query_pool" in text) and "JEPA" in text:
        return any(marker in text for marker in ("baseline", "compatibility", "兼容", "对照", "pooler"))
    condition_markers = (
        "condition_id_consumed",
        "blocked_condition_fields",
        "forbidden_condition_fields",
        "gps_condition",
        "image_condition",
    )
    if any(marker in text for marker in condition_markers):
        return any(
            marker in text
            for marker in (
                "diagnostic",
                "diagnostics",
                "诊断",
                "forbidden",
                "blocked",
                "禁用",
                "安全边界",
                "MUST NOT",
                "不读取",
                "不消费",
                "condition id",
            )
        )
    return False


def _nearby_text(lines: list[str], line_index: int, radius: int = 12) -> str:
    start = max(0, line_index - radius)
    end = min(len(lines), line_index + radius + 1)
    return "\n".join(lines[start:end])


def test_maintainer_context_index_schema_is_valid():
    assert MAINTAINER_CONTEXT_INDEX_PATH.exists()
    assert MAINTAINER_CONTEXT_INDEX["kind"] == "maintainer_context_index"
    assert len(MAINTAINER_CONTEXT_INDEX["routing"]["task_types"]) >= 8
    assert PYTHON_ENTRYPOINT_ALLOWLIST
    assert SHELL_ORCHESTRATION_ALLOWLIST
    assert FUSION_ROOT_YAML_ALLOWLIST
    assert EXISTING_MODEL_REGISTRATION_ALLOWLIST
    assert BATCH_RUNTIME_FUNCTION_ALLOWLIST
    assert MAINTAINER_CONTEXT_INDEX["governance"]["architecture_sizing_baseline"]
    assert HOTSPOT_SYMBOL_BUDGETS
    assert HEALTH_CHECK_COMMANDS
    assert RETIRED_CONFIG_TOKENS


def test_pyproject_scripts_match_maintainer_context_index():
    assert_pyproject_scripts_match_index(ROOT, MAINTAINER_CONTEXT_INDEX)


def test_thin_cli_aliases_delegate_without_workflow_logic():
    forbidden_markers = (
        "for epoch in",
        "for batch in",
        ".backward(",
        "optimizer.step(",
        "model.train(",
        "model.eval(",
        ".forward(",
        "forward_model(",
        "run_model_step(",
        "build_dataloaders(",
        "DataLoader(",
        "Dataset(",
        "pd.read_csv(",
        "csv.DictReader(",
        "torch.optim",
    )
    violations: list[str] = []

    for rel_path, metadata in sorted(PYTHON_ENTRYPOINT_METADATA.items()):
        if metadata["lifecycle"] != "thin_cli_alias":
            continue
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        owner_module = metadata.get("owner_module")
        if not isinstance(owner_module, str) or not owner_module:
            violations.append(f"{rel_path} is thin_cli_alias but has no owner_module")
            continue
        owner_imports = (
            f"from {owner_module} import",
            f"import {owner_module}",
        )
        if not any(snippet in text for snippet in owner_imports):
            violations.append(f"{rel_path} does not directly delegate owner module {owner_module}")
        if len(text.splitlines()) > 80:
            violations.append(f"{rel_path} is too long for a thin CLI alias")
        for marker in forbidden_markers:
            if marker in text:
                violations.append(f"{rel_path} contains workflow marker {marker!r}")

    assert violations == []


def test_beambench_check_dataset_aliases_delegate_owner_module():
    script_text = (ROOT / "scripts" / "check_dataset.py").read_text(encoding="utf-8")
    cli_alias_path = SRC / "kd_sensing" / "cli" / "beambench_check_dataset.py"
    cli_alias_text = cli_alias_path.read_text(encoding="utf-8")

    assert PYTHON_ENTRYPOINT_METADATA["scripts/check_dataset.py"]["owner_module"] == "kd_sensing.cli.beambench_check_dataset"
    assert "from kd_sensing.cli.beambench_check_dataset import main" in script_text
    assert "from kd_sensing.baselines.beambench.dataset_check import main" in cli_alias_text
    assert "argparse.ArgumentParser" not in cli_alias_text
    assert "Dataset(" not in cli_alias_text


def test_runtime_source_does_not_import_maintainer_context_helper():
    helper_markers = (
        "tests.helpers.maintainer_context",
        "helpers.maintainer_context",
    )
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((SRC / "kd_sensing").rglob("*.py"))
        if any(marker in path.read_text(encoding="utf-8") for marker in helper_markers)
    ]

    assert violations == []


def test_source_surface_does_not_track_local_artifacts():
    violations: list[str] = []
    for raw_path in _tracked_paths():
        path = Path(raw_path)
        parts = set(path.parts)
        suffix = path.suffix.lower()
        if "__pycache__" in parts or ".pytest_cache" in parts:
            violations.append(raw_path)
        elif suffix in {".pyc", ".pyo"}:
            violations.append(raw_path)
        elif raw_path.startswith(("outputs/", "logs/")):
            violations.append(raw_path)
        elif raw_path.startswith("dataset/") and raw_path != "dataset/.gitkeep":
            violations.append(raw_path)
        elif suffix in {".pth", ".pt", ".ckpt"} and not raw_path.startswith("All_models/"):
            violations.append(raw_path)

    assert violations == []


def test_governance_scans_stay_on_source_docs_specs_and_tests():
    forbidden_prefixes = ("dataset/", "outputs/", "logs/", "cache/", ".pytest_cache/")
    forbidden_suffixes = (".pth", ".pt", ".ckpt")
    violations = [
        path
        for path in _governance_scan_paths()
        if path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)
    ]

    assert violations == []


def test_model_registrations_are_documented_or_allowlisted():
    documented_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in _governance_scan_paths()
        if path.startswith(("docs/", "openspec/specs/", "openspec/changes/"))
    )
    violations: list[str] = []
    for rel_path in _tracked_paths():
        if not rel_path.startswith("src/kd_sensing/models/") or not rel_path.endswith(".py"):
            continue
        for name, line_number in _model_registration_names(ROOT / rel_path):
            if name in EXISTING_MODEL_REGISTRATION_ALLOWLIST:
                continue
            if name in documented_text:
                continue
            violations.append(
                f"{rel_path}:{line_number} registers MODELS name '{name}' without current spec, "
                "active change artifact, inventory entry, or explicit allowlist."
            )

    assert violations == []


def test_retired_legacy_model_registry_names_are_not_model_allowlist_entries():
    payload = yaml.safe_load(LEGACY_MODEL_RETIREMENT_FIXTURE.read_text(encoding="utf-8"))
    retired_model_names = {
        str(entry["name"])
        for entry in payload["retired"]
        if entry.get("registry") == "models"
    }

    assert retired_model_names.isdisjoint(EXISTING_MODEL_REGISTRATION_ALLOWLIST)


def test_extension_guide_defaults_to_modular_model_extension():
    guide = (ROOT / "docs" / "extension_guide.md").read_text(encoding="utf-8")
    add_model = guide.split("## Add a Model", 1)[1].split("## Add a Dataset", 1)[0]
    default_section = add_model.split("### Whole-model Exceptions", 1)[0]

    assert "modular_sequence" in default_section
    assert "@ENCODERS.register" in default_section or "@REPRESENTATION_CORES.register" in default_section
    assert "@MODELS.register" not in default_section, (
        "Add a Model must not present direct whole-model registration as the default baseline path; "
        "use modular_sequence config or a subcomponent registry example first."
    )
    assert "### Whole-model Exceptions" in add_model
    assert "@MODELS.register" in add_model


def test_predictive_jepa_gate_stays_modular_representation_core():
    modular_source = (ROOT / "src" / "kd_sensing" / "models" / "modular.py").read_text(encoding="utf-8")

    assert '@REPRESENTATION_CORES.register("feature_consistency_gate")' in modular_source
    assert '@REPRESENTATION_CORES.register("jepa_feature_consistency_gate")' in modular_source
    assert '@MODELS.register("feature_consistency_gate")' not in modular_source
    assert '"predictive_condition_id"' in modular_source
    assert '"condition_id_consumed": False' in modular_source


def test_batch_runtime_extension_surface_is_allowlisted():
    violations: list[str] = []
    for rel_path, allowed in BATCH_RUNTIME_FUNCTION_ALLOWLIST.items():
        tree = ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))
        discovered = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (
                node.name.startswith("prepare_")
                or node.name.startswith("forward_")
                or node.name == "validate"
                or node.name.startswith("_validate_")
                or node.name.startswith("_resolve_validation_")
            )
        }
        extra = sorted(discovered - allowed)
        missing = sorted(allowed - discovered)
        for name in extra:
            violations.append(f"{rel_path}:{name} is a new batch/runtime/validation branch; update contract tests.")
        for name in missing:
            violations.append(f"{rel_path}:{name} is allowlisted but no longer exists; update the allowlist.")

    runtime_definitions: list[str] = []
    for rel_path in _tracked_paths():
        if not rel_path.endswith(".py") or not rel_path.startswith("src/kd_sensing/"):
            continue
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        if "def forward_task_model" in text and rel_path != "src/kd_sensing/engine/runtime.py":
            runtime_definitions.append(rel_path)
    assert violations == []
    assert runtime_definitions == []


def test_mainline_experiment_docs_are_indexed_and_current():
    lifecycles, _, _ = _openspec_lifecycle_inventory()
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")

    assert lifecycles.get("mainline-experiment-documentation") == "current"
    for rel_path in MAINLINE_EXPERIMENT_DOCS:
        path = ROOT / rel_path
        assert path.exists(), f"{rel_path} must exist"
        assert rel_path in inventory, f"{rel_path} must be classified in docs/project_surface_inventory.md"

    for index_path in MAINLINE_DOC_INDEX_PATHS:
        text = index_path.read_text(encoding="utf-8")
        for rel_path in MAINLINE_EXPERIMENT_DOCS:
            assert rel_path in text, f"{index_path.relative_to(ROOT)} must link to {rel_path}"


def test_priority_legacy_workflows_are_retired_from_current_surfaces():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    retired_commands = {
        "kd-sensing-run-amr-net-gps-image",
        "kd-sensing-run-jepa-msac",
    }
    retired_configs = {
        "configs/baselines/amr_net_gps_image.yaml",
        "configs/pretraining/jepa_msac_s32_smoke.yaml",
        "configs/pretraining/jepa_msac_s32_paper.yaml",
    }
    retired_modules = {
        "kd_sensing.cli.run_amr_net_gps_image",
        "kd_sensing.cli.run_jepa_msac",
        "kd_sensing.baselines.amr_net_gps_image",
        "kd_sensing.baselines.jepa_msac",
        "kd_sensing.models.jepa_msac",
        "kd_sensing.losses.jepa_msac",
    }

    assert retired_commands.isdisjoint(PACKAGE_CLI_METADATA)
    assert all(command not in pyproject for command in retired_commands)
    assert all(not (ROOT / rel_path).exists() for rel_path in retired_configs)
    for module_name in sorted(retired_modules):
        assert importlib.util.find_spec(module_name) is None


def test_priority_legacy_scripts_are_not_current_allowlist_entries():
    retired_scripts = {
        "scripts/mmw/visualize_gps_angle_beam_correspondence.py",
        "scripts/mmw/visualize_gps_prediction_trajectory.py",
        "scripts/mmw/visualize_prediction_error_label_distribution.py",
        "scripts/run_deepsense_gps_circular_soft_label.sh",
        "scripts/run_mmw_gps_circular_soft_label_ablation.sh",
        "scripts/run_mmw_sunny_modal15_l5p3_h123.sh",
        "scripts/run_mmw_sunny_modal15_l5p6_h246.sh",
    }

    assert retired_scripts.isdisjoint(PYTHON_ENTRYPOINT_ALLOWLIST)
    assert retired_scripts.isdisjoint(SHELL_ORCHESTRATION_ALLOWLIST)
    assert all(not (ROOT / rel_path).exists() for rel_path in retired_scripts)


def test_predictive_jepa_robustness_governance_boundaries_are_synchronized():
    lifecycles, _, _ = _openspec_lifecycle_inventory()
    assert lifecycles.get("predictive-jepa-robustness") == "current"

    spec_text = (ROOT / "openspec" / "specs" / "predictive-jepa-robustness" / "spec.md").read_text(
        encoding="utf-8"
    )
    assert "TBD" not in spec_text
    assert "P0-P5" in spec_text
    assert "strict comparable train-then-evaluate" in spec_text

    for rel_path in (
        "docs/experiment_matrix.md",
        "docs/experiment_protocols.md",
        "docs/mainline_model_catalog.md",
        "docs/result_claims_registry.md",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "P4_joint_predictive_recovery" in text, f"{rel_path} must identify the single train profile"
        assert "P0-P5" in text, f"{rel_path} must identify the full benchmark suite"
        assert "不等价于完整 P0-P5" in text or "does not equal a full P0-P5" in text

    smoke_manifest = (
        ROOT / "configs" / "diagnostics" / "jepa_gps_shortcut_benchmark_predictive_robustness_smoke.yaml"
    ).read_text(encoding="utf-8")
    for marker in (
        "synthetic_metrics",
        "mock_not_committed",
        "allow_missing_artifacts: true",
        "claim_status: mock/smoke",
        "provenance_status: unavailable",
    ):
        assert marker in smoke_manifest


def test_high_risk_result_wording_has_local_caveats():
    violations: list[str] = []

    for path in HIGH_RISK_RESULT_WORDING_PATHS:
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(ROOT).as_posix()
        for index, line in enumerate(lines):
            window = _nearby_text(lines, index)
            if any(marker in line for marker in FUTURE_TARGET_MARKERS):
                if not any(marker in window for marker in FUTURE_TARGET_CONTEXT_MARKERS):
                    violations.append(
                        f"{rel}:{index + 1} mentions future target without historical/ablation caveat"
                    )
            if "test_as_validation" in line:
                if not any(marker in window for marker in TEST_AS_VALIDATION_CONTEXT_MARKERS):
                    violations.append(
                        f"{rel}:{index + 1} mentions test_as_validation without upper-bound caveat"
                    )
            if "mock_data: true" in line:
                if not any(marker in window for marker in MOCK_SMOKE_CONTEXT_MARKERS):
                    violations.append(
                        f"{rel}:{index + 1} mentions mock_data without mock/smoke caveat"
                    )

    assert violations == []


def test_experiment_workflow_does_not_restore_legacy_kd_active_wording():
    spec = (ROOT / "openspec" / "specs" / "experiment-workflow" / "spec.md").read_text(
        encoding="utf-8"
    )
    forbidden_snippets = (
        "构建 image-only dataset、teacher/student 模型",
        "fusion teacher/student 模型、KD/loss",
        "构建对应蒸馏逻辑",
        "项目 MUST 提供 radar-only lightweight student no-KD 配置",
        "Fusion KD 配置 MUST 要求 teacher 和 student",
        "系统 MUST 构建只包含 image 和 gps 分支的 fusion teacher/student",
        "系统 MUST 构建五个模态输入所需的 dataset 字段和 fusion teacher/student 模型",
        "Raymobtime s008 selection",
    )

    assert [snippet for snippet in forbidden_snippets if snippet in spec] == []


def test_jepa_visual_architecture_sweep_uses_current_entrypoints_and_no_retired_routes():
    for path in JEPA_VISUAL_SWEEP_FILES:
        text = path.read_text(encoding="utf-8")
        if path.name == "jepa_visual_architecture_sweep_manifest.yaml":
            assert "conda run -n kd_mm_beam" in text
            assert "scripts/train.py" in text
            assert "scripts/evaluate.py" in text
        assert [pattern.pattern for pattern in LEGACY_ROUTE_PATTERNS if pattern.search(text)] == []
    assert not (ROOT / "train_jepa_visual_architecture_sweep.py").exists()
    assert not (ROOT / "run_jepa_visual_architecture_sweep.py").exists()


def test_cnn_hybrid_jepa_visual_prior_sweep_is_package_scoped_and_output_scoped():
    manifest = ROOT / "configs" / "diagnostics" / "cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml"
    module = SRC / "kd_sensing" / "diagnostics" / "cnn_hybrid_jepa_visual_prior_sweep.py"
    manifest_text = manifest.read_text(encoding="utf-8")
    module_text = module.read_text(encoding="utf-8")

    assert "outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep" in manifest_text
    assert "outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep" in module_text
    assert "conda run -n kd_mm_beam" in module_text
    assert "scripts/train.py" in module_text
    assert "scripts/evaluate.py" in module_text
    assert "/root/.container_env" in module_text
    assert "dataset" in module_text
    assert "All_models" in module_text
    assert not (ROOT / "train_cnn_hybrid_jepa_visual_prior_sweep.py").exists()
    assert not (ROOT / "run_cnn_hybrid_jepa_visual_prior_sweep.py").exists()


def test_project_surface_inventory_guardrails_are_current():
    fusion_yaml = sorted((ROOT / "configs" / "fusion").glob("*.yaml"))
    fusion_root_entries = {path.relative_to(ROOT).as_posix() for path in fusion_yaml}
    tracked = set(_tracked_paths())
    script_entries = {
        path
        for path in tracked
        if path.endswith(".py")
        and path.startswith(("scripts/", "tools/analysis/"))
    }
    shell_entries = {
        path
        for path in tracked
        if path.endswith(".sh") and path.startswith("scripts/")
    }

    assert fusion_root_entries == FUSION_ROOT_YAML_ALLOWLIST
    assert RETIRED_GENERATED_FUSION_CONFIGS.isdisjoint(
        fusion_root_entries
    )
    assert script_entries == set(PYTHON_ENTRYPOINT_ALLOWLIST)
    assert "scripts/eval_modality_subsets.py" not in script_entries
    assert "scripts/eval_modality_perturbation.py" not in script_entries
    assert shell_entries == set(SHELL_ORCHESTRATION_ALLOWLIST)



def test_shared_pytest_bootstrap_and_pytest_config_are_declared():
    conftest = ROOT / "tests" / "conftest.py"
    conftest_text = conftest.read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "SRC = ROOT / \"src\"" in conftest_text
    assert "sys.path.insert(0, path_text)" in conftest_text
    assert "kd_sensing" not in conftest_text
    assert "[tool.pytest.ini_options]" in pyproject
    assert "testpaths = [\"tests\"]" in pyproject
    assert "pytest.PytestUnknownMarkWarning" in pyproject
    assert "local_data:" in pyproject


def test_regular_tests_do_not_duplicate_src_path_bootstrap():
    violations: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _is_sys_path_insert_call(node):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(
                    f"{rel}:{node.lineno} duplicates sys.path bootstrap; use tests/conftest.py "
                    "or keep path control inside a subprocess code string for import-boundary probes."
                )

    assert violations == []


def test_health_inventory_documents_hotspots_and_commands():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    health_spec_path = (
        ROOT
        / "openspec"
        / "changes"
        / "strengthen-project-health-guardrails"
        / "specs"
        / "project-health-guardrails"
        / "spec.md"
    )
    if not health_spec_path.exists():
        health_spec_path = ROOT / "openspec" / "specs" / "project-health-guardrails" / "spec.md"
    health_spec = health_spec_path.read_text(encoding="utf-8")

    for command in HEALTH_CHECK_COMMANDS:
        assert command in inventory
        assert command in readme or command in health_spec
    for marker in REQUIRED_HOTSPOT_INVENTORY_MARKERS:
        assert marker in inventory
    for marker in CONFIG_LIFECYCLE_MARKERS:
        assert marker in inventory
    assert "新增热点维护规则" in inventory
    assert "tests/conftest.py" in inventory


def test_agent_navigation_document_covers_maintainer_boundaries():
    navigation_path = ROOT / "docs" / "agent_navigation.md"
    assert navigation_path.exists()

    navigation = navigation_path.read_text(encoding="utf-8")
    for marker in AGENT_NAVIGATION_MARKERS:
        assert marker in navigation
    assert "当前状态检查顺序" in navigation
    assert "权威来源优先级" in navigation
    assert "验证命令选择表" in navigation
    assert "不维护完整源码目录清单" in navigation


def test_agent_navigation_is_referenced_from_rules_and_inventory():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    navigation = (ROOT / "docs" / "agent_navigation.md").read_text(encoding="utf-8")
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/agent_navigation.md" in agents
    assert "docs/maintainer_context_index.yaml" in agents
    assert "docs/maintainer_context_index.yaml" in navigation
    assert "docs/maintainer_context_index.yaml" in inventory
    assert "机器可读治理" in inventory
    assert "非平凡改动前" in agents
    assert "docs/agent_navigation.md" in inventory
    assert "current agent/maintainer navigation" in inventory
    assert "不替代 README、AGENTS 或 OpenSpec specs" in inventory
    assert "docs/agent_navigation.md" in readme


def test_hotspot_action_metadata_documents_right_sizing_waves():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    navigation = (ROOT / "docs" / "agent_navigation.md").read_text(encoding="utf-8")

    assert {"right-size-accepted", "merge-candidate", "keep-and-test"} <= set(
        HOTSPOT_ACTION_METADATA["status_values"]
    )
    assert {"hard-fail", "headroom", "monitor", "accepted", "merge-required"} <= set(
        HOTSPOT_ACTION_METADATA["enforcement_values"]
    )
    assert {"split", "consolidate", "keep-and-test", "owner-facade", "hard-budget", "accepted-size"} <= set(
        HOTSPOT_ACTION_METADATA["planned_action_values"]
    )
    assert HOTSPOT_REMEDIATION_WAVES
    assert {wave["id"] for wave in HOTSPOT_REMEDIATION_WAVES} >= {
        "wave-0",
        "wave-1-beambench-image-ae-gps",
        "wave-2-datasets-trainer",
        "wave-3-evaluation-diagnostics",
        "wave-4-jepa-accepted-owners",
        "wave-5-consolidation-imports",
    }
    baseline = MAINTAINER_CONTEXT_INDEX["governance"]["architecture_sizing_baseline"]
    assert baseline["codegraph"]["python_files"] == baseline["ast"]["python_files"]
    assert "outputs/" in baseline["measurement_scope"]["excludes"]
    assert "dataset/" in baseline["measurement_scope"]["excludes"]
    assert "trend signals only" in "\n".join(baseline["interpretation"])
    for wave in HOTSPOT_REMEDIATION_WAVES:
        assert wave["planned_action"] in HOTSPOT_ACTION_METADATA["planned_action_values"]
        assert wave["public_surface_policy"] in HOTSPOT_ACTION_METADATA["public_surface_policy_values"]
        for command in wave["validation_commands"]:
            if "pytest" in command or "python " in command or "kd-sensing-" in command:
                assert command.startswith("conda run -n kd_mm_beam ")
        for rel_path in wave["target_paths"]:
            assert (ROOT / rel_path).exists()
        assert wave["rollback_note"].strip()

    for marker in (
        "architecture sizing baseline",
        "right-size-project-architecture",
        "right-size-accepted",
        "merge-candidate",
        "keep-and-test",
        "remediation wave",
    ):
        assert marker in inventory
        assert marker in navigation


def _hotspot_over_budget_allowed(actual: int, metadata: dict[str, object]) -> bool:
    max_lines = int(metadata["max_lines"])
    headroom = int(metadata.get("headroom_lines", 0))
    enforcement = str(metadata.get("enforcement", "hard-fail"))
    if actual <= max_lines:
        return True
    if enforcement == "hard-fail":
        return False
    return actual <= max_lines + headroom


def test_hotspot_static_budget_matches_inventory():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    budget_keys = set(HOTSPOT_SYMBOL_BUDGETS)
    violations: list[str] = []

    for rel_path, max_lines in HOTSPOT_FILE_BUDGETS.items():
        actual = len((ROOT / rel_path).read_text(encoding="utf-8").splitlines())
        metadata = HOTSPOT_FILE_METADATA[rel_path]
        if not _hotspot_over_budget_allowed(actual, metadata):
            headroom = int(metadata.get("headroom_lines", 0))
            violations.append(
                f"{rel_path} is {actual} lines, budget {max_lines} + headroom {headroom}; "
                "split suite-specific benchmark helpers, consolidate low-value boundaries, "
                "or update docs/project_surface_inventory.md with a reasoned right-sizing action."
            )
        if rel_path not in inventory:
            violations.append(f"{rel_path} is file-budgeted but missing from hotspot inventory")

    for (rel_path, symbol, kind), max_lines in HOTSPOT_SYMBOL_BUDGETS.items():
        lengths = _symbol_lengths(ROOT / rel_path)
        actual = lengths.get((symbol, kind))
        if actual is None:
            violations.append(f"{rel_path}:{symbol} missing from AST scan")
            continue
        metadata = HOTSPOT_SYMBOL_METADATA[(rel_path, symbol, kind)]
        if not _hotspot_over_budget_allowed(actual, metadata):
            headroom = int(metadata.get("headroom_lines", 0))
            violations.append(
                f"{rel_path}:{symbol} is {actual} lines, budget {max_lines} + headroom {headroom}; "
                "split to a narrow module, consolidate low-value boundaries, "
                "or update docs/project_surface_inventory.md with a reasoned right-sizing action."
            )
        if rel_path not in inventory or symbol not in inventory:
            violations.append(f"{rel_path}:{symbol} is budgeted but missing from hotspot inventory")

    for path in sorted((SRC / "kd_sensing").rglob("*.py")):
        rel_path = path.relative_to(ROOT).as_posix()
        for (symbol, kind), lines in _symbol_lengths(path).items():
            threshold = LONG_CLASS_LIMIT if kind == "class" else LONG_FUNCTION_LIMIT
            if lines <= threshold:
                continue
            if (rel_path, symbol, kind) in budget_keys:
                continue
            violations.append(
                f"{rel_path}:{symbol} {kind} is {lines} lines and not registered; "
                "update docs/project_surface_inventory.md, add a budget, or split the symbol."
            )

    assert violations == []


def test_config_lifecycle_inventory_and_references_are_current():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    for marker in CONFIG_LIFECYCLE_MARKERS:
        assert marker in inventory

    violations: list[str] = []
    for root in CONFIG_REFERENCE_SCAN_ROOTS:
        for path in _iter_scan_files(root):
            rel_source = path.relative_to(ROOT).as_posix()
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                for match in CONFIG_REFERENCE_RE.finditer(line):
                    rel_path = match.group(0).rstrip(".,;:")
                    if (ROOT / rel_path).exists():
                        continue
                    if _is_supported_virtual_config_reference(rel_path):
                        continue
                    if _is_classified_non_current_config_reference(rel_path, rel_source, line):
                        continue
                    violations.append(
                        f"{rel_source}:{line_number} references missing config {rel_path}; "
                        "create it, document it as a virtual/current config, or mark the reference retired/historical."
                    )

    assert violations == []


def test_root_and_docs_markdown_lifecycle_inventory_is_complete():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    root_docs = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("*.md")
        if path.name not in ROOT_DOCUMENT_LIFECYCLE_EXCLUSIONS
    }
    docs_docs = {path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*.md")}
    missing = sorted(rel_path for rel_path in root_docs | docs_docs if rel_path not in inventory)

    assert missing == []

    recommendation_violations: list[str] = []
    for path in CURRENT_DOCS_TO_CHECK_FOR_RETIRED_RECOMMENDATIONS:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not any(marker in line for marker in RETIRED_ROUTE_TEXT_MARKERS):
                continue
            if any(marker in line for marker in RETIRED_ROUTE_CLASSIFICATION_MARKERS):
                continue
            recommendation_violations.append(
                f"{rel}:{line_number} mentions a retired route without historical/retired wording: {line.strip()}"
            )

    assert recommendation_violations == []


def test_openspec_lifecycle_inventory_covers_current_specs():
    lifecycles, duplicates, invalid = _openspec_lifecycle_inventory()
    spec_capabilities = {
        path.parent.name
        for path in sorted((ROOT / "openspec" / "specs").glob("*/spec.md"))
    }
    active_change_capabilities = _active_change_spec_capabilities()

    missing = sorted(spec_capabilities - set(lifecycles))
    extra = sorted(set(lifecycles) - spec_capabilities - active_change_capabilities)

    assert duplicates == []
    assert invalid == []
    assert missing == [], (
        "Every openspec/specs/<capability>/spec.md must be classified in "
        "docs/project_surface_inventory.md as current, supporting, or retired-tombstone. "
        f"Missing: {missing}"
    )
    assert extra == [], f"Lifecycle inventory references non-current specs or active change specs: {extra}"
    assert set(lifecycles.values()) <= OPENSPEC_LIFECYCLE_ALLOWED


def test_retired_tombstone_specs_are_visibly_retired():
    lifecycles, _, _ = _openspec_lifecycle_inventory()
    wording_violations: list[str] = []
    active_violations: list[str] = []

    for capability, lifecycle in sorted(lifecycles.items()):
        if lifecycle != "retired-tombstone":
            continue
        path = ROOT / "openspec" / "specs" / capability / "spec.md"
        opening = f"{_openspec_purpose_text(path)} {_openspec_first_requirement(path)}"
        if not any(marker in opening for marker in RETIREMENT_WORDING_MARKERS):
            wording_violations.append(
                f"{path.relative_to(ROOT)} must state retirement, rejection, history, or migration-guard "
                "semantics in its Purpose or first requirement."
            )
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if ACTIVE_ENTRY_WORDING_RE.search(line) and not _has_non_current_context(line):
                active_violations.append(
                    f"{path.relative_to(ROOT)}:{line_number} contains active-entry wording without "
                    f"retired/historical context: {line.strip()}"
                )

    assert wording_violations == []
    assert active_violations == []


def test_current_specs_and_docs_do_not_recommend_retired_routes():
    lifecycles, _, _ = _openspec_lifecycle_inventory()
    scan_paths: list[Path] = []
    for capability, lifecycle in sorted(lifecycles.items()):
        if lifecycle != "current":
            continue
        scan_paths.extend(_current_or_active_spec_paths(capability))
    scan_paths.extend(path for path in CURRENT_WORKFLOW_DOCS if path.exists())
    violations: list[str] = []

    for path in scan_paths:
        rel = path.relative_to(ROOT).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not _has_legacy_route_reference(line):
                continue
            if not LEGACY_ACTIVE_WORDING_RE.search(line):
                continue
            if _has_non_current_context(line):
                continue
            if _has_allowed_current_jepa_context(line, _nearby_text(lines, line_number - 1, radius=4)):
                continue
            violations.append(
                f"{rel}:{line_number} describes a retired route with current/active wording: {line.strip()}"
            )

    assert violations == []


def test_current_jepa_compatibility_wording_is_not_retired_route_regression():
    allowed_lines = [
        "JEPA GPS-query baseline compatibility MUST preserve the existing gps_query_pool control behavior.",
        "condition_id_consumed=false documents a feature-consistency gate diagnostic safety boundary.",
        "blocked_condition_fields and forbidden_condition_fields list gps_condition and image_condition as forbidden diagnostics, not model inputs.",
    ]
    forbidden_lines = [
        "GPS residual is the active mainline workflow.",
        "Top8 selector standalone is the active mainline.",
        "HiST-Beam MUST support the default workflow.",
    ]

    for line in allowed_lines:
        assert _has_allowed_current_jepa_context(line, line)

    for line in forbidden_lines:
        assert _has_legacy_route_reference(line)
        assert LEGACY_ACTIVE_WORDING_RE.search(line)
        assert not _has_non_current_context(line)
        assert not _has_allowed_current_jepa_context(line, line)


def test_retired_top8_residual_bgam_viewer_routes_are_not_current_source_modules():
    retired_paths = [
        ROOT / "configs" / "deepsense6g_residual_fusion.yaml",
        ROOT / "configs" / "deepsense6g_camera_residual.yaml",
        ROOT / "configs" / "deepsense6g_top8_selector.yaml",
        ROOT / "configs" / "mmw_town_top8_selector.yaml",
        ROOT / "configs" / "deepsense6g_gps_lidar_bgam.yaml",
        ROOT / "configs" / "mmw_town_gps_lidar_bgam.yaml",
        ROOT / "configs" / "diagnostics" / "modality_visualization.yaml",
        ROOT / "configs" / "gps" / "gps_coarse_anchor_smoke.yaml",
        ROOT / "configs" / "gps" / "gps_coarse_anchor_target_adapt.yaml",
        ROOT / "configs" / "gps" / "gps_neural_coarse_smoke.yaml",
        ROOT / "src" / "kd_sensing" / "cli" / "gps_coarse_anchor.py",
        ROOT / "src" / "kd_sensing" / "cli" / "inspect_deepsense6g_residual_inputs.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_deepsense6g_residual_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_deepsense6g_residual_fusion.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_deepsense6g_top8_candidate_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_deepsense6g_top8_selector.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_mmw_town_top8_candidate_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_deepsense6g_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "evaluate_deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_mmw_town_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "evaluate_mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "export_viewer_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "visualize_modalities.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_camera_residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "geometry_residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_topk_candidate_manifest.py",
        ROOT / "src" / "kd_sensing" / "data" / "mmw_town_topk_candidate_manifest.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_dataset.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "data" / "mmw_town_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "engine" / "gps_coarse_anchor.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_residual_fusion.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_top8_selector.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "engine" / "mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "deepsense6g_residual_fusion.py",
        ROOT / "src" / "kd_sensing" / "models" / "topk_candidate_selector.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam_model.py",
        ROOT / "src" / "kd_sensing" / "losses" / "residual.py",
        ROOT / "src" / "kd_sensing" / "losses" / "gps_lidar_bgam_losses.py",
        ROOT / "src" / "kd_sensing" / "losses" / "topk_candidate_losses.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_cache.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_config.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_datasets.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_merge.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_paths.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_sampling.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_schema.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_stats.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_manifest_writer.py",
        ROOT / "src" / "kd_sensing" / "diagnostics" / "viewer_predictions.py",
        ROOT / "src" / "kd_sensing" / "data" / "datasets" / "raymobtime_s008.py",
        ROOT / "src" / "kd_sensing" / "preprocessing" / "raymobtime_s008.py",
        ROOT / "src" / "kd_sensing" / "models" / "raymobtime_s008.py",
        ROOT / "configs" / "raymobtime",
        ROOT / "docs" / "Raymobtime_s008_selection.md",
        ROOT / "tests" / "test_raymobtime_s008_selection.py",
        ROOT / "tests" / "test_gps_lidar_bgam_geometry.py",
        ROOT / "tests" / "test_gps_lidar_bgam_model.py",
        ROOT / "tests" / "test_gps_lidar_bgam_dataset.py",
        ROOT / "tests" / "test_gps_lidar_bgam_runner.py",
    ]
    violations = [path.relative_to(ROOT).as_posix() for path in retired_paths if path.exists()]

    assert violations == []


def test_retired_bgam_root_scripts_are_not_reintroduced():
    forbidden_root_entries = [
        ROOT / "train_gps_lidar_bgam.py",
        ROOT / "eval_gps_lidar_bgam.py",
        ROOT / "datasets" / "gps_lidar_dataset.py",
        ROOT / "models" / "gps_lidar_bgam.py",
    ]
    violations = [path.relative_to(ROOT).as_posix() for path in forbidden_root_entries if path.exists()]

    assert violations == []


def test_shell_orchestration_defaults_do_not_write_to_outputs_other():
    violations = []
    forbidden = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/other'
    for rel_path in sorted(SHELL_ORCHESTRATION_ALLOWLIST):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        if forbidden in text:
            violations.append(rel_path)

    assert violations == []


def test_runtime_output_defaults_use_canonical_partitions():
    partitions = canonical_runtime_partitions("outputs")

    assert DEFAULT_CONFIG["output"]["dir"] == "outputs"
    assert DEFAULT_CONFIG["checkpoint"]["registry"]["dir"] == "outputs"
    assert partitions["cache"] == "outputs/cache"
    assert partitions["cleanup_manifests"] == "outputs/cleanup_manifests"
    assert partitions["analysis"] == "outputs/analysis"
    assert partitions["visual_analysis"] == "outputs/visual_analysis"
    assert partitions["evaluations"] == "outputs/evaluations"
    assert partitions["scene"] == "outputs/scene<id>"
    assert partitions["scenegroup"] == "outputs/scenegroup_<range-or-list>"


def test_entrypoint_lifecycle_categories_are_explicit_and_documented():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    classifications = {
        **PYTHON_ENTRYPOINT_ALLOWLIST,
        **SHELL_ORCHESTRATION_ALLOWLIST,
    }

    assert set(classifications.values()) <= ENTRYPOINT_LIFECYCLES
    for rel_path, lifecycle in classifications.items():
        assert rel_path in inventory
        assert lifecycle in inventory


def test_hotspot_inventory_documents_facades_and_narrow_modules():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    required_paths = [
        "src/kd_sensing/engine/objective_metadata.py",
        "src/kd_sensing/engine/objectives/registry.py",
        "src/kd_sensing/engine/objectives/history.py",
        "src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_common.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_manifest.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_scenario_c.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_scenario_d.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_predictive.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_perturbations.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_artifacts.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_plots.py",
        "src/kd_sensing/diagnostics/jepa_benchmark_runner.py",
    ]

    for rel_path in required_paths:
        assert rel_path in inventory
    assert "不得从 `kd_sensing.engine.objective_metadata`" in inventory
    assert "kd_sensing.diagnostics.jepa_gps_shortcut_benchmark" in inventory
    assert "JEPA benchmark owner 模块" in inventory


def test_recipe_generated_advanced_yaml_paths_do_not_reenter_source_surface():
    recipe_file = SRC / "kd_sensing" / "config" / "canonical_recipes" / "advanced.py"
    recipe_text = recipe_file.read_text(encoding="utf-8")

    for retired_path in sorted(RETIRED_GENERATED_FUSION_CONFIGS):
        stem = Path(retired_path).stem
        assert not (ROOT / retired_path).exists()
        assert f'"{stem}"' not in recipe_text


def test_hotspot_facades_delegate_to_narrow_responsibility_modules():
    expectations = {
        "src/kd_sensing/models/csi.py": {
            "max_lines": 40,
            "forbidden": [
                "class PilotCSIChannelEstimator",
                "class CSIHardening",
                "class CSIViewTokenizer",
                "class PilotDualViewCSIEncoder",
            ],
            "helpers": {
                "src/kd_sensing/models/csi_estimation.py": "class PilotCSIChannelEstimator",
                "src/kd_sensing/models/csi_hardening.py": "class CSIHardening",
                "src/kd_sensing/models/csi_views.py": "class CSIViewTokenizer",
                "src/kd_sensing/models/csi_encoder.py": "class PilotDualViewCSIEncoder",
            },
        },
        "src/kd_sensing/engine/objective_metadata.py": {
            "max_lines": 20,
            "forbidden": [
                "PREDICTION_OBJECTIVES =",
                "def objective_spec",
                "_METRIC_ALIASES",
                "_HISTORY_FIELDS",
            ],
            "helpers": {
                "src/kd_sensing/engine/objectives/registry.py": "_METRIC_ALIASES",
                "src/kd_sensing/engine/objectives/history.py": "_HISTORY_FIELDS",
                "src/kd_sensing/engine/objectives/metadata.py": "def objective_runtime_metadata",
            },
        },
        "src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py": {
            "max_lines": 450,
            "forbidden": [
                "def _normalize_scenario_c_suite",
                "def _apply_scenario_c_async_position_feedback",
                "def aggregate_cxd_phase_diagram",
                "def _predictive_jepa_metric_row",
                "class OutputRegistry",
                "def _write_benchmark_figures",
            ],
            "helpers": {
                "src/kd_sensing/diagnostics/jepa_benchmark_manifest.py": "def validate_benchmark_manifest",
                "src/kd_sensing/diagnostics/jepa_benchmark_scenario_c.py": "def _apply_scenario_c_async_position_feedback",
                "src/kd_sensing/diagnostics/jepa_benchmark_scenario_d.py": [
                    "def aggregate_cxd_phase_diagram",
                    "def compute_modality_dominance",
                ],
                "src/kd_sensing/diagnostics/jepa_benchmark_runner.py": [
                    "def run_jepa_gps_shortcut_benchmark",
                    "def aggregate_robustness_summary",
                    "def _build_runner_manifest",
                ],
                "src/kd_sensing/diagnostics/jepa_benchmark_predictive.py": "def _predictive_jepa_metric_row",
                "src/kd_sensing/diagnostics/jepa_benchmark_perturbations.py": "def apply_benchmark_perturbation",
                "src/kd_sensing/diagnostics/jepa_benchmark_artifacts.py": "class OutputRegistry",
                "src/kd_sensing/diagnostics/jepa_benchmark_plots.py": "def _write_benchmark_figures",
            },
        },
        "src/kd_sensing/data/mmw/preparation.py": {
            "max_lines": 250,
            "forbidden": [
                "class MMWPreparationConfig",
                "class SensorFrame",
                "def index_sensor_frames",
                "def build_sequence_rows",
                "def derive_beam_power",
                "def _write_manifest_csv",
                "def build_relative_geometry",
                "def write_data_availability",
            ],
            "helpers": {
                "src/kd_sensing/data/mmw/preparation_config.py": "class MMWPreparationConfig",
                "src/kd_sensing/data/mmw/preparation_audit.py": "def validate_zip_inputs",
                "src/kd_sensing/data/mmw/preparation_index.py": "class PreparedFrame",
                "src/kd_sensing/data/mmw/preparation_splits.py": "def split_sequence_rows",
                "src/kd_sensing/data/mmw/preparation_beam_power.py": "def derive_beam_power",
                "src/kd_sensing/data/mmw/preparation_writers.py": "def build_prepared_artifacts",
                "src/kd_sensing/data/mmw/preparation_geometry.py": "def build_relative_geometry",
            },
        },
    }

    for facade, expectation in expectations.items():
        text = (ROOT / facade).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= expectation["max_lines"]
        assert [snippet for snippet in expectation["forbidden"] if snippet in text] == []
        for helper, snippets in expectation["helpers"].items():
            helper_text = (ROOT / helper).read_text(encoding="utf-8")
            expected_snippets = [snippets] if isinstance(snippets, str) else snippets
            for snippet in expected_snippets:
                assert snippet in helper_text


def test_first_batch_hotspot_facades_are_not_internal_helper_import_sources():
    forbidden = {
        "kd_sensing.data.mmw.preparation": {
            "build_sequence_rows": "kd_sensing.data.mmw.preparation_splits",
            "split_sequence_rows": "kd_sensing.data.mmw.preparation_splits",
            "compute_split_leakage_diagnostics": "kd_sensing.data.mmw.preparation_splits",
            "derive_beam_power_from_file": "kd_sensing.data.mmw.preparation_beam_power",
            "derive_beam_power": "kd_sensing.data.mmw.preparation_beam_power",
            "index_sensor_frames": "kd_sensing.data.mmw.preparation_index",
            "index_channel_files": "kd_sensing.data.mmw.preparation_index",
            "validate_zip_inputs": "kd_sensing.data.mmw.preparation_audit",
            "write_data_availability": "kd_sensing.data.mmw.preparation_audit",
        },
        "kd_sensing.diagnostics.jepa_gps_shortcut_benchmark": {
            "_normalize_scenario_c_suite": "kd_sensing.diagnostics.jepa_benchmark_scenario_c",
            "_apply_scenario_c_async_position_feedback": "kd_sensing.diagnostics.jepa_benchmark_scenario_c",
            "apply_benchmark_perturbation": "kd_sensing.diagnostics.jepa_benchmark_perturbations",
            "aggregate_cxd_phase_diagram": "kd_sensing.diagnostics.jepa_benchmark_scenario_d",
            "compute_modality_dominance": "kd_sensing.diagnostics.jepa_benchmark_scenario_d",
            "_predictive_jepa_metric_row": "kd_sensing.diagnostics.jepa_benchmark_predictive",
            "OutputRegistry": "kd_sensing.diagnostics.jepa_benchmark_artifacts",
            "_write_benchmark_figures": "kd_sensing.diagnostics.jepa_benchmark_plots",
        },
    }
    public_compat_allowed = {
        ROOT / "src" / "kd_sensing" / "data" / "mmw" / "preparation.py",
        ROOT / "src" / "kd_sensing" / "data" / "mmw" / "__init__.py",
        ROOT / "scripts" / "mmw" / "prepare_town10_skybridge.py",
        ROOT / "scripts" / "mmw" / "build_sequence_splits_from_manifest.py",
    }
    violations = []

    for path in (ROOT / "src").rglob("*.py"):
        if path in public_compat_allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in forbidden:
                continue
            imported = {alias.name for alias in node.names}
            for name, recommended in forbidden[node.module].items():
                if name in imported:
                    rel = path.relative_to(ROOT).as_posix()
                    violations.append(
                        f"{rel} imports {name} from {node.module}; import from {recommended} instead."
                    )

    assert violations == []


def test_duplicate_manifest_fallback_wrapper_is_not_reintroduced():
    wrapper = ROOT / "tools" / "visualization" / "export_viewer_manifest.py"
    assert not wrapper.exists()

    violations = []
    for root in [ROOT / "scripts", ROOT / "tools"]:
        for path in root.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            if "from kd_sensing.cli.export_viewer_manifest import main" in text:
                violations.append(rel)

    assert violations == []


def test_retired_bgam_viewer_console_scripts_are_not_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    retired_fragments = (
        "export-viewer-manifest",
        "visualize-modalities",
        "gps-lidar-bgam",
        "top8-candidate-manifest",
    )
    violations = [fragment for fragment in retired_fragments if fragment in pyproject]

    assert violations == []


def test_openspec_specs_have_real_purpose_text():
    violations = []
    for path in sorted((ROOT / "openspec" / "specs").glob("*/spec.md")):
        purpose = _openspec_purpose_text(path)
        if not purpose or len(purpose) < 50 or "TBD" in purpose:
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def _openspec_purpose_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "## Purpose":
            continue
        body: list[str] = []
        for item in lines[index + 1 :]:
            if item.startswith("## "):
                break
            stripped = item.strip()
            if stripped:
                body.append(stripped)
        return " ".join(body)
    return ""


@pytest.mark.parametrize(
    "statement",
    [
        "import kd_sensing.config",
        "from kd_sensing.utils.paths import resolve_path",
        "import kd_sensing.registries",
    ],
)
def test_light_imports_do_not_import_default_components(statement: str):
    modules = _run_import_probe(statement)

    assert modules == {
        "artifact_registry": False,
        "diagnostics": False,
        "models": False,
        "deepsense6g": False,
        "synthetic": False,
    }


def test_config_import_does_not_import_runtime_boundaries():
    modules = _run_architecture_boundary_probe("import kd_sensing.config")

    assert modules == {
        "torch": False,
        "data_factory": False,
        "deepsense6g": False,
        "models": False,
        "visualization_core": False,
        "trainer": False,
    }


def test_retired_top8_residual_modules_are_not_importable():
    modules = _run_module_presence_probe(
        """
import importlib.util
for module in (
    "kd_sensing.cli.inspect_deepsense6g_residual_inputs",
    "kd_sensing.cli.prepare_deepsense6g_top8_candidate_manifest",
    "kd_sensing.cli.gps_coarse_anchor",
    "kd_sensing.cli.export_viewer_manifest",
    "kd_sensing.cli.visualize_modalities",
    "kd_sensing.cli.prepare_deepsense6g_gps_lidar_bgam_manifest",
    "kd_sensing.cli.run_deepsense6g_gps_lidar_bgam",
    "kd_sensing.cli.evaluate_deepsense6g_gps_lidar_bgam",
    "kd_sensing.cli.prepare_mmw_town_gps_lidar_bgam_manifest",
    "kd_sensing.cli.run_mmw_town_gps_lidar_bgam",
    "kd_sensing.cli.evaluate_mmw_town_gps_lidar_bgam",
    "kd_sensing.data.deepsense6g_residual",
    "kd_sensing.data.deepsense6g_topk_candidate_manifest",
    "kd_sensing.data.mmw_town_topk_candidate_manifest",
    "kd_sensing.data.deepsense6g_gps_lidar_bgam_dataset",
    "kd_sensing.data.deepsense6g_gps_lidar_bgam_manifest",
    "kd_sensing.data.mmw_town_gps_lidar_bgam_manifest",
    "kd_sensing.engine.gps_coarse_anchor",
    "kd_sensing.engine.deepsense6g_gps_lidar_bgam",
    "kd_sensing.engine.mmw_town_gps_lidar_bgam",
    "kd_sensing.models.topk_candidate_selector",
    "kd_sensing.models.gps_lidar_bgam",
    "kd_sensing.models.gps_lidar_bgam_model",
    "kd_sensing.losses.residual",
    "kd_sensing.losses.gps_lidar_bgam_losses",
    "kd_sensing.losses.topk_candidate_losses",
    "kd_sensing.diagnostics.viewer_manifest",
    "kd_sensing.diagnostics.viewer_manifest_cache",
    "kd_sensing.diagnostics.viewer_manifest_config",
    "kd_sensing.diagnostics.viewer_manifest_datasets",
    "kd_sensing.diagnostics.viewer_manifest_merge",
    "kd_sensing.diagnostics.viewer_manifest_paths",
    "kd_sensing.diagnostics.viewer_manifest_sampling",
    "kd_sensing.diagnostics.viewer_manifest_schema",
    "kd_sensing.diagnostics.viewer_manifest_stats",
    "kd_sensing.diagnostics.viewer_manifest_writer",
    "kd_sensing.diagnostics.viewer_predictions",
):
    assert importlib.util.find_spec(module) is None, module
""",
        {
            "trainer": "kd_sensing.engine.trainer",
            "data_factory": "kd_sensing.engine.data_factory",
        },
    )

    assert modules == {
        "trainer": False,
        "data_factory": False,
    }


def test_engine_light_submodule_does_not_import_heavy_boundaries():
    modules = _run_module_presence_probe(
        "import kd_sensing.engine.model_output",
        {
            "builder_aggregate": _dotted("kd_sensing", "engine", "_builders_impl"),
            "transform_aggregate": _dotted("kd_sensing", "data", "transform_ops", "_legacy"),
            "pandas": "pandas",
            "scipy": "scipy",
        },
    )

    assert modules == {
        "builder_aggregate": False,
        "transform_aggregate": False,
        "pandas": False,
        "scipy": False,
    }


def test_current_diagnostics_light_submodule_does_not_import_visualization_stack():
    modules = _run_module_presence_probe(
        "import kd_sensing.diagnostics.run_index",
        {
            "matplotlib": "matplotlib",
            "viewer_manifest": "kd_sensing.diagnostics.viewer_manifest",
            "visualization_core": "kd_sensing.diagnostics.visualization.core",
        },
    )

    assert modules == {
        "matplotlib": False,
        "viewer_manifest": False,
        "visualization_core": False,
    }


def test_beam_loss_submodule_does_not_import_training_registry_or_transforms():
    modules = _run_module_presence_probe(
        "import kd_sensing.losses",
        {
            "builder_facade": _dotted("kd_sensing", "engine", "builders"),
            "builder_aggregate": _dotted("kd_sensing", "engine", "_builders_impl"),
            "transform_aggregate": _dotted("kd_sensing", "data", "transform_ops", "_legacy"),
        },
    )

    assert modules == {
        "builder_facade": False,
        "builder_aggregate": False,
        "transform_aggregate": False,
    }


def test_lazy_package_exports_remain_available():
    _run_module_presence_probe(
        "\n".join(
            [
                "from kd_sensing.engine import train",
                "from kd_sensing.losses import FocalLoss",
                "assert train is not None",
                "assert FocalLoss is not None",
            ]
        ),
        {},
    )


def test_models_package_import_is_lazy_and_public_symbols_remain_available():
    modules = _run_module_presence_probe(
        "import kd_sensing.models",
        {
            "fusion": "kd_sensing.models.fusion",
            "gps": "kd_sensing.models.gps",
            "image": "kd_sensing.models.image",
            "lidar": "kd_sensing.models.lidar",
            "mmwave": "kd_sensing.models.mmwave",
            "radar": "kd_sensing.models.radar",
            "modular": "kd_sensing.models.modular",
            "bev_fusion_2604": "kd_sensing.models.bev_fusion_2604",
        },
    )

    assert modules == {key: False for key in modules}

    _run_module_presence_probe(
        "\n".join(
            [
                "from kd_sensing.models import FusionStrongModalityNet, GpsStrongModalityNet",
                "assert FusionStrongModalityNet is not None",
                "assert GpsStrongModalityNet is not None",
                "import kd_sensing.models as models",
                "assert 'FusionStrongModalityNet' in models.__all__",
                "assert 'BEVFusion2604Net' in models.__all__",
                "try:",
                "    getattr(models, 'Fusion' + 'ModalityNet')",
                "except AttributeError as exc:",
                "    assert 'FusionStrongModalityNet' in str(exc)",
                "else:",
                "    raise AssertionError('removed alias did not raise')",
            ]
        ),
        {},
    )


def test_modality_contract_normalizes_and_validates_modalities():
    assert MODALITY_ORDER == ("image", "radar", "gps", "lidar", "mmwave", "csi")
    assert normalize_modalities(["csi", "lidar", "image", "gps"]) == ("image", "gps", "lidar", "csi")

    with pytest.raises(ValueError, match="thermal"):
        normalize_modalities(["image", "thermal"])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_modalities(["gps", "gps"])
    with pytest.raises(ValueError, match="at least one"):
        normalize_modalities([])


def test_modality_contract_derives_dataset_flags_and_batch_keys():
    assert dataset_flags_for_modalities(["mmwave", "gps", "csi"]) == {
        "use_gps": True,
        "use_lidar": False,
        "use_mmwave": True,
        "use_csi": True,
    }
    assert batch_input_keys_for_modalities(["radar", "mmwave", "csi", "gps"]) == {
        "radar": "radar_batch",
        "mmwave": "mmwave_batch",
        "csi": "csi_batch",
        "gps": "gps_batch",
    }


def test_transform_helpers_are_available_from_narrow_modules():
    from kd_sensing.data.transform_ops.gps import GPSStandardScaler
    from kd_sensing.data.transform_ops.image import build_image_transform
    from kd_sensing.data.transform_ops.lidar import read_lidar_point_cloud

    assert GPSStandardScaler is not None
    assert build_image_transform is not None
    assert read_lidar_point_cloud is not None


def test_removed_facades_are_not_importable():
    import importlib

    for module_name in [
        _dotted("kd_sensing", "engine", "builders"),
        _dotted("kd_sensing", "engine", "_builders_impl"),
        _dotted("kd_sensing", "data", "transforms"),
        _dotted("kd_sensing", "data", "transform_ops", "_legacy"),
        _dotted("kd_sensing", "data", "datasets", "multimodal_nf"),
        _dotted("kd_sensing", "data", "deepverse"),
        _dotted("kd_sensing", "baselines", "gps_window"),
        _dotted("kd_sensing", "cli", "gps_window_baseline"),
        _dotted("kd_sensing", "diagnostics", "g2d_diagnostics"),
        _dotted("kd_sensing", "diagnostics", "visualization"),
        _dotted("kd_sensing", "distillation", "g2d"),
        _dotted("kd_sensing", "distillation", "g2d_smp"),
        _dotted("kd_sensing", "distillation", "teacher_ensemble"),
        _dotted("kd_sensing", "engine", "craf_training"),
        _dotted("kd_sensing", "engine", "g2d_training"),
        _dotted("kd_sensing", "engine", "marf_training"),
        _dotted("kd_sensing", "engine", "multimodal_nf_runtime"),
        _dotted("kd_sensing", "models", "fusion", "craf"),
        _dotted("kd_sensing", "models", "fusion", "marf"),
        _dotted("kd_sensing", "preprocessing", "multimodal_nf"),
        _dotted("kd_sensing", "preprocessing", "multimodal_nf_common"),
    ]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_internal_python_code_avoids_secondary_compatibility_layers():
    roots = [ROOT / "src" / "kd_sensing", ROOT / "scripts", ROOT / "tools"]
    forbidden_snippets = (
        f"from {_dotted('kd_sensing', 'engine', 'builders')} import",
        f"import {_dotted('kd_sensing', 'engine', 'builders')}",
        "from kd_sensing.engine import builders",
        _dotted("kd_sensing", "engine", "_builders_impl"),
        "from kd_sensing.engine import _builders_impl",
        _dotted("kd_sensing", "data", "transform_ops", "_legacy"),
        "from kd_sensing.data.transform_ops import _legacy",
        "from kd_sensing.engine.objective_metadata import",
        "import kd_sensing.engine.objective_metadata",
        "from kd_sensing.preprocessing.multimodal_nf_common import",
        "import kd_sensing.preprocessing.multimodal_nf_common",
        "from kd_sensing.diagnostics.viewer_manifest import _",
        "from kd_sensing.diagnostics import export_viewer_manifest",
    )
    violations = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for snippet in forbidden_snippets:
                if snippet in text:
                    violations.append(f"{path.relative_to(ROOT)} contains {snippet}")

    assert violations == []


def test_training_methods_are_connected_through_engine_extensions():
    trainer_text = (SRC / "kd_sensing" / "engine" / "trainer.py").read_text(encoding="utf-8")
    trainer_runtime_text = (SRC / "kd_sensing" / "engine" / "trainer_runtime_helpers.py").read_text(
        encoding="utf-8"
    )
    batch_step_text = (SRC / "kd_sensing" / "engine" / "batch_step.py").read_text(encoding="utf-8")
    forbidden = (
        "def _compute_craf_extra_losses",
        "def _compute_marf_extra_losses",
        "def _counterfactual_gate_loss",
        "G2DDiagnosticsAccumulator",
        "build_g2d_teacher_ensemble",
        "apply_smp_gradient_mask",
        "beam_soft_label_loss",
        "marf_residual_norm_loss",
    )

    assert [snippet for snippet in forbidden if snippet in trainer_text] == []
    assert "G2DTrainingExtension" not in trainer_text
    assert "CrafTrainingExtension" not in trainer_text
    assert "MarfTrainingExtension" not in trainer_text
    assert "BatchStepRunner" in trainer_text
    assert "extension.after_epoch" in trainer_text + trainer_runtime_text
    assert "extension.after_forward" in batch_step_text
    for rel_path in [
        "engine/g2d_training.py",
        "engine/craf_training.py",
        "engine/marf_training.py",
        "diagnostics/g2d_diagnostics.py",
        "distillation/g2d.py",
        "distillation/g2d_smp.py",
        "distillation/teacher_ensemble.py",
    ]:
        assert not (SRC / "kd_sensing" / rel_path).exists()


def test_training_orchestration_helpers_own_runtime_details():
    trainer_text = (SRC / "kd_sensing" / "engine" / "trainer.py").read_text(encoding="utf-8")
    helper_expectations = {
        "batch_step.py": "class BatchStepRunner",
        "training_metrics.py": "class EpochMetricsRecorder",
        "checkpointing.py": "class CheckpointManager",
        "artifacts.py": "class ArtifactWriter",
        "tensorboard_logging.py": "def write_tensorboard_scalars",
        "training_state.py": "class TrainingState",
    }

    assert "BatchStepRunner" in trainer_text
    assert "EpochMetricsRecorder" in trainer_text
    assert "CheckpointManager" in trainer_text
    assert "ArtifactWriter" in trainer_text
    assert "compute_prediction_loss" not in trainer_text
    assert "torch.save(" not in trainer_text
    assert "np.savez" not in trainer_text
    assert "SummaryWriter" not in trainer_text
    for module, snippet in helper_expectations.items():
        assert snippet in (SRC / "kd_sensing" / "engine" / module).read_text(encoding="utf-8")


def test_active_mainline_modules_do_not_import_legacy_kd_runtime_aggregate():
    active_modules = [
        "src/kd_sensing/engine/batch_step.py",
        "src/kd_sensing/engine/evaluation_pass.py",
        "src/kd_sensing/engine/evaluator.py",
        "src/kd_sensing/engine/validator.py",
        "src/kd_sensing/engine/objectives/history.py",
        "src/kd_sensing/engine/run_metadata.py",
        "src/kd_sensing/engine/training_extensions.py",
    ]
    forbidden_snippets = (
        "from kd_sensing.distillation.distillers import",
        "import kd_sensing.distillation.distillers",
        "from kd_sensing.distillation import KnowledgeDistillationLoss",
        "DISTILLERS.build(",
    )
    violations = []

    for rel_path in active_modules:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in text:
                violations.append(
                    f"{rel_path} imports legacy KD runtime '{snippet}'; "
                    "use the no-KD objective/method extension path or the explicit legacy builder in engine.optim."
                )

    assert violations == []


def test_retired_hist_engine_model_and_evaluation_sources_are_absent():
    retired_files = [
        "src/kd_sensing/cli/hist_beam_loso.py",
        "src/kd_sensing/engine/hist_beam_adaptation.py",
        "src/kd_sensing/engine/hist_beam_history_anchor.py",
        "src/kd_sensing/engine/hist_beam_loso_execution.py",
        "src/kd_sensing/engine/hist_beam_losses.py",
        "src/kd_sensing/engine/hist_beam_prototypes.py",
        "src/kd_sensing/engine/hist_beam_training.py",
        "src/kd_sensing/evaluation/hist_beam_outputs.py",
        "src/kd_sensing/evaluation/hist_beam_residuals.py",
        "src/kd_sensing/models/fusion/hist_beam.py",
    ]
    for rel_path in retired_files:
        assert not (ROOT / rel_path).exists()

    forbidden_imports = (
        "kd_sensing.engine.hist_beam_",
        "kd_sensing.evaluation.hist_beam_",
        "kd_sensing.models.fusion.hist_beam",
    )
    violations = []
    for path in (SRC / "kd_sensing").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_imports:
            if snippet in text:
                violations.append(f"{path.relative_to(ROOT)} contains retired Hist import '{snippet}'")
    assert violations == []


def test_config_io_pipeline_delegates_business_rules():
    io_text = (SRC / "kd_sensing" / "config" / "io.py").read_text(encoding="utf-8")
    helper_expectations = (
        ("source.py", "def load_config_source"),
        ("normalization.py", "def normalize_loaded_config"),
        ("validation.py", "def validate_loaded_config"),
        ("migration_guards.py", "def reject_removed_image_path_config"),
        ("migration_guards.py", "def reject_retired_raymobtime_config"),
    )

    assert "load_config_source" in io_text
    assert "normalize_loaded_config" in io_text
    assert "validate_loaded_config" in io_text
    assert "reject_removed_image_path_config" in io_text
    assert "build_virtual_config" not in io_text
    assert "future_beam" not in io_text
    assert "REMOVED_IMAGE_ENCODERS" not in io_text
    assert "snapshot_next_frame requires" not in io_text
    assert "configure_objective_defaults" not in io_text
    for module, snippet in helper_expectations:
        assert snippet in (SRC / "kd_sensing" / "config" / module).read_text(encoding="utf-8")


def test_prediction_task_boundaries_do_not_reintroduce_duplicate_tables_or_validation_paths():
    trainer_text = (SRC / "kd_sensing" / "engine" / "trainer.py").read_text(encoding="utf-8")
    validator_text = (SRC / "kd_sensing" / "engine" / "validator.py").read_text(encoding="utf-8")
    evaluator_text = (SRC / "kd_sensing" / "engine" / "evaluator.py").read_text(encoding="utf-8")

    assert "_EARLY_STOPPING_METRIC_ALIASES" not in trainer_text
    assert "_MAX_EARLY_STOPPING_METRICS" not in trainer_text
    assert "default_primary_metric" not in trainer_text
    assert "prepare_task_labels" not in validator_text
    assert "run_model_step" not in validator_text
    assert "compute_prediction_loss" not in validator_text
    assert "_cfg_uses_lidar" not in validator_text
    assert "_evaluation_uses_gps" not in evaluator_text
    assert "_evaluation_uses_lidar" not in evaluator_text
    assert "_evaluation_uses_mmwave" not in evaluator_text
    assert "run_evaluation_pass" in validator_text


def test_visualize_modalities_console_script_is_retired():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    alias_path = SRC / "kd_sensing" / "cli" / "visualize_modalities.py"

    assert "kd-sensing-visualize-modalities" not in pyproject
    assert not alias_path.exists()
