from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.modalities import (  # noqa: E402
    MODALITY_ORDER,
    batch_input_keys_for_modalities,
    dataset_flags_for_modalities,
    normalize_modalities,
)
from kd_sensing.config.defaults import DEFAULT_CONFIG  # noqa: E402
from kd_sensing.utils.runtime_output_layout import canonical_runtime_partitions  # noqa: E402

PYTHON_ENTRYPOINT_ALLOWLIST = {
    "scripts/analyze_csi_hardening_sweep.py": "research_diagnostic",
    "scripts/analysis/beambench_ae_gps_diagnostics.py": "research_diagnostic",
    "scripts/analysis/visualize_deepsense_beambench_correspondence.py": "research_diagnostic",
    "scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py": "research_diagnostic",
    "scripts/debug_eval_consistency.py": "research_diagnostic",
    "scripts/check_dataset.py": "dataset_preparation",
    "scripts/eval_baseline.py": "thin_cli_alias",
    "scripts/evaluate.py": "thin_cli_alias",
    "scripts/figures/draw_jepa_architecture.py": "research_diagnostic",
    "scripts/inspect_dataset.py": "dataset_preparation",
    "scripts/mmw/build_sequence_splits_from_manifest.py": "dataset_preparation",
    "scripts/mmw/prepare_town10_skybridge.py": "dataset_preparation",
    "scripts/mmw/visualize_gps_angle_beam_correspondence.py": "research_diagnostic",
    "scripts/mmw/visualize_gps_prediction_trajectory.py": "research_diagnostic",
    "scripts/mmw/visualize_prediction_error_label_distribution.py": "research_diagnostic",
    "scripts/mmw/visualize_town_label_distribution.py": "dataset_preparation",
    "scripts/preprocess.py": "thin_cli_alias",
    "scripts/profile_training_io.py": "research_diagnostic",
    "scripts/recommend_parallel_training.py": "research_diagnostic",
    "scripts/train_baseline.py": "thin_cli_alias",
    "scripts/train_beambench_image_ae_gps.py": "thin_cli_alias",
    "scripts/run_beambench_image_ae_gps_tableiii.py": "thin_cli_alias",
    "scripts/train.py": "thin_cli_alias",
}
SHELL_ORCHESTRATION_ALLOWLIST = {
    "scripts/run_csi_hardening_matrix.sh": "shell_orchestration",
    "scripts/run_deepsense_gps_circular_soft_label.sh": "shell_orchestration",
    "scripts/run_mmw_gps_circular_soft_label_ablation.sh": "shell_orchestration",
    "scripts/run_mmw_sunny_modal15_l5p3_h123.sh": "shell_orchestration",
    "scripts/run_mmw_sunny_modal15_l5p6_h246.sh": "shell_orchestration",
}
ENTRYPOINT_LIFECYCLES = {
    "package_cli",
    "thin_cli_alias",
    "research_diagnostic",
    "dataset_preparation",
    "shell_orchestration",
}
RETIRED_GENERATED_FUSION_CONFIGS = {
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml",
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml",
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml",
    "configs/fusion/craf_all_modalities_no_kd.yaml",
    "configs/fusion/craf_all_modalities_no_counterfactual.yaml",
    "configs/fusion/craf_all_modalities_fixed_prior_sanity.yaml",
    "configs/fusion/marf.yaml",
    "configs/fusion/marf_subset_training.yaml",
    "configs/fusion/marf_no_residual_ablation.yaml",
    "configs/fusion/marf_no_prior_bias_ablation.yaml",
    "configs/fusion/marf_no_subset_training_ablation.yaml",
}
FUSION_ROOT_YAML_ALLOWLIST = {
    "configs/fusion/all_modalities_lidar_supervised.yaml",
    "configs/fusion/all_modalities_supervised.yaml",
    "configs/fusion/beambench_image_ae_gps_direct.yaml",
    "configs/fusion/image_gps_resnet18_modular_supervised.yaml",
    "configs/fusion/image_gps_supervised.yaml",
    "configs/fusion/mmwave_csi_medium_degraded_supervised.yaml",
    "configs/fusion/mmwave_csi_supervised.yaml",
    "configs/fusion/radar_gps_supervised.yaml",
    "configs/fusion/radar_lidar_supervised.yaml",
    "configs/fusion/token_transformer_all_modalities_multitask_supervised.yaml",
    "configs/fusion/token_transformer_all_modalities_supervised.yaml",
    "configs/fusion/token_transformer_image_radar_supervised.yaml",
}
EXISTING_MODEL_REGISTRATION_ALLOWLIST = {
    "bev_fusion_2604",
    "camera_ae_frozen",
    "cls_token_transformer_fusion",
    "fusion_lightweight",
    "fusion_strong",
    "gps_conditioned_jepa",
    "gps_lightweight",
    "gps_only_neural_baseline",
    "gps_sequence_baseline",
    "gps_strong",
    "image_lightweight",
    "image_strong",
    "jepa_context_image",
    "lidar_feature_extractor",
    "lidar_lightweight",
    "lidar_strong",
    "mmwave_feature_extractor",
    "mmwave_lightweight",
    "mmwave_strong",
    "modular_sequence",
    "modular_sequence_model",
    "radar_feature_extractor",
    "radar_lightweight",
    "radar_strong",
    "resnet18_imagenet_rgb",
    "token_transformer_fusion",
    "vision_position_late_fusion",
    "vision_position_transformer_fusion",
}
BATCH_RUNTIME_FUNCTION_ALLOWLIST = {
    "src/kd_sensing/engine/batch.py": {
        "forward_model",
        "prepare_auxiliary_targets",
        "prepare_beam_power_targets",
        "prepare_beamspace_power_targets",
        "prepare_csi_inputs",
        "prepare_fusion_inputs",
        "prepare_geometry_inputs",
        "prepare_geometry_mask",
        "prepare_gps_bev_xy_inputs",
        "prepare_gps_inputs",
        "prepare_image_inputs",
        "prepare_labels",
        "prepare_lidar_inputs",
        "prepare_mmwave_inputs",
        "prepare_path_descriptors",
        "prepare_path_semantic_labels",
        "prepare_radar_inputs",
        "prepare_radio_semantic_labels",
        "prepare_reliability_metadata_inputs",
        "prepare_soft_beam_targets",
    },
    "src/kd_sensing/engine/runtime.py": {
        "forward_task_model",
        "prepare_task_auxiliary_targets",
        "prepare_task_batch",
        "prepare_task_inputs",
        "prepare_task_labels",
        "prepare_task_soft_beam_targets",
    },
    "src/kd_sensing/engine/validator.py": {
        "_resolve_validation_prior",
        "_validate_modality_subsets",
        "_validate_with_force_mask",
        "validate",
    },
}
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

HEALTH_CHECK_COMMANDS = (
    "openspec validate strengthen-project-health-guardrails --strict",
    "conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q",
    "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q",
)
CONFIG_LIFECYCLE_MARKERS = (
    "configs/<image|radar|gps|lidar|mmwave|csi>/{strong,lightweight,supervised}.yaml",
    "configs/fusion/experiments/jepa_image_gps/*.yaml",
    "configs/csi/hardening_matrix/*.yaml",
    "configs/csi/hardening_matrix/debug/*.yaml",
    "configs/fusion/csi_hardening_matrix/*.yaml",
    "configs/preprocess/*.yaml",
    "configs/diagnostics/modality_visualization.yaml",
    "configs/diagnostics/jepa_visual_analysis_2604.yaml",
    "configs/diagnostics/jepa_gps_shortcut_benchmark_*.yaml",
    "configs/baselines/*.yaml",
    "configs/pretraining/*.yaml",
    "retired history",
)
HOTSPOT_SYMBOL_BUDGETS = {
    ("src/kd_sensing/data/datasets/deepsense6g.py", "DeepSense6GDataset", "class"): 1210,
    ("src/kd_sensing/data/datasets/deepsense6g.py", "__init__", "function"): 265,
    ("src/kd_sensing/data/datasets/mmw.py", "MMWDataset", "class"): 600,
    ("src/kd_sensing/engine/trainer.py", "_train_inner", "function"): 320,
    ("src/kd_sensing/engine/mmw_town_gps_v2.py", "run_mmw_town_gps_v2", "function"): 280,
    ("src/kd_sensing/baselines/beambench/image_ae_gps.py", "run_image_ae_gps_training", "function"): 245,
    (
        "src/kd_sensing/baselines/beambench/image_ae_gps.py",
        "run_image_ae_gps_paper_split_training",
        "function",
    ): 265,
    (
        "src/kd_sensing/engine/deepsense6g_gps_lidar_bgam.py",
        "run_deepsense6g_gps_lidar_bgam",
        "function",
    ): 240,
    ("src/kd_sensing/engine/evaluation_pass.py", "run_evaluation_pass", "function"): 220,
}
REQUIRED_HOTSPOT_INVENTORY_MARKERS = (
    "src/kd_sensing/data/datasets/deepsense6g.py",
    "DeepSense6GDataset",
    "src/kd_sensing/data/datasets/mmw.py",
    "MMWDataset",
    "src/kd_sensing/engine/trainer.py",
    "_train_inner",
    "src/kd_sensing/baselines/beambench/image_ae_gps.py",
    "run_image_ae_gps_training",
    "run_image_ae_gps_paper_split_training",
    "src/kd_sensing/engine/evaluation_pass.py",
    "run_evaluation_pass",
    "src/kd_sensing/engine/batch.py",
    "src/kd_sensing/diagnostics/run_index.py",
    "src/kd_sensing/diagnostics/viewer_manifest.py",
)
LONG_FUNCTION_LIMIT = 230
LONG_CLASS_LIMIT = 590
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
RETIRED_CONFIG_TOKENS = (
    "hist_beam",
    "top8",
    "residual",
    "camera_residual",
    "coarse_anchor",
    "logits_kd",
    "rkd",
    "teacher_no_kd",
    "student_no_kd",
    "no_kd",
    "craf",
    "marf",
    "g2d",
    "multimodal_nf",
    "image_motion",
    "raymobtime",
    "raymobtime_s008",
)
ROOT_DOCUMENT_LIFECYCLE_EXCLUSIONS = set()
CURRENT_DOCS_TO_CHECK_FOR_RETIRED_RECOMMENDATIONS = (
    ROOT / "README.md",
    ROOT / "docs" / "experiment_matrix.md",
    ROOT / "docs" / "extension_guide.md",
    ROOT / "docs" / "training_throughput.md",
)
RETIRED_ROUTE_TEXT_MARKERS = (
    "HiST-Beam",
    "Top8 selector",
    "GPS residual",
    "camera residual",
    "logits_kd",
    "rkd",
    "teacher_no_kd",
    "student_no_kd",
    "no_kd",
    "Raymobtime s008",
    "raymobtime_s008",
)
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
OPENSPEC_LIFECYCLE_ALLOWED = {"current", "supporting", "retired-tombstone"}
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
    return sorted(path for path in result.stdout.decode("utf-8").split("\0") if path)


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


def _nearby_text(lines: list[str], line_index: int, radius: int = 12) -> str:
    start = max(0, line_index - radius)
    end = min(len(lines), line_index + radius + 1)
    return "\n".join(lines[start:end])


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


def test_amr_net_gps_image_quickstart_stays_synchronized():
    matrix = (ROOT / "docs" / "experiment_matrix.md").read_text(encoding="utf-8")
    required_matrix_markers = (
        "AMR-Net_gps_image",
        "configs/baselines/amr_net_gps_image.yaml",
        "kd-sensing-run-amr-net-gps-image",
        "blocked_official",
        "10000718",
        "LiDAR",
    )
    missing_matrix_markers = [marker for marker in required_matrix_markers if marker not in matrix]
    assert missing_matrix_markers == []

    for rel_path in (
        "docs/mainline_model_catalog.md",
        "docs/experiment_protocols.md",
        "docs/result_claims_registry.md",
        "docs/project_surface_inventory.md",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "AMR-Net_gps_image" in text, f"{rel_path} must mention AMR-Net_gps_image when the quickstart does"


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
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/agent_navigation.md" in agents
    assert "非平凡改动前" in agents
    assert "docs/agent_navigation.md" in inventory
    assert "current agent/maintainer navigation" in inventory
    assert "不替代 README、AGENTS 或 OpenSpec specs" in inventory
    assert "docs/agent_navigation.md" in readme


def test_hotspot_static_budget_matches_inventory():
    inventory = (ROOT / "docs" / "project_surface_inventory.md").read_text(encoding="utf-8")
    budget_keys = set(HOTSPOT_SYMBOL_BUDGETS)
    violations: list[str] = []

    for (rel_path, symbol, kind), max_lines in HOTSPOT_SYMBOL_BUDGETS.items():
        lengths = _symbol_lengths(ROOT / rel_path)
        actual = lengths.get((symbol, kind))
        if actual is None:
            violations.append(f"{rel_path}:{symbol} missing from AST scan")
            continue
        if actual > max_lines:
            violations.append(
                f"{rel_path}:{symbol} is {actual} lines, budget {max_lines}; "
                "split to a narrow module or update docs/project_surface_inventory.md with a reasoned budget."
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
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not _has_legacy_route_reference(line):
                continue
            if not LEGACY_ACTIVE_WORDING_RE.search(line):
                continue
            if _has_non_current_context(line):
                continue
            violations.append(
                f"{rel}:{line_number} describes a retired route with current/active wording: {line.strip()}"
            )

    assert violations == []


def test_retired_top8_residual_routes_are_not_current_source_modules():
    retired_paths = [
        ROOT / "configs" / "deepsense6g_residual_fusion.yaml",
        ROOT / "configs" / "deepsense6g_camera_residual.yaml",
        ROOT / "configs" / "deepsense6g_top8_selector.yaml",
        ROOT / "configs" / "mmw_town_top8_selector.yaml",
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
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_camera_residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "geometry_residual.py",
        ROOT / "src" / "kd_sensing" / "engine" / "gps_coarse_anchor.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_residual_fusion.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_top8_selector.py",
        ROOT / "src" / "kd_sensing" / "models" / "deepsense6g_residual_fusion.py",
        ROOT / "src" / "kd_sensing" / "models" / "topk_candidate_selector.py",
        ROOT / "src" / "kd_sensing" / "losses" / "residual.py",
        ROOT / "src" / "kd_sensing" / "data" / "datasets" / "raymobtime_s008.py",
        ROOT / "src" / "kd_sensing" / "preprocessing" / "raymobtime_s008.py",
        ROOT / "src" / "kd_sensing" / "models" / "raymobtime_s008.py",
        ROOT / "configs" / "raymobtime",
        ROOT / "docs" / "Raymobtime_s008_selection.md",
        ROOT / "tests" / "test_raymobtime_s008_selection.py",
    ]
    violations = [path.relative_to(ROOT).as_posix() for path in retired_paths if path.exists()]

    assert violations == []


def test_bgam_modules_stay_inside_package_boundaries():
    required_package_entries = [
        ROOT / "configs" / "deepsense6g_gps_lidar_bgam.yaml",
        ROOT / "configs" / "mmw_town_gps_lidar_bgam.yaml",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_deepsense6g_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "evaluate_deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "prepare_mmw_town_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "cli" / "run_mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "cli" / "evaluate_mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_dataset.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "data" / "mmw_town_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "engine" / "mmw_town_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam_model.py",
        ROOT / "src" / "kd_sensing" / "losses" / "gps_lidar_bgam_losses.py",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required_package_entries if not path.exists()]
    assert missing == []

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
        "src/kd_sensing/diagnostics/viewer_manifest.py",
        "src/kd_sensing/diagnostics/viewer_manifest_merge.py",
    ]

    for rel_path in required_paths:
        assert rel_path in inventory
    assert "不得从 `kd_sensing.engine.objective_metadata`" in inventory


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
        "src/kd_sensing/diagnostics/viewer_manifest.py": {
            "max_lines": 220,
            "forbidden": [
                "def _manifest_record",
                "def _cache_digest",
                "def _load_external_mapping",
                "def _save_raw_lidar_preview",
            ],
            "helpers": {
                "src/kd_sensing/diagnostics/viewer_manifest_schema.py": "def _json_ready",
                "src/kd_sensing/diagnostics/viewer_manifest_cache.py": "def _cache_digest",
                "src/kd_sensing/diagnostics/viewer_manifest_paths.py": "def _all_source_paths",
                "src/kd_sensing/diagnostics/viewer_manifest_merge.py": "def _attach_prediction_bundle",
                "src/kd_sensing/diagnostics/viewer_manifest_writer.py": "def _manifest_record",
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
        for helper, snippet in expectation["helpers"].items():
            assert snippet in (ROOT / helper).read_text(encoding="utf-8")


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
    "kd_sensing.data.deepsense6g_residual",
    "kd_sensing.engine.gps_coarse_anchor",
    "kd_sensing.models.topk_candidate_selector",
    "kd_sensing.losses.residual",
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


def test_viewer_manifest_light_helpers_do_not_import_runtime_stack():
    modules = _run_module_presence_probe(
        "\n".join(
            [
                "import kd_sensing.diagnostics.viewer_manifest_config",
                "import kd_sensing.diagnostics.viewer_manifest_sampling",
            ]
        ),
        {
            "matplotlib": "matplotlib",
            "pillow": "PIL.Image",
            "data_factory": "kd_sensing.engine.data_factory",
            "viewer_writer": "kd_sensing.diagnostics.viewer_manifest_writer",
            "visualization_core": "kd_sensing.diagnostics.visualization.core",
        },
    )

    assert modules == {
        "matplotlib": False,
        "pillow": False,
        "data_factory": False,
        "viewer_writer": False,
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
                "from kd_sensing.diagnostics import export_viewer_manifest",
                "from kd_sensing.losses import FocalLoss",
                "assert train is not None",
                "assert export_viewer_manifest is not None",
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
    assert "extension.after_epoch" in trainer_text
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


def test_visualize_modalities_console_script_is_thin_manifest_alias():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    alias_path = SRC / "kd_sensing" / "cli" / "visualize_modalities.py"
    alias_text = alias_path.read_text(encoding="utf-8")
    tree = ast.parse(alias_text)
    function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert 'kd-sensing-visualize-modalities = "kd_sensing.cli.visualize_modalities:main"' in pyproject
    assert function_names == ["main"]
    assert "argparse.ArgumentParser" not in alias_text
    assert "export_viewer_manifest_main" in alias_text

    command = shutil.which("kd-sensing-visualize-modalities")
    if command is not None:
        result = subprocess.run([command, "--help"], text=True, capture_output=True, check=False)
        assert result.returncode == 0
        assert "--cache-dir" in result.stdout
