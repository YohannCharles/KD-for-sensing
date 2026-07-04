import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INVENTORY = ROOT / "docs/project_surface_inventory.md"

CURRENT_CONFIG_GLOBS = (
    "configs/fusion/physics_informed_mmw*.yaml",
    "configs/fusion/experiments/jepa_image_gps/*.yaml",
    "configs/fusion/experiments/wcl2025_missing_modality/*.yaml",
    "configs/csi/hardening_matrix/*.yaml",
    "configs/csi/hardening_matrix/debug/*.yaml",
    "configs/fusion/csi_hardening_matrix/*.yaml",
    "configs/diagnostics/*.yaml",
    "configs/pretraining/*.yaml",
)

CURRENT_PATHS = (
    "configs/fusion/beambench_image_ae_gps_direct.yaml",
    "configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml",
    "configs/diagnostics/jepa_visual_analysis_2604.yaml",
    "configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml",
    "src/kd_sensing/engine/objectives/metadata.py",
    "src/kd_sensing/baselines/beambench/image_ae_gps_training.py",
    "src/kd_sensing/baselines/beambench/image_ae_gps_paper_split.py",
    "src/kd_sensing/baselines/rmbp_mm/workflow.py",
    "src/kd_sensing/cli/wcl2025_missing_modality.py",
    "src/kd_sensing/baselines/tii_vlrg_transformer.py",
    "src/kd_sensing/cli/tii_vlrg_transformer.py",
    "configs/baselines/tii_vlrg_transformer_reproduction.yaml",
    "docs/project_surface_inventory.md",
    "docs/maintainer_context_index.yaml",
)

DELETED_SURFACE_PATHS = (
    "src/kd_sensing/_typing.py",
    "src/kd_sensing/config/source.py",
    "src/kd_sensing/engine/objective_metadata.py",
    "src/kd_sensing/engine/objectives/history.py",
    "src/kd_sensing/engine/objectives/registry.py",
    "src/kd_sensing/data/dataset_runtime.py",
    "src/kd_sensing/data/transform_ops/normalization.py",
    "src/kd_sensing/config/canonical_recipes/common.py",
    "src/kd_sensing/config/canonical_recipes/advanced.py",
    "src/kd_sensing/config/canonical_recipes/fusion.py",
    "src/kd_sensing/config/canonical_recipes/objectives.py",
    "src/kd_sensing/config/canonical_recipes/__init__.py",
    "src/kd_sensing/models/fusion/networks.py",
    "src/kd_sensing/cli/beambench_check_dataset.py",
    "src/kd_sensing/baselines/beambench/image_ae_gps.py",
    "src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py",
    "src/kd_sensing/engine/loso_data.py",
    "configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml",
    "scripts/run_m2beam_single_modal_scene31_queue.sh",
    "scripts/run_rbma_strong_encoder_4gpu_queue.sh",
    "scripts/run_btapa_experiments.sh",
    "scripts/run_btapa_tau1_validation.sh",
    "scripts/run_csi_hardening_matrix.sh",
    "scripts/run_next_v3_experiments.sh",
    "scripts/run_night_grid_8gpu.sh",
    "scripts/run_proto_vs_btapa_8gpu.sh",
    "scripts/run_scene31_next_round.sh",
    "scripts/analyze_csi_hardening_sweep.py",
)

FORBIDDEN_IMPORTS = (
    "from kd_sensing._typing import",
    "from kd_sensing.config.source import",
    "from kd_sensing.engine.objective_metadata import",
    "import kd_sensing.engine.objective_metadata",
    "from kd_sensing.engine.objectives import",
    "from kd_sensing.data import",
    "from kd_sensing.data.datasets import",
    "from kd_sensing.data.transform_ops.normalization import",
    "from kd_sensing.baselines.beambench.image_ae_gps import",
    "import kd_sensing.baselines.beambench.image_ae_gps",
    "from kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep import",
    "import kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep",
    "from kd_sensing.engine.loso_data import",
    "import kd_sensing.engine.loso_data",
    "from kd_sensing.losses import",
    "from kd_sensing.baselines.rmbp_mm import",
    "from kd_sensing.data.mmw import",
    "from kd_sensing.engine import",
    "from kd_sensing.utils import",
    "from kd_sensing.preprocessing import",
    "from kd_sensing.evaluation import",
    "from kd_sensing.models.physics import",
    "from kd_sensing.diagnostics.jepa_benchmark_common import *",
    "from kd_sensing.models.fusion import",
)

RETIRED_TEXT_MARKERS = (
    "HiST-Beam",
    "Top8 selector",
    "GPS residual",
    "camera residual",
    "Raymobtime s008",
    "BGAM",
    "viewer manifest",
    "Gradio viewer",
    "CRAF",
    "MARF",
    "Multimodal-NF",
)

RETIREMENT_CONTEXT = (
    "退役",
    "历史",
    "拒绝",
    "墓碑",
    "防回流",
    "不得",
    "不再",
    "retired",
    "historical",
    "removed",
    "no longer",
    "tombstone",
)

SCENE31_MANIFEST_BACKED_CONFIG_ROOTS = (
    "configs/scene31/night_grid",
    "configs/scene31/next_round",
    "configs/scene31/funnel",
    "configs/scene31/magic_overnight",
)

SCENE31_RETAINED_YAML = {
    "configs/scene31/diagnostic_gps_only_strong.yaml",
    "configs/scene31/diagnostic_image_only_strong.yaml",
    "configs/scene31/diagnostic_lidar_only_strong.yaml",
    "configs/scene31/diagnostic_radar_only_strong.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_adba.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_fusiononly.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_modw1.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed3.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed3.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau4.yaml",
    "configs/scene31/main_v3_strong_reliability_proto.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_fullaux.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_fullaux_l05.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_hardgps.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_seed3.yaml",
    "configs/scene31/templates/main_v3_proto_es20_base.yaml",
    "configs/scene31/v4_weakkd_l01_t2.yaml",
    "configs/scene31/v4_weakkd_l02_t2.yaml",
    "configs/scene31/v4_weakkd_l03_t15.yaml",
}

SCENE31_LOCAL_MANUAL_RUNNERS = {
    "scripts/run_scene31_bc_apples_eval.sh",
    "scripts/run_scene31_bc_next.sh",
    "scripts/run_scene31_beamsoft_weak.sh",
    "scripts/run_scene31_funnel.sh",
    "scripts/run_scene31_magic_overnight.sh",
    "scripts/run_scene31_p0_fresh_eval.sh",
}

SCENE31_GENERATORS_AND_SUMMARIES = {
    "scripts/generate_scene31_funnel.py",
    "scripts/generate_scene31_magic_overnight.py",
    "scripts/generate_scene31_next_round.py",
    "scripts/summarize_scene31_bc_next.py",
    "scripts/summarize_scene31_beamsoft_weak.py",
    "scripts/summarize_scene31_funnel.py",
    "scripts/summarize_scene31_next_round.py",
    "scripts/summarize_scene31_p0_fresh_eval.py",
    "scripts/select_missing_aware_checkpoint.py",
}


def test_pyproject_console_scripts_point_to_existing_functions():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts

    for command, target in scripts.items():
        module_name, function_name = target.split(":", 1)
        module_path = SRC.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.exists(), f"{command} points to missing module {module_name}"
        names = _top_level_names(module_path)
        assert function_name in names, f"{command} points to missing function {function_name}"


def test_cli_modules_are_console_scripts_or_shared_helpers():
    scripts = _pyproject()["project"]["scripts"]
    console_modules = {target.split(":", 1)[0] for target in scripts.values()}
    shared_helpers = {"kd_sensing.cli.common"}
    violations: list[str] = []

    for path in sorted((SRC / "kd_sensing/cli").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module_name = ".".join(path.with_suffix("").relative_to(SRC).parts)
        names = _top_level_names(path)
        has_runnable = "main" in names or "console_main" in names or "build_parser" in names
        if has_runnable and module_name not in console_modules and module_name not in shared_helpers:
            violations.append(f"{module_name} ({_rel(path)})")

    assert not violations


def test_current_paths_and_config_globs_are_real():
    for rel_path in CURRENT_PATHS:
        assert (ROOT / rel_path).exists(), rel_path
    for pattern in CURRENT_CONFIG_GLOBS:
        matches = sorted(ROOT.glob(pattern))
        assert matches, pattern
        assert all(path.is_file() for path in matches)


def test_deleted_facades_and_one_shot_script_do_not_return():
    for rel_path in DELETED_SURFACE_PATHS:
        assert not (ROOT / rel_path).exists(), rel_path


def test_runtime_sources_do_not_use_future_annotations_or_star_imports():
    violations: list[str] = []
    for root in (SRC, ROOT / "scripts", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            rel = _rel(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    if any(alias.name == "annotations" for alias in node.names):
                        violations.append(f"{rel}:{node.lineno} future annotations import")
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                    violations.append(f"{rel}:{node.lineno} runtime star import")

    assert not violations


def test_deleted_current_references_do_not_return():
    stale = (
        "kd_sensing.cli.beambench_check_dataset",
        "configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml",
    )
    allowed_context = ("已删除", "不要求", "退役", "历史", "deleted", "retired", "historical")
    current_paths = [
        ROOT / "README.md",
        ROOT / "configs/baselines/beambench_reproduction.yaml",
        ROOT / "docs/model_architecture_inventory.md",
        ROOT / "docs/extension_guide.md",
        *sorted((ROOT / "openspec/specs").glob("*/spec.md")),
    ]
    violations: list[str] = []
    for path in current_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        text = "\n".join(lines)
        for fragment in stale:
            if fragment not in text:
                continue
            for index, line in enumerate(lines):
                if fragment not in line:
                    continue
                window = "\n".join(lines[max(0, index - 2) : index + 3]).lower()
                if not any(marker.lower() in window for marker in allowed_context):
                    violations.append(f"{_rel(path)}:{index + 1}: {fragment}")
    assert not violations


def test_internal_code_uses_owner_modules_not_retired_facades():
    violations: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_IMPORTS:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_baseline_workflows_do_not_register_model_components():
    registry_fragments = (
        "@MODELS.register",
        "@ENCODERS.register",
        "@PROJECTORS.register",
        "@REPRESENTATION_CORES.register",
        "@HEADS.register",
    )
    violations: list[str] = []
    for path in sorted((SRC / "kd_sensing/baselines").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in registry_fragments:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_model_owners_do_not_depend_on_baseline_workflows():
    forbidden_fragments = (
        "from kd_sensing.baselines",
        "import kd_sensing.baselines",
        "from ..baselines",
        "from .baselines",
    )
    violations: list[str] = []
    for path in sorted((SRC / "kd_sensing/models").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_lightweight_package_markers_do_not_grow_eager_barrel_exports():
    package_markers = (
        "src/kd_sensing/data/__init__.py",
        "src/kd_sensing/data/mmw/__init__.py",
        "src/kd_sensing/data/transform_ops/__init__.py",
        "src/kd_sensing/diagnostics/__init__.py",
        "src/kd_sensing/baselines/beambench/__init__.py",
        "src/kd_sensing/baselines/rmbp_mm/__init__.py",
        "src/kd_sensing/engine/__init__.py",
        "src/kd_sensing/evaluation/__init__.py",
        "src/kd_sensing/losses/__init__.py",
        "src/kd_sensing/models/__init__.py",
        "src/kd_sensing/models/physics/__init__.py",
        "src/kd_sensing/preprocessing/__init__.py",
        "src/kd_sensing/utils/__init__.py",
    )
    forbidden_fragments = (
        "from kd_sensing.",
        "import kd_sensing.",
        "from .datasets",
        "from .jepa_",
        "from .image_ae_gps",
        "from .fusion",
        "from .normalization",
    )
    violations: list[str] = []
    for rel_path in package_markers:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{rel_path}: {fragment}")
    assert not violations


def test_legacy_registry_fixture_stays_small():
    text = (ROOT / "tests/fixtures/legacy_model_registry_retirement.yaml").read_text(encoding="utf-8")
    assert text.count("\n  - name:") <= 16
    assert "migration_target:" not in text
    assert "error_hint:" not in text


def test_retired_route_mentions_are_contextualized():
    docs = [
        ROOT / "README.md",
        ROOT / "docs/project_surface_inventory.md",
        ROOT / "docs/research_notes.md",
        ROOT / "docs/agent_navigation.md",
    ]
    violations: list[str] = []
    for path in docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not any(marker in line for marker in RETIRED_TEXT_MARKERS):
                continue
            window = "\n".join(lines[max(0, index - 2) : index + 3]).lower()
            if not any(marker.lower() in window for marker in RETIREMENT_CONTEXT):
                violations.append(f"{_rel(path)}:{index + 1}: {line.strip()}")
    assert not violations


def test_tracked_runtime_artifacts_are_not_in_source_control():
    tracked = set(_git_ls_files())
    forbidden_prefixes = ("outputs/", "logs/", "cache/", "outputs/cache/", ".pytest_cache/")
    codegraph_violations = [path for path in tracked if path.startswith(".codegraph/") and path != ".codegraph/.gitignore"]
    violations = [
        path
        for path in tracked
        if path.startswith(forbidden_prefixes) or "__pycache__/" in path or path.endswith(".pyc")
    ]
    dataset_violations = [path for path in tracked if path.startswith("dataset/") and path != "dataset/.gitkeep"]
    all_models_violations = [path for path in tracked if path.startswith("All_models/")]

    assert not codegraph_violations
    assert not violations
    assert not dataset_violations
    assert not all_models_violations


def test_ponytail_followup_artifacts_are_not_tracked():
    tracked = {path for path in _git_ls_files() if (ROOT / path).exists()}
    forbidden = {
        "legacy_knowledge_decoupling_cleanup_manifest.json",
        "scripts/run_priority_v3_budget.sh",
    }
    metadata = [path for path in tracked if path.startswith("src/") and ".egg-info/" in path]

    assert forbidden.isdisjoint(tracked)
    assert not metadata


def test_current_openspec_specs_have_real_purpose():
    violations: list[str] = []
    placeholders = ("tbd", "created by archiving", "update purpose", "todo")
    for path in sorted((ROOT / "openspec/specs").glob("*/spec.md")):
        purpose = _purpose_section(path)
        lowered = purpose.lower()
        if not purpose or len(purpose) < 10 or any(marker in lowered for marker in placeholders):
            violations.append(_rel(path))
    assert not violations


def test_openspec_specs_match_lifecycle_inventory():
    spec_root = ROOT / "openspec/specs"
    spec_dirs = {path.parent.name for path in spec_root.glob("*/spec.md")}
    all_dirs = {path.name for path in spec_root.iterdir() if path.is_dir()}
    rows = re.findall(
        r"^\| `([^`]+)` \| `(current|supporting|retired-tombstone)` \|",
        INVENTORY.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    capabilities = [capability for capability, _lifecycle in rows]

    assert all_dirs == spec_dirs
    assert len(capabilities) == len(set(capabilities))
    assert set(capabilities) == spec_dirs


def test_current_validation_commands_reference_existing_openspec_targets():
    active_changes = _active_openspec_change_names()
    current_specs = {path.parent.name for path in (ROOT / "openspec/specs").glob("*/spec.md")}
    valid_targets = active_changes | current_specs
    command_pattern = re.compile(r"openspec validate\s+(?!--all\b)(?P<target>[A-Za-z0-9_.<>{}-]+)\s+--strict")
    violations: list[str] = []

    for path in _current_validation_command_paths():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in command_pattern.finditer(line):
                target = match.group("target")
                if "<" in target or ">" in target:
                    continue
                if target not in valid_targets:
                    violations.append(f"{_rel(path)}:{line_number}: {match.group(0)}")

    index_text = (ROOT / "docs/maintainer_context_index.yaml").read_text(encoding="utf-8")
    assert "openspec validate --all --strict" in index_text
    assert not violations, "Use openspec validate --all --strict or a current spec/active change target."


def test_project_surface_inventory_sizing_baseline_declares_scan_method():
    section = _inventory_section("## 项目健康护栏基线", "当前 AST 热点清单如下")
    required_markers = {
        "统计口径": ("on-disk", "tracked-only", "扫描口径"),
        "扫描范围": ("src/kd_sensing", "tests/", "scripts/", "configs/"),
        "排除项": ("dataset/", "outputs/", "logs/", "cache", "checkpoint"),
        "非硬 KPI": ("趋势信号", "非硬 KPI"),
    }
    missing: list[str] = []
    for label, markers in required_markers.items():
        if not all(marker in section for marker in markers):
            missing.append(label)
    assert not missing, f"Inventory sizing baseline is missing: {', '.join(missing)}"


def test_fusion_root_yaml_matches_inventory_classification():
    actual = sorted(path.name for path in (ROOT / "configs/fusion").glob("*.yaml"))
    inventory = _inventory_section("`configs/fusion/` 根目录保留分类如下：", "已迁移到")
    listed = sorted(set(re.findall(r"`([^`]+\.yaml)`", inventory)))
    assert actual == listed


def test_scripts_are_classified_in_inventory():
    scripts = sorted(
        path
        for path in _git_ls_files()
        if path.startswith("scripts/")
        and Path(path).suffix in {".py", ".sh"}
        and (ROOT / path).exists()
        and "__pycache__" not in Path(path).parts
    )
    inventory = INVENTORY.read_text(encoding="utf-8")
    missing = [script for script in scripts if f"`{script}`" not in inventory]
    assert not missing


def test_scene31_generated_yaml_is_not_tracked_surface():
    existing = [
        _rel(path)
        for root in (ROOT / rel_path for rel_path in SCENE31_MANIFEST_BACKED_CONFIG_ROOTS)
        for path in sorted(root.glob("*.yaml"))
    ]
    assert not existing


def test_scene31_retained_yaml_surface_is_explicitly_registered():
    actual = {
        _rel(path)
        for path in (ROOT / "configs/scene31").rglob("*.yaml")
        if path.is_file()
    }
    assert actual == SCENE31_RETAINED_YAML
    inventory = INVENTORY.read_text(encoding="utf-8")
    for rel_path in sorted(SCENE31_RETAINED_YAML):
        if rel_path.startswith("configs/scene31/templates/"):
            continue
        assert rel_path.split("/")[-1].split("_", 1)[0] in inventory or rel_path in inventory


def test_scene31_local_manual_scripts_are_explicitly_registered():
    tracked = {path for path in _git_ls_files() if (ROOT / path).exists()}
    actual_runners = {path for path in tracked if path.startswith("scripts/run_scene31_") and path.endswith(".sh")}
    actual_tools = {
        path
        for path in tracked
        if (
            path.startswith("scripts/generate_scene31_")
            or path.startswith("scripts/summarize_scene31_")
            or path == "scripts/select_missing_aware_checkpoint.py"
        )
    }
    inventory = INVENTORY.read_text(encoding="utf-8")
    assert actual_runners == SCENE31_LOCAL_MANUAL_RUNNERS
    assert actual_tools == SCENE31_GENERATORS_AND_SUMMARIES
    for rel_path in sorted(actual_runners | actual_tools):
        assert f"`{rel_path}`" in inventory


def test_root_temp_runbooks_do_not_return():
    assert not (ROOT / "test.md").exists()


def test_internal_sources_do_not_import_public_facade_helpers():
    restricted_dirs = ("diagnostics", "engine", "data", "models", "losses", "evaluation")
    allowed = {
        ROOT / "src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py",
        ROOT / "src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py",
    }
    fragments = (
        "from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import",
        "import kd_sensing.diagnostics.jepa_gps_shortcut_benchmark",
    )
    violations: list[str] = []
    for package in restricted_dirs:
        for path in (ROOT / "src/kd_sensing" / package).rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in fragments:
                if fragment in text:
                    violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_plain_pytest_files_do_not_insert_tests_path_bootstrap():
    violations: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line_number in _module_level_sys_path_insert_lines(tree):
            violations.append(f"{_rel(path)}:{line_number}")
    assert not violations, "Ordinary tests should use tests.<helper> imports or shared pytest bootstrap."


def test_deleted_active_openspec_changes_have_matching_archive_status():
    status_lines = _git_status_short()
    deleted_active_changes: set[str] = set()
    archive_changes: set[str] = set()

    for line in status_lines:
        status = line[:2]
        path = line[3:]
        if path.startswith("openspec/changes/archive/"):
            match = re.match(r"openspec/changes/archive/\d{4}-\d{2}-\d{2}-(?P<change>[^/]+)/", path)
            if match:
                archive_changes.add(match.group("change"))
            continue
        if not path.startswith("openspec/changes/") or "D" not in status:
            continue
        parts = path.split("/")
        if len(parts) >= 3:
            deleted_active_changes.add(parts[2])

    missing_archives = sorted(change for change in deleted_active_changes if change not in archive_changes)
    assert not missing_archives, (
        "Deleted OpenSpec active change directories must be paired with "
        f"openspec/changes/archive/<date>-<change>/ status entries: {missing_archives}"
    )


def test_retired_console_scripts_are_absent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    retired = (
        "kd-sensing-hist-beam-loso",
        "kd-sensing-run-amr-net-gps-image",
        "kd-sensing-run-jepa-msac",
        "kd-sensing-export-viewer-manifest",
        "kd-sensing-visualize-modalities",
    )
    assert all(command not in pyproject for command in retired)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _python_sources() -> list[Path]:
    roots = (SRC, ROOT / "tests", ROOT / "scripts")
    ignored = {Path(__file__).resolve()}
    return [
        ROOT / rel_path
        for rel_path in _git_ls_files()
        if rel_path.endswith(".py")
        and (ROOT / rel_path).exists()
        and any((ROOT / rel_path).is_relative_to(root) for root in roots)
        and (ROOT / rel_path).resolve() not in ignored
    ]


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def _active_openspec_change_names() -> set[str]:
    changes_root = ROOT / "openspec/changes"
    return {path.name for path in changes_root.iterdir() if path.is_dir() and path.name != "archive"}


def _current_validation_command_paths() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "docs/agent_navigation.md",
        ROOT / "docs/maintainer_context_index.yaml",
        ROOT / "docs/project_surface_inventory.md",
        *sorted((ROOT / "openspec/specs").glob("*/spec.md")),
    ]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _purpose_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Purpose\s*(?P<body>.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    return "" if match is None else match.group("body").strip()


def _inventory_section(start: str, end: str) -> str:
    text = INVENTORY.read_text(encoding="utf-8")
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def _is_sys_path_insert_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "insert"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def _module_level_sys_path_insert_lines(tree: ast.Module) -> list[int]:
    line_numbers: list[int] = []
    for node in tree.body:
        candidates = [node]
        if isinstance(node, ast.If):
            candidates.extend(node.body)
            candidates.extend(node.orelse)
        for candidate in candidates:
            if _is_sys_path_insert_expr(candidate):
                line_numbers.append(candidate.lineno)
    return line_numbers
