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
    "scripts/run_csi_hardening_matrix.sh",
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


def test_pyproject_console_scripts_point_to_existing_functions():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts

    for command, target in scripts.items():
        module_name, function_name = target.split(":", 1)
        module_path = SRC.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.exists(), f"{command} points to missing module {module_name}"
        names = _top_level_names(module_path)
        assert function_name in names, f"{command} points to missing function {function_name}"


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


def test_internal_code_uses_owner_modules_not_retired_facades():
    violations: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_IMPORTS:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_lightweight_package_markers_do_not_grow_eager_barrel_exports():
    package_markers = (
        "src/kd_sensing/data/__init__.py",
        "src/kd_sensing/data/transform_ops/__init__.py",
        "src/kd_sensing/diagnostics/__init__.py",
        "src/kd_sensing/baselines/beambench/__init__.py",
        "src/kd_sensing/models/__init__.py",
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


def test_current_openspec_specs_have_real_purpose():
    violations: list[str] = []
    placeholders = ("tbd", "created by archiving", "update purpose", "todo")
    for path in sorted((ROOT / "openspec/specs").glob("*/spec.md")):
        purpose = _purpose_section(path)
        lowered = purpose.lower()
        if not purpose or len(purpose) < 10 or any(marker in lowered for marker in placeholders):
            violations.append(_rel(path))
    assert not violations


def test_fusion_root_yaml_matches_inventory_classification():
    actual = sorted(path.name for path in (ROOT / "configs/fusion").glob("*.yaml"))
    inventory = _inventory_section("`configs/fusion/` 根目录保留分类如下：", "已迁移到")
    listed = sorted(set(re.findall(r"`([^`]+\.yaml)`", inventory)))
    assert actual == listed


def test_scripts_are_classified_in_inventory():
    scripts = sorted(
        _rel(path)
        for path in (ROOT / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts
    )
    inventory = INVENTORY.read_text(encoding="utf-8")
    missing = [script for script in scripts if f"`{script}`" not in inventory]
    assert not missing


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


def test_plain_pytest_files_do_not_duplicate_src_bootstrap():
    violations: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if _is_sys_path_insert_expr(node):
                violations.append(_rel(path))
    assert not violations


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
