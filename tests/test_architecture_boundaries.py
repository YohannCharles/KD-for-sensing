from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.modalities import (  # noqa: E402
    MODALITY_ORDER,
    batch_input_keys_for_modalities,
    dataset_flags_for_modalities,
    normalize_modalities,
)

PYTHON_ENTRYPOINT_ALLOWLIST = {
    "scripts/analysis/build_complementarity_cases.py": "research_diagnostic",
    "scripts/analyze_csi_hardening_sweep.py": "research_diagnostic",
    "scripts/build_teacher_registry.py": "research_diagnostic",
    "scripts/debug_eval_consistency.py": "research_diagnostic",
    "scripts/deepverse/download_dt31_assets.py": "dataset_preparation",
    "scripts/deepverse/generate_dt31_cache.py": "dataset_preparation",
    "scripts/eval_modality_perturbation.py": "research_diagnostic",
    "scripts/eval_modality_subsets.py": "research_diagnostic",
    "scripts/evaluate.py": "thin_cli_alias",
    "scripts/mmw/prepare_town10_skybridge.py": "dataset_preparation",
    "scripts/preprocess.py": "thin_cli_alias",
    "scripts/profile_training_io.py": "research_diagnostic",
    "scripts/recommend_parallel_training.py": "research_diagnostic",
    "scripts/train.py": "thin_cli_alias",
    "tools/analysis/analyze_conditional_utility.py": "research_diagnostic",
    "tools/analysis/collect_multimodal_imbalance_results.py": "research_diagnostic",
    "tools/analysis/run_conditional_utility_audit.py": "research_diagnostic",
    "tools/analysis/run_phase_1_5_utility_validation.py": "research_diagnostic",
    "tools/visualization/complementarity_explorer.py": "viewer_support",
    "tools/visualization/gradio_multimodal_viewer.py": "viewer_entrypoint",
    "tools/visualization/viewer_utils.py": "viewer_support",
}
SHELL_ORCHESTRATION_ALLOWLIST = {
    "scripts/run_csi_hardening_matrix.sh": "shell_orchestration",
}
RETIRED_GENERATED_FUSION_CONFIGS = {
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml",
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml",
    "configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml",
}


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


def test_project_surface_inventory_guardrails_are_current():
    fusion_yaml = sorted((ROOT / "configs" / "fusion").glob("*.yaml"))
    script_entries = {
        path.relative_to(ROOT).as_posix()
        for root in [ROOT / "scripts", ROOT / "tools" / "analysis", ROOT / "tools" / "visualization"]
        for path in root.rglob("*.py")
    }
    shell_entries = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").rglob("*.sh")
    }

    assert len(fusion_yaml) <= 27
    assert RETIRED_GENERATED_FUSION_CONFIGS.isdisjoint(
        {path.relative_to(ROOT).as_posix() for path in fusion_yaml}
    )
    assert script_entries == set(PYTHON_ENTRYPOINT_ALLOWLIST)
    assert shell_entries == set(SHELL_ORCHESTRATION_ALLOWLIST)


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
        if not purpose or len(purpose) < 50 or purpose == "TBD - created by archiving":
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


def test_diagnostics_light_submodule_does_not_import_visualization_stack():
    modules = _run_module_presence_probe(
        "import kd_sensing.diagnostics.g2d_diagnostics",
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


@pytest.mark.parametrize(
    "statement",
    [
        "import kd_sensing.diagnostics.visualization.config",
        "import kd_sensing.diagnostics.visualization.sampling",
        "import kd_sensing.diagnostics.visualization.writers",
    ],
)
def test_visualization_light_helpers_do_not_import_render_or_dataset_stack(statement: str):
    modules = _run_module_presence_probe(
        statement,
        {
            "torch": "torch",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "pil_image": "PIL.Image",
            "data_factory": "kd_sensing.engine.data_factory",
            "model_builder": "kd_sensing.engine.optim",
            "visualization_core": "kd_sensing.diagnostics.visualization.core",
            "visualization_datasets": "kd_sensing.diagnostics.visualization.datasets",
            "visualization_render": "kd_sensing.diagnostics.visualization.render",
        },
    )

    assert modules == {key: False for key in modules}


def test_distillation_tool_submodule_does_not_import_training_registry_or_transforms():
    modules = _run_module_presence_probe(
        "import kd_sensing.distillation.g2d_smp",
        {
            "builder_facade": _dotted("kd_sensing", "engine", "builders"),
            "builder_aggregate": _dotted("kd_sensing", "engine", "_builders_impl"),
            "distillers": "kd_sensing.distillation.distillers",
            "transform_aggregate": _dotted("kd_sensing", "data", "transform_ops", "_legacy"),
        },
    )

    assert modules == {
        "builder_facade": False,
        "builder_aggregate": False,
        "distillers": False,
        "transform_aggregate": False,
    }


def test_lazy_package_exports_remain_available():
    _run_module_presence_probe(
        "\n".join(
            [
                "from kd_sensing.engine import train",
                "from kd_sensing.diagnostics import export_viewer_manifest",
                "from kd_sensing.distillation import FocalLoss",
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
        },
    )

    assert modules == {key: False for key in modules}

    _run_module_presence_probe(
        "\n".join(
            [
                "from kd_sensing.models import FusionTeacherModalityNet, GpsModalityNet",
                "assert FusionTeacherModalityNet is not None",
                "assert GpsModalityNet is not None",
                "import kd_sensing.models as models",
                "assert 'FusionTeacherModalityNet' in models.__all__",
                "try:",
                "    getattr(models, 'Fusion' + 'ModalityNet')",
                "except AttributeError as exc:",
                "    assert 'FusionTeacherModalityNet' in str(exc)",
                "else:",
                "    raise AssertionError('removed alias did not raise')",
            ]
        ),
        {},
    )


def test_modality_contract_normalizes_and_validates_modalities():
    assert MODALITY_ORDER[:6] == ("image", "radar", "gps", "lidar", "mmwave", "csi")
    assert MODALITY_ORDER[-2:] == ("coord", "ray")
    assert normalize_modalities(["csi", "lidar", "image", "gps"]) == ("image", "gps", "lidar", "csi")
    assert normalize_modalities(["ray", "coord", "lidar", "image"]) == ("image", "lidar", "coord", "ray")

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
        "use_coord": False,
        "use_ray": False,
    }
    assert batch_input_keys_for_modalities(["radar", "mmwave", "csi", "coord", "ray"]) == {
        "radar": "radar_batch",
        "mmwave": "mmwave_batch",
        "csi": "csi_batch",
        "coord": "coord_batch",
        "ray": "ray_batch",
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
    assert "G2DTrainingExtension" in trainer_text
    assert "CrafTrainingExtension" in trainer_text
    assert "MarfTrainingExtension" in trainer_text
    assert "BatchStepRunner" in trainer_text
    assert "extension.after_epoch" in trainer_text

    assert "class G2DTrainingExtension" in (SRC / "kd_sensing" / "engine" / "g2d_training.py").read_text(encoding="utf-8")
    assert "class CrafTrainingExtension" in (SRC / "kd_sensing" / "engine" / "craf_training.py").read_text(encoding="utf-8")
    assert "class MarfTrainingExtension" in (SRC / "kd_sensing" / "engine" / "marf_training.py").read_text(encoding="utf-8")
    assert "extension.after_forward" in (SRC / "kd_sensing" / "engine" / "batch_step.py").read_text(encoding="utf-8")


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


def test_config_io_pipeline_delegates_business_rules():
    io_text = (SRC / "kd_sensing" / "config" / "io.py").read_text(encoding="utf-8")
    helper_expectations = {
        "source.py": "def load_config_source",
        "normalization.py": "def normalize_loaded_config",
        "validation.py": "def validate_loaded_config",
        "migration_guards.py": "def reject_removed_image_path_config",
        "dataset_rules/raymobtime.py": "def validate_raymobtime_config",
    }

    assert "load_config_source" in io_text
    assert "normalize_loaded_config" in io_text
    assert "validate_loaded_config" in io_text
    assert "reject_removed_image_path_config" in io_text
    assert "build_virtual_config" not in io_text
    assert "future_beam" not in io_text
    assert "REMOVED_IMAGE_ENCODERS" not in io_text
    assert "snapshot_next_frame requires" not in io_text
    assert "configure_objective_defaults" not in io_text
    for module, snippet in helper_expectations.items():
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


def test_g2d_algorithm_module_does_not_import_runtime_builders_or_teacher_runtime():
    path = SRC / "kd_sensing" / "distillation" / "g2d.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden = {
        "kd_sensing.engine.optim",
        "kd_sensing.engine.batch",
        "kd_sensing.engine.g2d_training",
        "kd_sensing.distillation.teacher_ensemble",
        "kd_sensing.utils.artifact_registry",
        "kd_sensing.utils.checkpoint",
    }
    assert sorted(imported_modules & forbidden) == []


def test_visualization_core_is_thin_and_submodules_own_implementations():
    core_text = (SRC / "kd_sensing" / "diagnostics" / "visualization" / "core.py").read_text(encoding="utf-8")
    core_lines = core_text.splitlines()
    forbidden_in_core = (
        "class VisualizationConfig",
        "class SampleCandidate",
        "def build_diagnostic_datasets",
        "def collect_candidates",
        "def tensor_stats",
        "def render_sample_overview",
        "def write_samples_jsonl",
    )

    assert len(core_lines) < 300
    assert [snippet for snippet in forbidden_in_core if snippet in core_text] == []
    for module, snippet in {
        "config.py": "class VisualizationConfig",
        "datasets.py": "def build_diagnostic_datasets",
        "sampling.py": "def collect_candidates",
        "stats.py": "def tensor_stats",
        "render.py": "def render_sample_overview",
        "writers.py": "def write_samples_jsonl",
    }.items():
        text = (SRC / "kd_sensing" / "diagnostics" / "visualization" / module).read_text(encoding="utf-8")
        assert snippet in text


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
