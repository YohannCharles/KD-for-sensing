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
    "scripts/analyze_csi_hardening_sweep.py": "research_diagnostic",
    "scripts/analysis/beambench_ae_gps_diagnostics.py": "research_diagnostic",
    "scripts/analysis/visualize_deepsense_beambench_correspondence.py": "research_diagnostic",
    "scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py": "research_diagnostic",
    "scripts/debug_eval_consistency.py": "research_diagnostic",
    "scripts/check_dataset.py": "dataset_preparation",
    "scripts/eval_baseline.py": "thin_cli_alias",
    "scripts/evaluate.py": "thin_cli_alias",
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
    "tools/visualization/gradio_multimodal_viewer.py": "viewer_entrypoint",
    "tools/visualization/viewer_constants.py": "viewer_support",
    "tools/visualization/viewer_figures.py": "viewer_support",
    "tools/visualization/viewer_manifest_io.py": "viewer_support",
    "tools/visualization/viewer_prediction_tables.py": "viewer_support",
    "tools/visualization/viewer_utils.py": "viewer_support",
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
    "viewer_entrypoint",
    "viewer_support",
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
    fusion_root_entries = {path.relative_to(ROOT).as_posix() for path in fusion_yaml}
    tracked = set(_tracked_paths())
    script_entries = {
        path
        for path in tracked
        if path.endswith(".py")
        and path.startswith(("scripts/", "tools/analysis/", "tools/visualization/"))
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


def test_gps_lidar_bgam_workflow_stays_inside_package_boundaries():
    forbidden_root_entries = [
        ROOT / "train_gps_lidar_bgam.py",
        ROOT / "eval_gps_lidar_bgam.py",
        ROOT / "datasets" / "gps_lidar_dataset.py",
        ROOT / "models" / "gps_lidar_bgam.py",
    ]
    for path in forbidden_root_entries:
        assert not path.exists()

    forbidden_imports = []
    bgam_paths = [
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_dataset.py",
        ROOT / "src" / "kd_sensing" / "data" / "deepsense6g_gps_lidar_bgam_manifest.py",
        ROOT / "src" / "kd_sensing" / "engine" / "deepsense6g_gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam.py",
        ROOT / "src" / "kd_sensing" / "models" / "gps_lidar_bgam_model.py",
    ]
    for path in bgam_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("datasets", "models", "src.run_")):
                    forbidden_imports.append(f"{path.relative_to(ROOT)} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("datasets", "models", "src.run_")):
                        forbidden_imports.append(f"{path.relative_to(ROOT)} imports {alias.name}")

    assert forbidden_imports == []


def test_shell_orchestration_defaults_do_not_write_to_outputs_other():
    violations = []
    forbidden = 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/other'
    for rel_path in sorted(SHELL_ORCHESTRATION_ALLOWLIST):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        if forbidden in text:
            violations.append(rel_path)

    assert violations == []


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
        "src/kd_sensing/data/deepverse/label_builder.py",
        "src/kd_sensing/data/deepverse/label_writers.py",
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
        "tools/visualization/viewer_utils.py": {
            "max_lines": 140,
            "forbidden": [
                "def load_manifest",
                "def filter_samples",
                "def make_beam_confidence_figure",
                "def _legacy_prediction_summary_row",
            ],
            "helpers": {
                "tools/visualization/viewer_manifest_io.py": "def load_manifest",
                "tools/visualization/viewer_figures.py": "def make_future_distribution_plot",
                "tools/visualization/viewer_prediction_tables.py": "def _legacy_prediction_summary_row",
            },
        },
        "src/kd_sensing/preprocessing/raymobtime_s008.py": {
            "max_lines": 100,
            "forbidden": [
                "def _assign_splits",
                "def _load_ray_table",
                "def normalize_beam_labels",
                "def _cache_metadata",
            ],
            "helpers": {
                "src/kd_sensing/preprocessing/raymobtime_s008_index.py": "def _assign_splits",
                "src/kd_sensing/preprocessing/raymobtime_s008_beam_labels.py": "def normalize_beam_labels",
                "src/kd_sensing/preprocessing/raymobtime_s008_ray_features.py": "def _load_ray_table",
                "src/kd_sensing/preprocessing/raymobtime_s008_cache.py": "def _cache_metadata",
            },
        },
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
        "src/kd_sensing/data/deepverse/label_builder.py": {
            "max_lines": 650,
            "forbidden": [
                "def write_label_cache",
                "def _blockage_metadata",
                "def _get_sample",
                "def _write_json",
            ],
            "helpers": {
                "src/kd_sensing/data/deepverse/label_scene.py": "class MobilityTrace",
                "src/kd_sensing/data/deepverse/label_targets.py": "def _blockage_metadata",
                "src/kd_sensing/data/deepverse/label_writers.py": "def write_label_cache",
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


def test_deepsense6g_residual_cli_imports_are_package_scoped_and_light():
    modules = _run_module_presence_probe(
        "import kd_sensing.cli.inspect_deepsense6g_residual_inputs\n"
        "import kd_sensing.cli.prepare_deepsense6g_residual_manifest\n"
        "import kd_sensing.cli.run_deepsense6g_residual_fusion\n"
        "import kd_sensing.cli.plot_deepsense6g_residual_fusion\n"
        "import kd_sensing.cli.compare_deepsense6g_residual_with_gps_v2",
        {
            "top_level_residual": "src.inspect_deepsense6g_residual_inputs",
            "trainer": "kd_sensing.engine.trainer",
            "data_factory": "kd_sensing.engine.data_factory",
        },
    )

    assert modules == {
        "top_level_residual": False,
        "trainer": False,
        "data_factory": False,
    }


def test_deepsense6g_top8_selector_cli_imports_are_package_scoped():
    modules = _run_module_presence_probe(
        "import kd_sensing.cli.prepare_deepsense6g_top8_candidate_manifest\n"
        "import kd_sensing.cli.run_deepsense6g_top8_selector\n"
        "import kd_sensing.cli.plot_deepsense6g_top8_selector\n"
        "import kd_sensing.cli.compare_deepsense6g_top8_selector_with_gps_v2",
        {
            "top_level_run": "src.run_deepsense6g_top8_selector",
            "top_level_data": "src.data.deepsense6g_topk_candidate_manifest",
            "top_level_models": "src.models.topk_candidate_selector",
            "top_level_losses": "src.losses.topk_candidate_losses",
            "trainer": "kd_sensing.engine.trainer",
            "data_factory": "kd_sensing.engine.data_factory",
        },
    )

    assert modules == {
        "top_level_run": False,
        "top_level_data": False,
        "top_level_models": False,
        "top_level_losses": False,
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
        _dotted("kd_sensing", "data", "datasets", "multimodal_nf"),
        _dotted("kd_sensing", "diagnostics", "g2d_diagnostics"),
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
