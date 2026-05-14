from __future__ import annotations

import ast
import json
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


def test_modality_contract_normalizes_and_validates_modalities():
    assert MODALITY_ORDER == ("image", "radar", "gps", "lidar", "mmwave")
    assert normalize_modalities(["lidar", "image", "gps"]) == ("image", "gps", "lidar")

    with pytest.raises(ValueError, match="thermal"):
        normalize_modalities(["image", "thermal"])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_modalities(["gps", "gps"])
    with pytest.raises(ValueError, match="at least one"):
        normalize_modalities([])


def test_modality_contract_derives_dataset_flags_and_batch_keys():
    assert dataset_flags_for_modalities(["mmwave", "gps"]) == {
        "use_gps": True,
        "use_lidar": False,
        "use_mmwave": True,
    }
    assert batch_input_keys_for_modalities(["radar", "mmwave"]) == {
        "radar": "radar_batch",
        "mmwave": "mmwave_batch",
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
    assert "extension.after_forward" in trainer_text
    assert "extension.after_epoch" in trainer_text

    assert "class G2DTrainingExtension" in (SRC / "kd_sensing" / "engine" / "g2d_training.py").read_text(encoding="utf-8")
    assert "class CrafTrainingExtension" in (SRC / "kd_sensing" / "engine" / "craf_training.py").read_text(encoding="utf-8")
    assert "class MarfTrainingExtension" in (SRC / "kd_sensing" / "engine" / "marf_training.py").read_text(encoding="utf-8")


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
