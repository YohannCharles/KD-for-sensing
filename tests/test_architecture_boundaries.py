from __future__ import annotations

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


def _run_import_probe(statement: str) -> dict[str, bool]:
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
{statement}
modules = {{
    "scenario9": "kd_sensing.data.datasets.scenario9" in sys.modules,
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
        "scenario9": False,
        "synthetic": False,
    }


def test_engine_light_submodule_does_not_import_heavy_boundaries():
    modules = _run_module_presence_probe(
        "import kd_sensing.engine.model_output",
        {
            "builders_impl": "kd_sensing.engine._builders_impl",
            "legacy_transforms": "kd_sensing.data.transform_ops._legacy",
            "pandas": "pandas",
            "scipy": "scipy",
        },
    )

    assert modules == {
        "builders_impl": False,
        "legacy_transforms": False,
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
            "builders": "kd_sensing.engine.builders",
            "builders_impl": "kd_sensing.engine._builders_impl",
            "distillers": "kd_sensing.distillation.distillers",
            "legacy_transforms": "kd_sensing.data.transform_ops._legacy",
        },
    )

    assert modules == {
        "builders": False,
        "builders_impl": False,
        "distillers": False,
        "legacy_transforms": False,
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


def test_legacy_transform_imports_remain_available():
    from kd_sensing.data.transforms import GPSStandardScaler, build_image_transform, read_lidar_point_cloud

    assert GPSStandardScaler is not None
    assert build_image_transform is not None
    assert read_lidar_point_cloud is not None


def test_transform_ops_legacy_facade_remains_available():
    from kd_sensing.data.transform_ops import _legacy
    from kd_sensing.data.transform_ops.gps import GPSStandardScaler as DirectGPSStandardScaler
    from kd_sensing.data.transform_ops.image import build_image_transform as direct_build_image_transform
    from kd_sensing.data.transform_ops.lidar import read_lidar_point_cloud as direct_read_lidar_point_cloud

    assert _legacy.GPSStandardScaler is DirectGPSStandardScaler
    assert _legacy.build_image_transform is direct_build_image_transform
    assert _legacy.read_lidar_point_cloud is direct_read_lidar_point_cloud
