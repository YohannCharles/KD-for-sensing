from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kd_sensing.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_amr_net_gps_image_config_path_is_retired():
    config_path = ROOT / "configs" / "baselines" / "amr_net_gps_image.yaml"

    assert not config_path.exists()
    with pytest.raises(ValueError, match="AMR-Net_gps_image.*retired"):
        load_config(config_path)


def test_amr_net_gps_image_console_script_is_not_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "kd-sensing-run-amr-net-gps-image" not in pyproject
    assert "kd_sensing.cli.run_amr_net_gps_image" not in pyproject


@pytest.mark.parametrize(
    "module_name",
    [
        "kd_sensing.cli.run_amr_net_gps_image",
        "kd_sensing.baselines.amr_net_gps_image",
    ],
)
def test_amr_net_gps_image_module_paths_are_not_current(module_name: str):
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    "override",
    [
        "experiment.baseline_preset=amr_net_gps_image",
        "experiment.paper=amr_net_gps_image",
        "model.primary.model_name=AMR-Net_gps_image",
        "model.primary.baseline_preset=amr_net_gps_image",
        "amr_net_gps_image.enabled=true",
    ],
)
def test_amr_net_gps_image_config_content_fails_fast(override: str):
    with pytest.raises(ValueError, match="AMR-Net_gps_image|priority legacy workflows"):
        load_config(overrides=[override])
