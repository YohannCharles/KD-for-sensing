from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kd_sensing.config import load_config
from kd_sensing.engine.objectives.metadata import objective_spec, resolve_prediction_objective
from kd_sensing.registries import MODELS, RegistryError, import_default_components


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "rel_path",
    [
        "configs/pretraining/jepa_msac_s32_smoke.yaml",
        "configs/pretraining/jepa_msac_s32_paper.yaml",
    ],
)
def test_jepa_msac_config_paths_are_retired(rel_path: str):
    config_path = ROOT / rel_path

    assert not config_path.exists()
    with pytest.raises(ValueError, match="JEPA-MSAC.*retired"):
        load_config(config_path)


def test_jepa_msac_console_script_is_not_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "kd-sensing-run-jepa-msac" not in pyproject
    assert "kd_sensing.cli.run_jepa_msac" not in pyproject


@pytest.mark.parametrize(
    "module_name",
    [
        "kd_sensing.cli.run_jepa_msac",
        "kd_sensing.baselines.jepa_msac",
        "kd_sensing.models.jepa_msac",
        "kd_sensing.losses.jepa_msac",
    ],
)
def test_jepa_msac_module_paths_are_not_current(module_name: str):
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    "override",
    [
        "experiment.objective=jepa_msac_pretraining",
        "model.primary.type=jepa_msac",
        "workflow.jepa_msac.enabled=true",
        "loss.jepa_msac.weight=1.0",
    ],
)
def test_jepa_msac_config_content_fails_fast(override: str):
    with pytest.raises(ValueError, match="JEPA-MSAC|priority legacy workflow"):
        load_config(overrides=[override])


def test_jepa_msac_objective_and_model_registry_are_retired():
    with pytest.raises(ValueError, match="experiment.objective must be one of"):
        resolve_prediction_objective({"experiment": {"objective": "jepa_msac_pretraining"}})
    with pytest.raises(ValueError, match="Unknown prediction objective"):
        objective_spec("jepa_msac_pretraining")

    import_default_components()
    assert "jepa_msac" not in MODELS.list()
    with pytest.raises(RegistryError, match="jepa_msac"):
        MODELS.build({"type": "jepa_msac"})
