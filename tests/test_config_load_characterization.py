from pathlib import Path

import pytest

from kd_sensing.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "u0": ("U0", "u_mask_beam_jepa"),
    "amber_full": ("amber_full", "modular_sequence"),
    "rmbp_mm": ("rmbp_mm", "modular_sequence"),
}


@pytest.mark.parametrize(("recipe", "expected"), RECIPES.items())
def test_tracked_mmw_recipe_loads_without_runtime_input(recipe: str, expected: tuple[str, str]):
    path = ROOT / "configs/mmw" / f"{recipe}.yaml"
    cfg = load_config(path)
    name, model_type = expected

    assert path.exists()
    assert cfg["experiment"]["name"] == name
    assert cfg["data"]["dataset"]["type"] == "mmw"
    assert cfg["model"]["primary"]["type"] == model_type
    assert cfg["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar"]


def test_u0_recipe_contains_no_retired_training_sections():
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")

    assert cfg["loss"]["u_mask_beam_jepa"]["superset_consistency"]["enabled"] is True
    assert cfg["temporal_missing"]["preserve_unmasked_for_superset"] is True
    assert "bcacl" not in cfg
    assert "cmsbl" not in cfg


def test_tracked_deepsense6g_t2_recipe_loads_without_runtime_input():
    cfg = load_config(ROOT / "configs/deepsense6g/t2.yaml")

    assert cfg["experiment"]["name"] == "T2"
    assert cfg["data"]["dataset"]["type"] == "deepsense6g"
    assert cfg["data"]["dataset"]["scene"] == 31
    assert cfg["model"]["primary"]["type"] == "u_mask_beam_jepa"
    assert cfg["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar"]


def test_recipe_base_and_cli_overrides_keep_temporal_windows_in_sync():
    cfg = load_config(
        ROOT / "configs/mmw/u0.yaml",
        overrides=[
            "temporal_missing.history_window=3",
            "temporal_missing.prediction_window=2",
            "data.dataloader.train_batch_size=7",
        ],
    )

    assert cfg["data"]["dataset"]["seq_len"] == 3
    assert cfg["data"]["dataset"]["num_pred"] == 2
    assert cfg["model"]["primary"]["seq_length"] == 3
    assert cfg["model"]["primary"]["num_pred"] == 2
    assert cfg["data"]["dataloader"]["train_batch_size"] == 7
