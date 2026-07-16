from pathlib import Path

import pytest

from kd_sensing.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "t2": ("T2", "u_mask_beam_jepa"),
    "s1": ("S1", "u_mask_beam_jepa"),
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


def test_t2_and_s1_only_differ_in_superset_consistency():
    t2 = load_config(ROOT / "configs/mmw/t2.yaml")
    s1 = load_config(ROOT / "configs/mmw/s1.yaml")

    assert t2["loss"]["u_mask_beam_jepa"]["superset_consistency"]["enabled"] is True
    assert s1["loss"]["u_mask_beam_jepa"]["superset_consistency"]["enabled"] is False
    assert t2["temporal_missing"]["preserve_unmasked_for_superset"] is True
    assert s1["temporal_missing"]["preserve_unmasked_for_superset"] is False


def test_recipe_base_and_cli_overrides_keep_temporal_windows_in_sync():
    cfg = load_config(
        ROOT / "configs/mmw/t2.yaml",
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
