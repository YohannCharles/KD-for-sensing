import hashlib
import json
from pathlib import Path

import pytest

from kd_sensing.config import load_config
from tools.run_topology_predictor import _bind_preprocessed_caches


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


def test_deepsense6g_secondary_transfer_recipes_load_without_runtime_input():
    prototype = load_config(ROOT / "tools/configs/deepsense6g/prototype_only.yaml")
    amber = load_config(ROOT / "tools/configs/deepsense6g/amber_full.yaml")
    rmbp = load_config(ROOT / "tools/configs/deepsense6g/rmbp_mm.yaml")

    assert prototype["model"]["primary"]["prototype_topology_id"] == "linear_index_v1"
    assert prototype["loss"]["four_modal_topology"]["prototype_topology"]["id"] == "linear_index_v1"
    assert prototype["training"]["checkpoint_selection"] == "last"
    assert amber["model"]["primary"]["representation_core"]["type"] == "amber_full_adaptive_mask_transformer"
    assert rmbp["model"]["primary"]["representation_core"]["type"] == "rmbp_channel_attention_fusion"


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


def test_topology_runtime_cache_binding_is_explicit_and_protocol_bound(tmp_path: Path):
    frame_root = tmp_path / "frames"
    gps_root = tmp_path / "gps"
    for relative in ("sunny/image_derived", "sunny/lidar_bev"):
        (frame_root / relative).mkdir(parents=True)
    gps_root.mkdir()
    cache_path = gps_root / "sunny__scene.npz"
    cache_path.write_bytes(b"coordinate-cache")
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    protocol = {
        "protocol_id": "mmw_id_stratified_block_v1",
        "protocol_version": 1,
        "split_manifest_hash": "a" * 64,
        "protocol_fingerprint": "b" * 64,
        "split_seed": 0,
        "block_size": 32,
        "data_source_hash": "c" * 64,
        "window_config_hash": "d" * 64,
        "weather_binding": True,
    }
    manifest = dict(protocol) | {
        "roles": ["train", "validation"],
        "strict_cache_coverage": True,
        "test_evaluated": False,
        "outer_test_accessed": False,
        "domains": [{"domain_id": "sunny/scene", "sha256": digest}],
    }
    manifest_path = gps_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "data": {"dataset": {"domains": [{"id": "sunny/scene"}]}},
        "data_protocol": protocol,
    }

    binding = _bind_preprocessed_caches(
        cfg,
        frame_cache_root=frame_root,
        gps_coordinate_cache_root=gps_root,
    )

    assert cfg["data"]["dataset"]["frame_cache_strict"] is True
    assert binding["gps_coordinate_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest["protocol_fingerprint"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cache provenance mismatch"):
        _bind_preprocessed_caches(
            cfg,
            frame_cache_root=frame_root,
            gps_coordinate_cache_root=gps_root,
        )


@pytest.mark.parametrize("name", ("radar_robust_on", "radar_robust_off"))
def test_radar_robust_templates_are_self_contained_and_use_whole_schedule(name: str):
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")

    temporal = cfg["temporal_missing"]
    assert cfg["output"]["dir"] == "outputs/four_modal_topology_predictor_radar_robust"
    assert temporal["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert temporal["panel_size"] == 480
    assert temporal["condition_counts"] == {
        "clean": 120,
        "drop1": 120,
        "drop2": 120,
        "drop3": 120,
        "token20": 0,
        "token40": 0,
        "token60": 0,
        "token80": 0,
        "token90": 0,
    }
    assert cfg["model"]["primary"]["prototype_topology_id"] == "cyclic_index_v1"
    loss = cfg["loss"]["four_modal_topology"]
    if name.endswith("_on"):
        assert (loss["unimodal_soft_weight"], loss["lambda_proto"], loss["lambda_modality_proto"]) == (0.1, 0.1, 0.0)
        assert loss["use_beam_prototype_alignment"] is True
    else:
        assert (loss["unimodal_soft_weight"], loss["lambda_proto"], loss["lambda_modality_proto"]) == (0.0, 0.0, 0.0)
        assert loss["use_beam_prototype_alignment"] is False


@pytest.mark.parametrize(
    ("name", "topology_enabled"),
    (("radar_robust_static_reliability_off", False), ("radar_robust_static_reliability_on", True)),
)
def test_static_reliability_templates_are_matched_and_fresh(name: str, topology_enabled: bool) -> None:
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")

    assert cfg["model"]["primary"]["fusion_mode"] == "trainable_static_reliability"
    assert cfg["temporal_missing"]["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert cfg["training"]["epochs"] == 40
    assert cfg["training"]["max_epochs"] == 40
    assert cfg["output"]["dir"] == "outputs/four_modal_topology_predictor_static_reliability"
    loss = cfg["loss"]["four_modal_topology"]
    assert bool(loss["use_beam_prototype_alignment"]) is topology_enabled
    assert float(loss["unimodal_soft_weight"]) == (0.1 if topology_enabled else 0.0)
    assert float(loss["lambda_proto"]) == (0.1 if topology_enabled else 0.0)


@pytest.mark.parametrize(
    ("name", "topology_enabled"),
    (("radar_robust_bounded_static_reliability_off", False), ("radar_robust_bounded_static_reliability_on", True)),
)
def test_bounded_static_reliability_templates_are_matched_and_fresh(
    name: str, topology_enabled: bool
) -> None:
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")

    assert cfg["model"]["primary"]["fusion_mode"] == "bounded_static_reliability"
    assert cfg["temporal_missing"]["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert cfg["training"]["epochs"] == 40
    assert cfg["training"]["max_epochs"] == 40
    assert cfg["output"]["dir"] == "outputs/four_modal_topology_predictor_bounded_static_reliability"
    loss = cfg["loss"]["four_modal_topology"]
    assert bool(loss["use_beam_prototype_alignment"]) is topology_enabled
    assert float(loss["unimodal_soft_weight"]) == (0.1 if topology_enabled else 0.0)
    assert float(loss["lambda_proto"]) == (0.1 if topology_enabled else 0.0)


@pytest.mark.parametrize(
    ("name", "topology_enabled", "soft_weight", "prototype_weight"),
    (
        ("radar_robust_masked_feature_fusion_on", True, 0.1, 0.1),
        ("radar_robust_masked_feature_fusion_off", False, 0.0, 0.0),
        ("radar_robust_masked_feature_fusion_soft_only", False, 0.1, 0.0),
        ("radar_robust_masked_feature_fusion_prototype_only", True, 0.0, 0.1),
    ),
)
def test_masked_feature_fusion_templates_are_matched_and_fresh(
    name: str,
    topology_enabled: bool,
    soft_weight: float,
    prototype_weight: float,
) -> None:
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")

    assert cfg["model"]["primary"]["fusion_mode"] == "masked_feature_mlp"
    assert cfg["temporal_missing"]["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert cfg["training"]["epochs"] == 40
    assert cfg["training"]["max_epochs"] == 40
    assert cfg["output"]["dir"] == "outputs/four_modal_topology_predictor_masked_feature_fusion"
    loss = cfg["loss"]["four_modal_topology"]
    assert bool(loss["use_beam_prototype_alignment"]) is topology_enabled
    assert float(loss["unimodal_soft_weight"]) == soft_weight
    assert float(loss["lambda_proto"]) == prototype_weight
    assert float(loss["lambda_modality_proto"]) == 0.0


@pytest.mark.parametrize(
    ("name", "joint_weight"),
    (
        ("radar_robust_masked_feature_fusion_hard", 0.0),
        ("radar_robust_masked_feature_fusion_joint", 0.1),
    ),
)
def test_joint_topology_templates_only_toggle_one_joint_weight(name: str, joint_weight: float) -> None:
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")
    loss = cfg["loss"]["four_modal_topology"]

    assert cfg["model"]["primary"]["fusion_mode"] == "masked_feature_mlp"
    assert cfg["temporal_missing"]["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert cfg["training"]["epochs"] == 40
    assert cfg["training"]["max_epochs"] == 40
    assert cfg["output"]["dir"] == "outputs/four_modal_topology_predictor_joint_topology"
    assert loss["joint_topology_weight"] == joint_weight
    assert loss["unimodal_soft_weight"] == 0.0
    assert loss["use_beam_prototype_alignment"] is False
    assert loss["lambda_proto"] == 0.0
    assert loss["lambda_modality_proto"] == 0.0


@pytest.mark.parametrize(
    ("name", "expected", "output_suffix"),
    (
        (
            "radar_robust_soft_only",
            {"hard_label_smoothing": 0.0, "unimodal_soft_weight": 0.1, "use_beam_prototype_alignment": False,
             "lambda_proto": 0.0, "lambda_modality_proto": 0.0},
            "soft_only",
        ),
        (
            "radar_robust_prototype_only",
            {"hard_label_smoothing": 0.0, "unimodal_soft_weight": 0.0, "use_beam_prototype_alignment": True,
             "lambda_proto": 0.1, "lambda_modality_proto": 0.0},
            "prototype_only",
        ),
        (
            "radar_robust_uniform_ls",
            {"hard_label_smoothing": 0.1, "unimodal_soft_weight": 0.0, "use_beam_prototype_alignment": False,
             "lambda_proto": 0.0, "lambda_modality_proto": 0.0},
            "uniform_ls",
        ),
    ),
)
def test_radar_robust_ablation_templates_use_preregistered_loss_and_independent_outputs(
    name: str, expected: dict[str, object], output_suffix: str
):
    cfg = load_config(ROOT / "tools/configs/topology_predictor" / f"{name}.yaml")

    assert cfg["temporal_missing"]["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert cfg["training"]["epochs"] == 40
    assert cfg["training"]["max_epochs"] == 40
    loss = cfg["loss"]["four_modal_topology"]
    for field, value in expected.items():
        assert loss[field] == value
    assert cfg["output"]["dir"] == f"outputs/four_modal_topology_predictor_radar_robust_{output_suffix}"
