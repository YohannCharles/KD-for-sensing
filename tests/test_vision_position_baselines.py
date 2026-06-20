from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from kd_sensing.config import load_config
from kd_sensing.engine.batch import prepare_fusion_inputs
from kd_sensing.engine.evaluation_pass import _metadata_rows_from_batch
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.optim import build_task_criterion
from kd_sensing.engine.run_metadata import prediction_setup_metadata
from kd_sensing.engine.validator import validate
from kd_sensing.evaluation.metrics import calculate_topk_accuracy
from kd_sensing.registries import MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


def _prepared_steps(history: int = 5, horizon: int = 3) -> int:
    return history + horizon - 1


def test_evaluation_metadata_rows_preserve_horizon_sequences() -> None:
    metadata = {
        "dataset_index": torch.tensor([10, 11, 12, 13]),
        "sample_id": ["a", "b", "c", "d"],
        "raw_target_beam": [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([5, 6, 7, 8]),
            torch.tensor([9, 10, 11, 12]),
        ],
        "target_beam_label_source": [
            ("s0", "s1", "s2", "s3"),
            ("t0", "t1", "t2", "t3"),
            ("u0", "u1", "u2", "u3"),
        ],
        "beam_label_mapping": {"permutation": []},
    }

    rows = _metadata_rows_from_batch(metadata)

    assert len(rows) == 4
    assert rows[2]["dataset_index"] == 12
    assert rows[2]["raw_target_beam"] == [3, 7, 11]
    assert rows[2]["target_beam_label_source"] == ["s2", "t2", "u2"]
    assert rows[2]["beam_label_mapping"] == {"permutation": []}


def _camera_late_fusion_cfg() -> dict:
    return {
        "type": "vision_position_late_fusion",
        "baseline_preset": "camera_ae_gps",
        "modalities": ["image", "gps"],
        "feature_size": 16,
        "fusion_hidden_size": 16,
        "num_classes": 64,
        "num_pred": 3,
        "history_length": 5,
        "temporal_aggregation": "mean",
        "image_encoder_type": "camera_ae_frozen",
        "image_encoder": {
            "type": "camera_ae_frozen",
            "latent_dim": 8,
            "output_dim": 16,
            "image_size": 64,
            "require_checkpoint": False,
        },
        "gps_encoder": {"type": "gps_mlp", "output_dim": 16, "hidden_size": 16, "dropout": 0.0},
        "dropout": 0.0,
    }


def _resnet_late_fusion_cfg() -> dict:
    return {
        "type": "vision_position_late_fusion",
        "baseline_preset": "resnet_gps",
        "modalities": ["image", "gps"],
        "feature_size": 16,
        "fusion_hidden_size": 16,
        "temporal_hidden_size": 16,
        "num_classes": 64,
        "num_pred": 3,
        "history_length": 5,
        "temporal_aggregation": "gru",
        "image_encoder_type": "resnet18_imagenet_rgb",
        "image_encoder": {
            "type": "resnet18_imagenet_rgb",
            "output_dim": 16,
            "pretrained": False,
            "weights": None,
            "freeze_backbone": True,
        },
        "gps_encoder": {"type": "gps_mlp", "output_dim": 16, "hidden_size": 16, "dropout": 0.0},
        "dropout": 0.0,
    }


def _transformer_cfg() -> dict:
    return {
        "type": "vision_position_transformer_fusion",
        "baseline_preset": "transformer_image_gps",
        "modalities": ["image", "gps"],
        "feature_size": 16,
        "d_model": 16,
        "num_classes": 64,
        "num_pred": 3,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "max_seq_len": 5,
        "history_length": 5,
        "gps_input_size": 3,
        "token_organization": "cls_time_major_image_gps_tokens",
    }


def _gps_sequence_cfg() -> dict:
    return {
        "type": "gps_sequence_baseline",
        "baseline_preset": "gps_only_neural",
        "gps_input_size": 3,
        "feature_size": 16,
        "hidden_size": 16,
        "temporal_model": "lstm",
        "num_classes": 64,
        "num_pred": 3,
        "history_length": 5,
        "dropout": 0.0,
    }


def _assert_logits_contract(output: dict, batch_size: int = 2) -> None:
    assert output["logits"].shape == (batch_size, 3, 64)
    assert output["input_features"].shape[0] == batch_size
    assert output["output_features"].shape[:2] == (batch_size, 3)


def _assert_whole_model_metadata_contract(model: object, *, registry_name: str, modalities: list[str]) -> dict:
    assert hasattr(model, "training_strategy_metadata")
    metadata = model.training_strategy_metadata()
    assert metadata["model_registry_name"] == registry_name
    assert metadata["modalities"] == modalities
    assert metadata["architecture_category"] == "whole_model_exception"
    assert "uses_external_checkpoint" in metadata
    assert "freeze_policy" in metadata
    assert "consumes_reliability_metadata" in metadata
    return metadata


def test_vision_position_models_forward_shape_contracts():
    import_default_components()
    batch_size = 2
    steps = _prepared_steps()
    camera = MODELS.build(_camera_late_fusion_cfg())
    _assert_logits_contract(
        camera(
            image_batch=torch.rand(batch_size, steps, 3, 64, 64),
            gps_batch=torch.rand(batch_size, steps, 3),
        )
    )

    resnet = MODELS.build(_resnet_late_fusion_cfg())
    _assert_logits_contract(
        resnet(
            image_batch=torch.rand(batch_size, steps, 3, 224, 224),
            gps_batch=torch.rand(batch_size, steps, 3),
        )
    )

    transformer = MODELS.build(_transformer_cfg())
    _assert_logits_contract(
        transformer(
            image_batch=torch.rand(batch_size, steps, 3, 224, 224),
            gps_batch=torch.rand(batch_size, steps, 3),
        )
    )

    gps_only = MODELS.build(_gps_sequence_cfg())
    _assert_logits_contract(gps_only(gps_batch=torch.rand(batch_size, steps, 3)))


def test_vision_position_whole_model_metadata_contracts_are_auditable():
    import_default_components()

    camera = MODELS.build(_camera_late_fusion_cfg())
    camera_metadata = _assert_whole_model_metadata_contract(
        camera,
        registry_name="vision_position_late_fusion",
        modalities=["image", "gps"],
    )
    assert camera_metadata["freeze_policy"]["image_encoder"] is True

    transformer = MODELS.build(_transformer_cfg())
    _assert_whole_model_metadata_contract(
        transformer,
        registry_name="vision_position_transformer_fusion",
        modalities=["image", "gps"],
    )

    gps_only = MODELS.build(_gps_sequence_cfg())
    _assert_whole_model_metadata_contract(
        gps_only,
        registry_name="gps_sequence_baseline",
        modalities=["gps"],
    )


@pytest.mark.parametrize(
    ("preset", "primary_type", "modalities"),
    [
        ("camera_ae_gps", "vision_position_late_fusion", ["image", "gps"]),
        ("resnet_gps", "vision_position_late_fusion", ["image", "gps"]),
        ("transformer_image_gps", "vision_position_transformer_fusion", ["image", "gps"]),
        ("gps_only_neural", "gps_sequence_baseline", ["gps"]),
    ],
)
def test_vision_position_virtual_configs_load(preset: str, primary_type: str, modalities: list[str]):
    cfg = load_config(ROOT / f"configs/fusion/{preset}.yaml")

    assert cfg["experiment"]["task"] == "fusion"
    assert cfg["experiment"]["baseline_preset"] == preset
    assert cfg["model"]["modalities"] == modalities
    assert cfg["model"]["primary"]["modalities"] == modalities
    assert cfg["model"]["primary"]["type"] == primary_type
    assert cfg["data"]["dataset"]["seq_len"] == 1
    assert cfg["data"]["dataset"]["num_pred"] == 1
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "paper_distance_angle"
    assert cfg["data"]["dataset"]["gps_angle_offset_source"] == "paper_scene_default"
    assert cfg["data"]["dataset"]["beam_target_source"] == "current"
    assert cfg["model"]["num_pred"] == 1
    assert cfg["model"]["primary"]["num_pred"] == 1
    assert cfg["model"]["primary"]["gps_input_size"] == 2
    assert cfg["model"]["num_classes"] == 64
    assert cfg["evaluation"]["k_values"] == [1, 3, 5]
    assert cfg["evaluation"]["dba_distance_mode"] == "linear"
    assert cfg["evaluation"]["label_space"] == "64_beam"
    if preset == "camera_ae_gps":
        assert cfg["beambench_paper"]["table_iii_equivalent"] is False
        assert cfg["beambench_paper"]["protocol_aligned"] is True
        assert cfg["beambench_paper"]["recommended_table_iii_config"] == "configs/fusion/beambench_image_ae_gps_direct.yaml"
    if preset == "gps_only_neural":
        assert cfg["beambench_paper"]["table_iii_equivalent"] is False
        assert cfg["beambench_paper"]["protocol_aligned"] is False
        assert "Classical*" in cfg["beambench_paper"]["paper_rows_not_equivalent"]
        assert cfg["beambench_paper"]["recommended_table_iii_source"].endswith("challenge.py --type_list gps_dense")


def test_vision_position_field_selection_uses_only_enabled_modalities():
    image_gps = load_config(ROOT / "configs/fusion/resnet_gps.yaml")
    gps_only = load_config(ROOT / "configs/fusion/gps_only_neural.yaml")

    assert resolve_enabled_modalities(image_gps) == ("image", "gps")
    assert resolve_enabled_modalities(gps_only) == ("gps",)
    for key in ("use_lidar", "use_mmwave", "use_csi"):
        assert image_gps["data"]["dataset"][key] is False
        assert gps_only["data"]["dataset"][key] is False

    device = torch.device("cpu")
    image_gps_inputs = prepare_fusion_inputs(
        {
            "image": torch.rand(2, 5, 3, 64, 64),
            "gps": torch.rand(2, 5, 3),
            "target_beam": torch.zeros(2, 3, dtype=torch.long),
            "input_beam": torch.zeros(2, 5, dtype=torch.long),
        },
        seq_length=5,
        num_pred=3,
        device=device,
        modalities=image_gps["model"]["primary"]["modalities"],
        image_profile=image_gps["model"]["primary"]["image_profile"],
    )
    assert sorted(image_gps_inputs) == ["gps_batch", "image_batch"]

    gps_inputs = prepare_fusion_inputs(
        {"gps": torch.rand(2, 5, 3), "target_beam": torch.zeros(2, 3, dtype=torch.long)},
        seq_length=5,
        num_pred=3,
        device=device,
        modalities=gps_only["model"]["primary"]["modalities"],
    )
    assert sorted(gps_inputs) == ["gps_batch"]


def test_vision_position_error_paths_are_clear(tmp_path: Path):
    import_default_components()
    missing = tmp_path / "missing_camera_ae.pth"
    cfg = _camera_late_fusion_cfg()
    cfg["image_encoder"]["require_checkpoint"] = True
    cfg["image_encoder"]["checkpoint_path"] = str(missing)
    with pytest.raises(FileNotFoundError, match="Camera AE checkpoint"):
        MODELS.build(cfg)

    bad_profile = _resnet_late_fusion_cfg()
    bad_profile["image_encoder"]["image_channels"] = 1
    with pytest.raises(ValueError, match="Image encoder/profile mismatch"):
        MODELS.build(bad_profile)

    model = MODELS.build(_camera_late_fusion_cfg())
    with pytest.raises(ValueError, match="sequence dimensions must match"):
        model(
            image_batch=torch.rand(2, _prepared_steps(), 3, 64, 64),
            gps_batch=torch.rand(2, _prepared_steps() - 1, 3),
        )


def test_vision_position_metrics_and_metadata_are_auditable():
    cfg = load_config(
        ROOT / "configs/fusion/gps_only_neural.yaml",
        [
            "data.dataset.type=synthetic_sequence",
            "data.dataset.mock_data=true",
            "data.dataloader.num_workers=0",
            "model.primary.feature_size=8",
            "model.primary.hidden_size=8",
        ],
    )
    setup = prediction_setup_metadata(cfg)
    assert setup["baseline_preset"] == "gps_only_neural"
    assert setup["mock_data"] is True
    assert setup["uses_neural_network"] is True
    assert setup["metric_profile"] == "beambench_linear_topk"

    outputs = torch.randn(2, 1, 64)
    labels = torch.tensor([[0], [3]])
    topk, _ = calculate_topk_accuracy(outputs, labels, k_values=[1, 3, 5])
    assert set(topk) == {1, 3, 5}

    import_default_components()
    model = MODELS.build(cfg["model"]["primary"])
    samples = [
        {
            "gps": torch.rand(1, 2),
            "input_beam": torch.zeros(1, dtype=torch.long),
            "target_beam": torch.tensor([0], dtype=torch.long),
        },
        {
            "gps": torch.rand(1, 2),
            "input_beam": torch.zeros(1, dtype=torch.long),
            "target_beam": torch.tensor([3], dtype=torch.long),
        },
    ]
    dataloader = DataLoader(samples, batch_size=2)
    metrics = validate(
        model,
        dataloader,
        cfg,
        build_task_criterion(cfg),
        torch.device("cpu"),
    )
    assert isinstance(metrics["topk"], dict)
    assert set(metrics["topk"]) == {"1", "3", "5"}
    assert metrics["configured_topk"] == [1, 3, 5]
    assert "val_top1_avg" in metrics
    assert "val_top3_avg" in metrics
    assert "val_top5_avg" in metrics
    assert metrics["baseline_preset"] == "gps_only_neural"
    assert metrics["mock_data"] is True
    assert metrics["metric_profile"] == "beambench_linear_topk"
