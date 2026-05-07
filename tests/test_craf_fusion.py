from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.distillation.craf_losses import beam_soft_label_loss, sequence_cross_entropy  # noqa: E402
from kd_sensing.engine.craf_training import (  # noqa: E402
    generate_counterfactual_drop_masks,
    generate_modality_dropout_mask,
    loss_delta_to_gate_target,
)
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.models.fusion.craf import ReliabilityEstimator  # noqa: E402
from kd_sensing.registries import MODELS, import_default_components  # noqa: E402


def test_model_output_adapter_supports_legacy_tuple_and_craf_dict():
    logits = torch.randn(2, 5, 8)
    features = torch.randn(2, 5, 4)
    enhanced = torch.randn(2, 5, 6)

    legacy = adapt_model_output((logits, features, enhanced))
    assert legacy.logits is logits
    assert legacy.input_features is features
    assert legacy.output_features is enhanced
    assert legacy.diagnostics == {}

    craf = adapt_model_output({"logits": logits[:, -3:], "reliability": torch.ones(2, 2)})
    assert craf.logits.shape == (2, 3, 8)
    assert craf.input_features.shape == (2, 3, 8)
    assert craf.output_features.shape == (2, 3, 8)
    assert craf.diagnostics["reliability"].shape == (2, 2)
    assert select_prediction_slots(logits[:, -3:], num_pred=2).shape == (2, 3, 8)
    assert select_prediction_slots(logits, num_pred=2).shape == (2, 3, 8)


def test_craf_model_builds_modalities_and_rejects_invalid_configs():
    import_default_components()
    model = MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["mmwave", "image", "radar", "gps", "lidar"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {"min_gate": 0.05},
        }
    )

    assert model.modalities == ("image", "radar", "gps", "lidar", "mmwave")
    assert "craf_fusion" in MODELS.list()
    assert "token_transformer_fusion" in MODELS.list()

    with pytest.raises(ValueError, match="Unknown"):
        MODELS.build(
            {
                "type": "craf_fusion",
                "modalities": ["image", "wifi"],
                "feature_size": 16,
                "d_model": 16,
                "num_classes": 8,
                "num_pred": 1,
            }
        )
    with pytest.raises(ValueError, match="duplicates"):
        MODELS.build(
            {
                "type": "craf_fusion",
                "modalities": ["gps", "gps"],
                "feature_size": 16,
                "d_model": 16,
                "num_classes": 8,
                "num_pred": 1,
            }
        )


def test_craf_forward_shapes_and_force_mask_for_gps_mmwave():
    model = MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["mmwave", "gps"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 2,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {
                "min_gate": 0.2,
                "use_dataset_prior": True,
                "dataset_prior": {"gps": 0.8, "mmwave": 0.4},
            },
        }
    )
    model.eval()

    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(2, 4, 3),
            mmwave_batch=torch.randn(2, 4, 64),
            force_modality_mask=torch.tensor([[True, False], [False, True]]),
        )

    assert output["logits"].shape == (2, 3, 8)
    assert output["reliability"].shape == (2, 2)
    assert output["unimodal_logits"].shape == (2, 2, 3, 8)
    assert output["confidence"].shape == (2, 2, 2)
    assert output["effective_modality_mask"].tolist() == [[True, False], [False, True]]
    assert output["token_padding_mask"].shape == (2, 2, 4)
    assert torch.all(output["token_padding_mask"][0, 1])
    assert torch.all(output["token_padding_mask"][1, 0])
    assert output["reliability"][0, 1].item() == 0.0
    assert output["reliability"][1, 0].item() == 0.0
    assert output["reliability"][0, 0].item() >= 0.2
    assert output["reliability"][1, 1].item() >= 0.2


def test_token_transformer_baseline_forward_does_not_gate_available_tokens():
    model = MODELS.build(
        {
            "type": "token_transformer_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
        }
    )
    model.eval()

    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 3, 64),
            force_modality_mask=torch.tensor([[True, False]]),
        )

    assert output["logits"].shape == (1, 2, 8)
    assert output["reliability"].tolist() == [[1.0, 0.0]]


def test_reliability_estimator_masks_unavailable_and_applies_min_gate():
    estimator = ReliabilityEstimator(
        8,
        2,
        min_gate=0.25,
        use_dataset_prior=True,
        dataset_prior=[0.7, 0.2],
        modalities=("gps", "mmwave"),
    )
    output = estimator(
        torch.zeros(1, 2, 8),
        torch.zeros(1, 2, 2),
        torch.tensor([[True, False]]),
    )

    assert output.shape == (1, 2)
    assert output[0, 0].item() >= 0.25
    assert output[0, 1].item() == 0.0


def test_craf_loss_and_mask_helpers_handle_ignore_index_and_targets():
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [2, 3, 4]])

    loss, per_sample = sequence_cross_entropy(logits, labels)
    assert loss.ndim == 0
    assert per_sample.shape == (2,)
    assert beam_soft_label_loss(logits, labels, sigma=1.5, circular=True).ndim == 0

    available = torch.ones(4, 3, dtype=torch.bool)
    keep = generate_modality_dropout_mask(available, drop_prob=1.0, min_keep=1)
    assert torch.all(keep.sum(dim=1) >= 1)

    sample_one = generate_counterfactual_drop_masks(available, mode="sample_one")
    assert len(sample_one) == 1
    assert torch.all(sample_one[0][1].sum(dim=1) == 1)

    leave_one_out = generate_counterfactual_drop_masks(available, mode="leave_one_out")
    assert len(leave_one_out) == 3

    target = loss_delta_to_gate_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 0.5]),
        torch.tensor([[True, False, False], [False, True, False]]),
    )
    assert target.shape == (2, 3)
    assert target[0, 0].item() > 0.5
    assert 0.0 <= target[1, 1].item() <= 1.0


def test_craf_example_configs_load_without_affecting_legacy_config():
    craf = load_config(ROOT / "configs/fusion/craf_all_modalities_no_kd.yaml")
    image_radar = load_config(ROOT / "configs/fusion/craf_image_radar_no_kd.yaml")
    baseline = load_config(ROOT / "configs/fusion/token_transformer_image_radar_no_kd.yaml")
    legacy = load_config(ROOT / "configs/fusion/no_kd.yaml")

    assert craf["model"]["student"]["type"] == "craf_fusion"
    assert craf["model"]["student"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert image_radar["model"]["student"]["modalities"] == ["image", "radar"]
    assert baseline["model"]["student"]["type"] == "token_transformer_fusion"
    assert baseline["training"]["counterfactual"]["enabled"] is False
    assert legacy["model"]["student"]["type"] == "fusion_student"
    assert legacy["training"]["counterfactual"]["enabled"] is False
    assert legacy["loss"]["beam_soft"]["weight"] == 0.0


def test_craf_synthetic_train_and_evaluate_workflow(tmp_path: Path):
    cfg = _gps_mmwave_craf_smoke_cfg(tmp_path, epochs=1)
    result = train(cfg)
    run_dir = Path(result["run_dir"])
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))

    assert train_log["epoch_logs"][0]["train_batches"] == 2
    assert "craf_reliability" in train_log["epoch_logs"][0]
    assert train_log["train_counterfactual_loss"][0] >= 0.0

    eval_result = evaluate(
        cfg,
        weights=str(run_dir / "checkpoints" / "last.pth"),
        output_dir=str(tmp_path / "eval"),
    )
    assert eval_result["metrics"]["loss"] >= 0.0
    assert (Path(eval_result["run_dir"]) / "metrics.json").exists()


def _gps_mmwave_craf_smoke_cfg(tmp_path: Path, *, epochs: int) -> dict:
    return load_config(
        ROOT / "configs/fusion/craf_image_radar_no_kd.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seq_len=2",
            "data.dataset.num_pred=1",
            "data.dataset.num_classes=8",
            "data.dataset.use_gps=true",
            "data.dataset.use_mmwave=true",
            "data.dataset.mmwave_normalize=false",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "model.feature_size=16",
            "model.num_classes=8",
            "model.seq_length_teacher=2",
            "model.seq_length_student=2",
            "model.num_pred=1",
            "model.teacher.modalities=[\"gps\",\"mmwave\"]",
            "model.teacher.gps_input_size=3",
            "model.teacher.mmwave_input_size=64",
            "model.student.modalities=[\"gps\",\"mmwave\"]",
            "model.student.feature_size=16",
            "model.student.d_model=16",
            "model.student.num_classes=8",
            "model.student.num_pred=1",
            "model.student.num_heads=4",
            "model.student.num_layers=1",
            "model.student.gps_input_size=3",
            "model.student.mmwave_input_size=64",
            "loss.beam_soft.enabled=true",
            "loss.beam_soft.weight=0.1",
            "loss.unimodal_aux.weight=0.1",
            f"training.epochs={epochs}",
            "training.modality_dropout.enabled=true",
            "training.modality_dropout.drop_prob=0.2",
            "training.counterfactual.enabled=true",
            "training.counterfactual.start_epoch=0",
            "training.counterfactual.weight=0.1",
            "scheduler.type=none",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.run_name=craf_smoke",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
