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
from kd_sensing.distillation.craf_losses import (  # noqa: E402
    beam_soft_label_loss,
    counterfactual_sequence_ce,
    sequence_cross_entropy,
)
from kd_sensing.engine.craf_training import (  # noqa: E402
    generate_context_marginal_masks,
    generate_counterfactual_drop_masks,
    generate_modality_dropout_mask,
    loss_delta_to_binary_gate_target,
    loss_delta_to_gate_target,
)
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.trainer import _unimodal_aux_loss, train  # noqa: E402
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

    craf = adapt_model_output(
        {
            "logits": logits[:, -3:],
            "input_features": features[:, -3:],
            "output_features": enhanced[:, -3:],
            "reliability": torch.ones(2, 2),
        }
    )
    assert craf.logits.shape == (2, 3, 8)
    assert craf.input_features.shape == (2, 3, 4)
    assert craf.output_features.shape == (2, 3, 6)
    assert craf.diagnostics["reliability"].shape == (2, 2)
    missing_features = adapt_model_output({"logits": logits[:, -3:]})
    assert missing_features.input_features is None
    assert missing_features.output_features is None
    assert adapt_model_output((logits, None, "missing")).output_features is None
    with pytest.raises(ValueError, match="must contain"):
        adapt_model_output({"confidence": torch.ones(2, 2)})
    assert select_prediction_slots(logits[:, -2:], num_pred=2).shape == (2, 2, 8)
    assert select_prediction_slots(logits, num_pred=2).shape == (2, 2, 8)
    with pytest.raises(ValueError, match="future labels require 6 slots"):
        select_prediction_slots(logits, num_pred=6)


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

    assert output["logits"].shape == (2, 2, 8)
    assert model.horizon == model.num_pred == 2
    assert model.prediction_head.horizon == 2
    assert model.unimodal_head.horizon == 2
    assert output["reliability"].shape == (2, 2)
    assert output["unimodal_logits"].shape == (2, 2, 2, 8)
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

    assert output["logits"].shape == (1, 1, 8)
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


def test_fixed_prior_gate_uses_configured_prior_as_gate():
    estimator = ReliabilityEstimator(
        4,
        2,
        gate_type="fixed_prior",
        use_dataset_prior=True,
        dataset_prior={"gps": 0.9, "mmwave": 0.2},
        modalities=("gps", "mmwave"),
    )
    output = estimator(
        torch.randn(1, 2, 4),
        torch.zeros(1, 2, 2),
        torch.tensor([[True, False]]),
    )

    assert output[0, 0].item() == pytest.approx(0.9)
    assert output[0, 1].item() == 0.0


def test_softmax_gate_normalizes_available_modalities_and_uses_temperature():
    estimator = ReliabilityEstimator(
        1,
        2,
        hidden_size=1,
        gate_type="softmax",
        min_gate=0.0,
        scale_by_available=True,
        modalities=("gps", "mmwave"),
    )
    with torch.no_grad():
        estimator.net[0].weight.zero_()
        estimator.net[0].weight[0, 0] = 1.0
        estimator.net[0].bias.zero_()
        estimator.net[2].weight.fill_(1.0)
        estimator.net[2].bias.zero_()

    modality_repr = torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]])
    confidence = torch.zeros(2, 2, 2)
    available = torch.tensor([[True, True], [True, False]])
    smooth = estimator(modality_repr, confidence, available, gate_temperature=5.0)
    sharp = estimator(modality_repr, confidence, available, gate_temperature=0.1)

    assert torch.allclose(smooth[0].sum(), torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(sharp[0].sum(), torch.tensor(2.0), atol=1e-5)
    assert sharp[0].max().item() > smooth[0].max().item()
    assert sharp[1, 1].item() == 0.0
    assert sharp[1, 0].item() == pytest.approx(1.0)


def test_craf_forward_can_force_warmup_gate_to_one():
    model = MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {"gate_type": "softmax", "min_gate": 0.1},
        }
    )
    model.eval()
    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 3, 64),
            force_modality_mask=torch.tensor([[True, False]]),
            force_reliability_gate=1.0,
            gate_temperature=2.0,
        )

    assert output["reliability"].tolist() == [[1.0, 0.0]]
    assert output["gate_temperature"].item() == pytest.approx(2.0)


def test_craf_loss_and_mask_helpers_handle_ignore_index_and_targets():
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [2, 3, 4]])

    loss, per_sample = sequence_cross_entropy(logits, labels)
    assert loss.ndim == 0
    assert per_sample.shape == (2,)
    assert torch.allclose(counterfactual_sequence_ce(logits, labels), per_sample)
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

    binary_target, valid = loss_delta_to_binary_gate_target(
        torch.tensor([0.2, -0.3, 0.01]),
        torch.tensor(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        ),
        ignore_delta_eps=0.05,
    )
    assert binary_target.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert valid.tolist() == [[True, False, False], [False, True, False], [False, False, False]]

    context_specs = generate_context_marginal_masks(available, num_samples=2, min_keep=1)
    assert len(context_specs) == 2
    for context_mask, with_target_mask, target_mask in context_specs:
        assert torch.all((context_mask & target_mask) == 0)
        assert torch.equal(with_target_mask, context_mask | target_mask)
        assert torch.all(target_mask.sum(dim=1) == 1)
        assert torch.all(context_mask.sum(dim=1) >= 1)


def test_unimodal_aux_loss_requires_exact_future_horizon():
    labels = torch.tensor([[0, 1], [2, 3]])
    zero = torch.tensor(0.0)
    valid_logits = torch.zeros(2, 2, 2, 5)
    extra_slot_logits = torch.zeros(2, 2, 3, 5)

    assert _unimodal_aux_loss(
        valid_logits,
        labels,
        torch.ones(2, 2, dtype=torch.bool),
        num_pred=2,
        ignore_index=-100,
        zero=zero,
    ).ndim == 0
    with pytest.raises(ValueError, match="exactly match num_pred"):
        _unimodal_aux_loss(
            extra_slot_logits,
            labels,
            torch.ones(2, 2, dtype=torch.bool),
            num_pred=2,
            ignore_index=-100,
            zero=zero,
        )


def test_craf_example_configs_load_without_affecting_legacy_config():
    craf = load_config(ROOT / "configs/fusion/craf_all_modalities_no_kd.yaml")
    stabilized = load_config(ROOT / "configs/fusion/craf_all_modalities_stabilized_no_kd.yaml")
    token_all = load_config(ROOT / "configs/fusion/token_transformer_all_modalities_no_kd.yaml")
    no_cf = load_config(ROOT / "configs/fusion/craf_all_modalities_no_counterfactual.yaml")
    prior_sanity = load_config(ROOT / "configs/fusion/craf_all_modalities_fixed_prior_sanity.yaml")
    image_radar = load_config(ROOT / "configs/fusion/craf_image_radar_no_kd.yaml")
    baseline = load_config(ROOT / "configs/fusion/token_transformer_image_radar_no_kd.yaml")
    legacy = load_config(ROOT / "configs/fusion/no_kd.yaml")

    assert craf["model"]["student"]["type"] == "craf_fusion"
    assert craf["model"]["student"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert stabilized["model"]["student"]["reliability"]["gate_type"] == "softmax"
    assert stabilized["training"]["counterfactual"]["mode"] == "context_marginal"
    assert token_all["model"]["student"]["type"] == "token_transformer_fusion"
    assert no_cf["training"]["counterfactual"]["enabled"] is False
    assert prior_sanity["model"]["student"]["reliability"]["gate_type"] == "fixed_prior"
    assert prior_sanity["model"]["student"]["reliability"]["use_dataset_prior"] is True
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


def test_craf_stabilized_training_logs_warmup_ramp_and_counterfactual_stats(tmp_path: Path):
    cfg = _gps_mmwave_craf_stabilized_cfg(tmp_path)
    result = train(cfg)
    train_log = json.loads((Path(result["run_dir"]) / "train_log.json").read_text(encoding="utf-8"))
    first_epoch, second_epoch = train_log["epoch_logs"]

    assert first_epoch["loss/gate_weight_effective"] == pytest.approx(0.0)
    assert first_epoch["loss/unimodal_aux_weight"] == pytest.approx(0.1)
    assert first_epoch["craf_reliability"]["gps"] == pytest.approx(1.0)
    assert first_epoch["craf_reliability"]["mmwave"] == pytest.approx(1.0)
    assert second_epoch["loss/gate_weight_effective"] == pytest.approx(0.1)
    assert second_epoch["loss/unimodal_aux_weight"] == pytest.approx(0.0)
    assert any(key.startswith("cf/delta_mean_") for key in second_epoch)
    assert any(key.startswith("cf/target_valid_rate_") for key in second_epoch)


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


def _gps_mmwave_craf_stabilized_cfg(tmp_path: Path) -> dict:
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
            "model.student.reliability.gate_type=softmax",
            "model.student.reliability.gate_temperature_start=2.0",
            "model.student.reliability.gate_temperature_end=1.0",
            "model.student.reliability.gate_temperature_anneal_epochs=1",
            "model.student.reliability.min_gate=0.0",
            "loss.beam_soft.enabled=false",
            "loss.beam_soft.weight=0.0",
            "loss.unimodal_aux.weight=0.0",
            "loss.uni_weight_warmup=0.1",
            "loss.uni_weight_after_warmup=0.0",
            "loss.gate_weight=0.2",
            "loss.gate_ramp_epochs=2",
            "training.epochs=2",
            "training.warmup_epochs=1",
            "training.modality_dropout.enabled=false",
            "training.counterfactual.enabled=true",
            "training.counterfactual.mode=sample_one",
            "training.counterfactual.start_epoch=1",
            "training.counterfactual.ignore_delta_eps=0.0",
            "training.counterfactual.num_drop_per_batch=1",
            "scheduler.type=none",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.run_name=craf_stabilized",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
