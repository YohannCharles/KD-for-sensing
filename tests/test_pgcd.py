from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.data.sensor_degradation import SensorDegradationGenerator, assert_pgcd_channel_free
from kd_sensing.losses.pgcd import (
    beam_topology_distance_matrix,
    debiased_topology_drift,
    pgcd_quality_loss,
)
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.engine.training_extensions import BatchState
from kd_sensing.models.pgcd import PrototypeGuidedDegradationRouter
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


@ENCODERS.register("pgcd_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scalar = values.float().reshape(values.shape[0], values.shape[1], -1).mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _generator() -> SensorDegradationGenerator:
    return SensorDegradationGenerator(17)


def _gps() -> torch.Tensor:
    return torch.tensor([[10.0, 0.0, 1.0]]).expand(5, -1).clone()


@pytest.mark.parametrize(
    ("sensor", "value", "corruption"),
    (
        ("image", torch.ones(5, 3, 16, 16), "gaussian_blur"),
        ("lidar", torch.ones(5, 3, 16, 16), "point_dropout"),
        ("radar", torch.ones(5, 2, 16, 16), "detection_dropout"),
        ("gps", _gps(), "slow_bias_drift"),
    ),
)
def test_l0_is_identity_and_l4_is_missing(sensor, value, corruption) -> None:
    clean = _generator().generate(value, sensor, "sample", "sunny", 0, corruption, training=False)
    missing = _generator().generate(value, sensor, "sample", "sunny", 4, corruption, training=False)
    assert torch.equal(clean.corrupted_inputs, value)
    assert bool(clean.availability_mask.all())
    assert torch.count_nonzero(missing.corrupted_inputs) == 0
    assert not bool(missing.availability_mask.any())


def test_dropout_and_occlusion_are_monotonic() -> None:
    image = torch.ones(5, 3, 32, 32)
    lidar = torch.ones(5, 3, 32, 32)
    image_kept = []
    lidar_kept = []
    for severity in (1, 2, 3):
        image_result = _generator().generate(
            image, "image", "same", "sunny", severity, "patch_occlusion", training=False
        ).corrupted_inputs
        lidar_result = _generator().generate(
            lidar, "lidar", "same", "sunny", severity, "point_dropout", training=False
        ).corrupted_inputs
        image_kept.append(int(image_result.ne(0).sum()))
        lidar_kept.append(int(lidar_result.ne(0).sum()))
    assert image_kept[0] > image_kept[1] > image_kept[2]
    assert lidar_kept[0] > lidar_kept[1] > lidar_kept[2]


def test_radar_dropout_shares_ra_da_bin_mask() -> None:
    radar = torch.ones(5, 2, 16, 16)
    result = _generator().generate(
        radar, "radar", "sample", "rainy", 2, "detection_dropout", training=False
    ).corrupted_inputs
    assert torch.equal(result[:, 0].eq(0), result[:, 1].eq(0))


def test_gps_slow_drift_is_time_correlated_and_monotonic() -> None:
    offsets = []
    for severity in (1, 2, 3):
        result = _generator().generate(
            _gps(), "gps", "sample", "sunny", severity, "slow_bias_drift", training=False
        ).corrupted_inputs
        assert torch.allclose(result[0], _gps()[0], atol=1e-5)
        offsets.append(float(torch.linalg.vector_norm(result[-1] - _gps()[-1])))
    assert offsets[0] < offsets[1] < offsets[2]


def test_stale_frame_never_uses_future() -> None:
    value = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    result = _generator().generate(
        value, "gps", "sample", "sunny", 2, "one_step_stale", training=False
    )
    assert torch.equal(result.corrupted_inputs[1:], value[:-1])
    assert torch.equal(result.degradation_metadata["stale_mask"], torch.tensor([0, 1, 1, 1, 1], dtype=torch.bool))


def test_eval_is_deterministic_and_sample_specific() -> None:
    value = torch.ones(5, 3, 24, 24)
    first = _generator().generate(value, "image", "a", "foggy", 2, "patch_occlusion", training=False)
    replay = _generator().generate(value, "image", "a", "foggy", 2, "patch_occlusion", training=False)
    other = _generator().generate(value, "image", "b", "foggy", 2, "patch_occlusion", training=False)
    assert torch.equal(first.corrupted_inputs, replay.corrupted_inputs)
    assert not torch.equal(first.corrupted_inputs, other.corrupted_inputs)


def test_replicated_source_frames_share_corruption() -> None:
    value = torch.ones(5, 3, 24, 24)
    result = _generator().generate(
        value,
        "image",
        "sample",
        "sunny",
        2,
        "patch_occlusion",
        training=False,
        source_frame_ids=("a", "a", "b", "c", "c"),
    ).corrupted_inputs
    assert torch.equal(result[0], result[1])
    assert torch.equal(result[3], result[4])


def test_channel_config_and_batch_tensors_fail_closed() -> None:
    with pytest.raises(ValueError, match="forbids channel"):
        assert_pgcd_channel_free({"pgcd": {"use_channel": True}})
    with pytest.raises(ValueError, match="forbidden communication tensor"):
        assert_pgcd_channel_free({}, {"future_beam_power": torch.ones(2, 64)})
    with pytest.raises(ValueError, match="include_router_utility_targets=false"):
        assert_pgcd_channel_free({"data": {"dataset": {"include_router_utility_targets": True}}})


def test_topology_drift_self_is_zero_and_far_exceeds_neighbor() -> None:
    distance = beam_topology_distance_matrix(64, topology_id="cyclic_index_v1")
    clean = F.one_hot(torch.tensor([[0]]), 64).float()
    neighbor = F.one_hot(torch.tensor([[1]]), 64).float()
    far = F.one_hot(torch.tensor([[32]]), 64).float()
    self_drift, _ = debiased_topology_drift(clean, clean, distance)
    near_drift, _ = debiased_topology_drift(clean, neighbor, distance)
    far_drift, _ = debiased_topology_drift(clean, far, distance)
    assert torch.allclose(self_drift, torch.zeros_like(self_drift))
    assert bool((far_drift > near_drift).all())


def test_quality_target_detaches_teacher_and_estimator_gets_gradient() -> None:
    clean = torch.randn(2, 4, 64, requires_grad=True)
    corrupted = torch.randn(2, 4, 64, requires_grad=True)
    predicted = torch.rand(2, 4, requires_grad=True)
    result = pgcd_quality_loss(
        predicted,
        clean,
        corrupted,
        torch.tensor([[0], [8]]),
        torch.ones(2, 4, dtype=torch.bool),
        severity=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        corrupted_mask=torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool),
        variant="c7",
        topology_id="cyclic_index_v1",
    )
    result.total.backward()
    assert clean.grad is None
    assert predicted.grad is not None and bool(predicted.grad.abs().sum().gt(0))


def test_router_masks_missing_weights_and_beta_is_non_negative() -> None:
    router = PrototypeGuidedDegradationRouter(
        d_model=8, num_modalities=4, num_timesteps=3, variant="c7", hidden_dim=16, embedding_dim=2
    )
    features = torch.randn(2, 3, 4, 8)
    evidence = torch.randn(2, 3, 4, 64)
    available = torch.ones(2, 3, 4, dtype=torch.bool)
    available[0, :, 2] = False
    output = router(features, evidence, available)
    missing = ~available.reshape(2, -1)
    assert torch.equal(output["weights"].masked_select(missing), torch.zeros_like(output["weights"].masked_select(missing)))
    assert torch.allclose(output["weights"].sum(dim=-1), torch.ones(2))
    assert float(output["beta_reliability"].detach()) >= 0.0


def test_quality_estimator_api_has_no_teacher_or_injected_metadata() -> None:
    parameters = set(inspect.signature(PrototypeGuidedDegradationRouter.predict).parameters)
    assert not parameters & {"clean_features", "clean_logits", "severity", "corruption_type", "weather", "target_beam"}


def _model_config(variant: str = "c7") -> dict:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar", "gps", "lidar"],
        "seq_length": 3,
        "d_model": 8,
        "num_classes": 64,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "uniform_mean",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {name: {"type": "pgcd_test_sequence", "output_dim": 8} for name in ("image", "radar", "gps", "lidar")},
        "pgcd": {"variant": variant, "hidden_dim": 16, "embedding_dim": 2, "dropout": 0.0},
    }


def test_pgcd_model_reconstructs_fused_logits() -> None:
    model = MODELS.build(_model_config())
    inputs = {f"{name}_batch": torch.randn(2, 3, 1) for name in ("image", "radar", "gps", "lidar")}
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[0, 2, 3] = False
    output = model(**inputs, modality_temporal_mask=mask, missing_mask=mask.any(dim=1))
    weights = output["pgcd_block_router_weights"]
    assert output["pgcd_block_evidence_logits"].shape == (2, 12, 64)
    assert torch.allclose(output["logits"][:, 0], (weights.unsqueeze(-1) * output["pgcd_block_evidence_logits"]).sum(dim=1))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def _config(variant: str = "c7") -> dict:
    return {
        "data": {"dataset": {"include_router_utility_targets": False}},
        "model": {"primary": {"fusion_type": "uniform_mean", "head_type": "prototype", "pgcd": {"variant": variant}}},
        "temporal_missing": {"enabled": False},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "use_beam_prototype_alignment": True,
                "router_oracle_weight": 0.0,
                "superset_consistency": {"enabled": False},
                "pgcd": {"variant": variant},
            }
        },
    }


def test_pgcd_config_is_strict_and_channel_free() -> None:
    resolved = u_mask_beam_jepa_config(_config())
    assert resolved["pgcd"]["variant"] == "c7"
    invalid = _config()
    invalid["data"]["dataset"]["include_router_utility_targets"] = True
    with pytest.raises(ValueError, match="include_router_utility_targets=false"):
        u_mask_beam_jepa_config(invalid)


def test_pgcd_training_extension_runs_clean_corrupt_backward() -> None:
    model = MODELS.build(_model_config("c7"))
    cfg = _config("c7")
    cfg["model"]["primary"] = _model_config("c7")
    cfg["loss"]["type"] = "cross_entropy"
    context = type(
        "Context",
        (),
        {
            "cfg": cfg,
            "task": "fusion",
            "model_cfg": cfg["model"],
            "training_cfg": {},
            "primary_model": model,
            "task_criterion": nn.CrossEntropyLoss(),
            "run_dir": Path("outputs/pgcd_quick_search/test"),
            "device": torch.device("cpu"),
            "num_pred": 1,
            "num_classes": 64,
            "seq_length": 3,
            "non_blocking": False,
        },
    )()
    batch = {
        "image": torch.randn(2, 3, 3, 4, 4),
        "radar_ra": torch.rand(2, 3, 128, 64),
        "radar_da": torch.rand(2, 3, 128, 64),
        "gps": torch.tensor([[[10.0, 0.0, 1.0]] * 3] * 2),
        "lidar": torch.rand(2, 3, 3, 4, 4),
        "target_beam": torch.tensor([[2], [8]]),
        "sample_id": ["sample-a", "sample-b"],
        "domain_metadata": {"condition": ["sunny", "foggy"]},
        "gps_scaler_mean": torch.zeros(2, 3),
        "gps_scaler_scale": torch.ones(2, 3),
    }
    extension = UMaskBeamJEPATrainingExtension()
    state = extension.setup(context)
    controls = extension.before_forward(context, state, batch, batch["target_beam"], epoch=0, step=0)
    step = run_model_step(
        model,
        "fusion",
        batch,
        seq_length=3,
        num_pred=1,
        device=torch.device("cpu"),
        extra_model_kwargs=controls.model_kwargs,
    )
    batch_state = BatchState(
        epoch=0,
        step=0,
        batch=batch,
        labels=batch["target_beam"],
        primary_output=step.model_output,
        primary_logits=step.logits,
        controls=controls,
    )
    loss = extension.compute_base_loss(context, state, batch_state)
    assert loss is not None and torch.isfinite(loss.total_loss)
    loss.total_loss.backward()
    assert model.pgcd_router is not None
    assert any(parameter.grad is not None for parameter in model.pgcd_router.quality_estimator.parameters())
