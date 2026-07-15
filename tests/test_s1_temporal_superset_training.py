from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kd_sensing.data.difficulty.operators.temporal import (
    TEMPORAL_SUPERSET_PAYLOAD_KEY,
    TemporalMissingOperator,
)
from kd_sensing.data.difficulty.schema import DifficultyContext, normalize_config_difficulty
from kd_sensing.losses.u_mask_beam_jepa import (
    UMaskBeamJEPATrainingExtension,
    _beam_monotonic_ranking_loss,
    _circular_beam_risk,
    _confidence_gated_temperature_kl,
    u_mask_beam_jepa_config,
    u_mask_beam_jepa_loss,
)


MODALITIES = ("image", "gps")


def _batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.arange(1, 1 + 2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3),
        "gps": torch.arange(101, 101 + 2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3),
        "target_beam": torch.tensor([[0], [1]]),
    }


def _apply_temporal_operator(batch: dict[str, torch.Tensor], *, preserve: bool) -> dict[str, torch.Tensor]:
    cfg = {
        "experiment": {"seed": 7},
        "model": {"primary": {"modalities": list(MODALITIES)}},
        "temporal_missing": {
            "enabled": True,
            "mode": "block",
            "prob": 1.0,
            "block_len": 2,
            "apply": "train",
            "seed": 7,
            "ensure_at_least_one_frame": True,
            "preserve_unmasked_for_superset": preserve,
        },
    }
    profiles = normalize_config_difficulty(cfg)
    profile = profiles[0]
    operator_cfg = profile.operators[0]
    TemporalMissingOperator(**dict(operator_cfg.params))(
        batch,
        config=operator_cfg,
        profile=profile,
        context=DifficultyContext(stage="train", split="train", seed=7, step=0),
    )
    return batch


def _loss_output(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"logits": logits, "output_features": logits[:, 0]}


def test_temporal_operator_preserves_unmasked_references_and_base_mask_only_when_opted_in() -> None:
    batch = _batch()
    original_image = batch["image"]
    original_gps = batch["gps"]
    _apply_temporal_operator(batch, preserve=True)

    payload = batch[TEMPORAL_SUPERSET_PAYLOAD_KEY]
    student_mask = batch["modality_temporal_mask"]
    base_mask = payload["base_mask"]
    assert payload["inputs"]["image"] is original_image
    assert payload["inputs"]["gps"] is original_gps
    assert payload["inputs"]["image"].data_ptr() == original_image.data_ptr()
    assert bool((student_mask & ~base_mask).any()) is False
    assert bool(student_mask.any(dim=(1, 2)).all())
    assert bool(base_mask.any(dim=(1, 2)).all())
    assert torch.all(batch["image"][~student_mask[:, :, 0]] == 0)
    assert torch.all(batch["gps"][~student_mask[:, :, 1]] == 0)
    assert torch.equal(payload["inputs"]["image"], original_image)

    disabled = _apply_temporal_operator(_batch(), preserve=False)
    assert TEMPORAL_SUPERSET_PAYLOAD_KEY not in disabled


def test_superset_config_metadata_is_method_owned_and_rejects_feature_l2() -> None:
    value = {
        "enabled": True,
        "confidence_gated_kl": True,
        "kl_weight": 0.2,
        "temperature": 2.0,
        "beam_monotonic_rank": True,
        "rank_weight": 0.1,
        "rank_tolerance": 0.05,
        "feature_l2_weight": 0.0,
    }
    resolved = u_mask_beam_jepa_config(
        {
            "model": {"primary": {}},
            "training": {"superset_consistency": value},
            "loss": {"u_mask_beam_jepa": {"enabled": True, "superset_consistency": value}},
        }
    )["superset_consistency"]

    assert resolved["mode"] == "same_primary_model_online_stop_gradient"
    assert resolved["confidence_gated_kl"] is True
    assert resolved["beam_monotonic_rank"] is True
    assert resolved["rank_tolerance"] == 0.05
    assert "rank_margin" not in resolved
    assert resolved["feature_l2_weight"] == 0.0
    assert not ({"teacher_checkpoint", "teacher_source", "legacy_kd", "distillation", "distiller"} & set(resolved))

    with pytest.raises(ValueError, match="feature_l2_weight must remain 0"):
        u_mask_beam_jepa_config(
            {
                "training": {"superset_consistency": {"enabled": True, "feature_l2_weight": 0.1}},
                "loss": {"u_mask_beam_jepa": {"enabled": True}},
            }
        )
    with pytest.raises(ValueError, match="use rank_tolerance"):
        u_mask_beam_jepa_config(
            {
                "training": {"superset_consistency": {"enabled": True, "rank_margin": 0.1}},
                "loss": {"u_mask_beam_jepa": {"enabled": True}},
            }
        )


def test_confidence_gated_temperature_kl_uses_correctness_entropy_and_weighted_normalization() -> None:
    student = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[8.0, 0.0, 0.0, 0.0]], [[0.0, 8.0, 0.0, 0.0]], [[0.1, 0.0, 0.0, 0.0]]]
    )
    labels = torch.zeros(3, 1, dtype=torch.long)
    temperature = 2.0

    weighted, raw, gate = _confidence_gated_temperature_kl(
        student,
        teacher,
        labels,
        temperature=temperature,
    )
    per_sample = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean(dim=1) * temperature**2
    expected = (per_sample * gate).sum() / gate.sum()

    assert gate[1] == 0
    assert gate[0] > gate[2] > 0
    assert torch.allclose(weighted, expected)
    assert torch.allclose(raw, per_sample.mean())
    weighted.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()

    integrated_student = student.detach().clone().requires_grad_(True)
    integrated = u_mask_beam_jepa_loss(
        _loss_output(integrated_student),
        labels,
        use_teacher=False,
        use_jepa_loss=False,
        teacher_output={"logits": teacher},
        use_superset_confidence_gated_kl=True,
        lambda_superset_consistency=0.2,
        superset_temperature=temperature,
    )
    diagnostics = integrated["diagnostics"]
    assert diagnostics["superset_consistency/gate_active_ratio"] == pytest.approx(2 / 3)
    assert diagnostics["superset_consistency/weighted_kl"] == pytest.approx(weighted.detach().item())
    assert diagnostics["superset_consistency/feature_l2_weight"] == 0.0


def test_circular_beam_risk_wraparound_tolerance_diagnostics_and_zero_weight_behavior() -> None:
    labels = torch.tensor([[0]])
    at_last_beam = torch.full((1, 1, 64), -30.0)
    at_last_beam[0, 0, 63] = 30.0
    at_second_beam = torch.full((1, 1, 64), -30.0)
    at_second_beam[0, 0, 2] = 30.0
    assert _circular_beam_risk(at_last_beam, labels).item() == pytest.approx(1.0, abs=1e-6)

    rank, teacher_risk, student_risk, partial_excess, superset_worse = _beam_monotonic_ranking_loss(
        at_second_beam,
        at_last_beam,
        labels,
        tolerance=0.25,
    )
    assert teacher_risk.item() == pytest.approx(1.0, abs=1e-6)
    assert student_risk.item() == pytest.approx(2.0, abs=1e-6)
    assert rank.item() == pytest.approx(0.75, abs=1e-6)
    assert partial_excess.tolist() == [True]
    assert superset_worse.tolist() == [False]

    student = at_second_beam.clone().requires_grad_(True)
    base = u_mask_beam_jepa_loss(_loss_output(student), labels, use_teacher=False, use_jepa_loss=False)
    ranked = u_mask_beam_jepa_loss(
        _loss_output(student),
        labels,
        use_teacher=False,
        use_jepa_loss=False,
        teacher_output={"logits": at_last_beam},
        use_beam_monotonic_rank=True,
        lambda_beam_monotonic_rank=0.0,
        beam_monotonic_tolerance=0.25,
    )
    assert torch.equal(ranked["loss"], base["loss"])
    diagnostics = ranked["diagnostics"]
    assert diagnostics["loss/beam_monotonic_rank"] == pytest.approx(0.75, abs=1e-6)
    assert diagnostics["beam_monotonic_rank/risk_gap"] == pytest.approx(1.0, abs=1e-6)
    assert diagnostics["beam_monotonic_rank/partial_excess_violation_rate"] == 1.0
    assert diagnostics["beam_monotonic_rank/superset_worse_rate"] == 0.0


def test_active_rank_gradient_step_lowers_student_risk_and_keeps_teacher_stop_gradient() -> None:
    labels = torch.tensor([[0]])
    student = torch.tensor([[[0.0, 0.0, 2.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    before = _circular_beam_risk(student, labels).detach()
    loss, _, _, partial_excess, _ = _beam_monotonic_ranking_loss(
        student,
        teacher,
        labels,
        tolerance=0.0,
    )

    loss.backward()
    assert partial_excess.tolist() == [True]
    assert teacher.grad is None
    with torch.no_grad():
        student -= student.grad
    after = _circular_beam_risk(student, labels).detach()

    assert after.item() < before.item()


class _PrimaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modalities = MODALITIES
        self.dropout = torch.nn.Dropout()


def test_superset_forward_is_shared_no_grad_eval_and_restores_exact_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _apply_temporal_operator(_batch(), preserve=True)
    payload = batch[TEMPORAL_SUPERSET_PAYLOAD_KEY]
    primary = _PrimaryModel()
    primary.train()
    primary.dropout.eval()
    calls: list[dict[str, object]] = []

    def fake_run_model_step(model, task, teacher_batch, **kwargs):
        calls.append({"batch": teacher_batch, "kwargs": kwargs})
        assert model.training is False
        assert primary.dropout.training is False
        assert torch.is_grad_enabled() is False
        assert teacher_batch["image"] is payload["inputs"]["image"]
        assert torch.equal(teacher_batch["modality_temporal_mask"], payload["base_mask"])
        assert torch.equal(kwargs["extra_model_kwargs"]["missing_mask"], payload["base_mask"].any(dim=1))
        return SimpleNamespace(
            logits=torch.randn(2, 1, 64),
            model_output=SimpleNamespace(output_features=torch.randn(2, 8)),
        )

    monkeypatch.setattr("kd_sensing.engine.runtime.run_model_step", fake_run_model_step)
    value = {
        "enabled": True,
        "confidence_gated_kl": True,
        "beam_monotonic_rank": True,
        "feature_l2_weight": 0.0,
    }
    cfg = {
        "model": {"primary": {"modalities": list(MODALITIES)}},
        "training": {"superset_consistency": value},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "missing_pattern": {"available_modalities": list(MODALITIES)},
                "superset_consistency": value,
            }
        },
    }
    context = SimpleNamespace(
        cfg=cfg,
        primary_model=primary,
        model_cfg=cfg["model"],
        device=torch.device("cpu"),
        task="fusion",
        seq_length=5,
        num_pred=1,
        non_blocking=False,
        run_dir=Path("."),
    )
    state = {"config": u_mask_beam_jepa_config(cfg)}

    UMaskBeamJEPATrainingExtension().before_forward(context, state, batch, batch["target_beam"], epoch=0)

    assert len(calls) == 1
    assert primary.training is True
    assert primary.dropout.training is False
    assert state["online_teacher"]["logits"].requires_grad is False


def test_disabled_superset_consistency_executes_no_teacher_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kd_sensing.engine.runtime.run_model_step",
        lambda *args, **kwargs: pytest.fail("disabled superset consistency must not run a teacher forward"),
    )
    primary = _PrimaryModel()
    cfg = {
        "model": {"primary": {"modalities": list(MODALITIES)}},
        "training": {"superset_consistency": {"enabled": False}},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "missing_pattern": {"available_modalities": list(MODALITIES)},
            }
        },
    }
    context = SimpleNamespace(
        cfg=cfg,
        primary_model=primary,
        model_cfg=cfg["model"],
        device=torch.device("cpu"),
    )
    state = {"config": u_mask_beam_jepa_config(cfg)}
    batch = _batch()

    UMaskBeamJEPATrainingExtension().before_forward(context, state, batch, batch["target_beam"], epoch=0)

    assert state["online_teacher"] is None
    assert TEMPORAL_SUPERSET_PAYLOAD_KEY not in batch
