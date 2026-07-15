from types import SimpleNamespace

import pytest
import torch

from kd_sensing.engine.scalar_metrics import materialize_batch_scalars
from kd_sensing.engine.prediction_objectives import PredictionTargets, prediction_observation_counts
from kd_sensing.engine.training_metrics import EpochMetricsRecorder


def _recorder() -> EpochMetricsRecorder:
    return EpochMetricsRecorder(
        objective="multitask",
        objective_metadata={
            "primary_loss": "multitask_total",
            "enabled_targets": ["target_beam", "occlusion_label", "position_target"],
            "enabled_heads": ["beam", "occlusion", "position"],
            "loss_weights": {"beam": 1.0, "occlusion": 1.0, "position": 1.0},
        },
        early_stopping_metric="val_multitask_loss",
        early_stopping_mode="min",
    )


def _batch_result(
    means: dict[str, float],
    denominators: dict[str, float],
    *,
    accuracy_correct: float,
    accuracy_total: float,
):
    numerators = {
        key: float(value) * float(denominators.get(key, 0.0))
        for key, value in means.items()
    }
    numerators["acc"] = float(accuracy_correct)
    denominators = {**denominators, "acc": float(accuracy_total)}
    return SimpleNamespace(
        metric_numerators=numerators,
        metric_denominators=denominators,
        scalar_diagnostics={},
    )


def test_epoch_metrics_use_each_metrics_own_effective_observation_count():
    recorder = _recorder()
    recorder.start_epoch(1e-3)
    recorder.update_batch(
        _batch_result(
            {
                "loss": 1.0,
                "task_loss": 2.0,
                "beam_soft_loss": 3.0,
                "unimodal_loss": 0.0,
                "occlusion_loss": 10.0,
                "position_loss": 0.0,
                "multitask_loss": 13.0,
                "los_loss": 0.0,
                "link_quality_loss": 0.0,
                "selection_multitask_loss": 0.0,
                "jepa_loss": 0.0,
            },
            {
                "loss": 2.0,
                "task_loss": 2.0,
                "beam_soft_loss": 2.0,
                "unimodal_loss": 2.0,
                "occlusion_loss": 1.0,
                "position_loss": 0.0,
                "multitask_loss": 2.0,
                "los_loss": 0.0,
                "link_quality_loss": 0.0,
                "selection_multitask_loss": 0.0,
                "jepa_loss": 0.0,
            },
            accuracy_correct=1.0,
            accuracy_total=2.0,
        ),
        step=0,
    )
    progress = recorder.update_batch(
        _batch_result(
            {
                "loss": 4.0,
                "task_loss": 5.0,
                "beam_soft_loss": 6.0,
                "unimodal_loss": 0.0,
                "occlusion_loss": 20.0,
                "position_loss": 7.0,
                "multitask_loss": 31.0,
                "los_loss": 0.0,
                "link_quality_loss": 0.0,
                "selection_multitask_loss": 0.0,
                "jepa_loss": 0.0,
            },
            {
                "loss": 1.0,
                "task_loss": 1.0,
                "beam_soft_loss": 1.0,
                "unimodal_loss": 1.0,
                "occlusion_loss": 2.0,
                "position_loss": 1.0,
                "multitask_loss": 1.0,
                "los_loss": 0.0,
                "link_quality_loss": 0.0,
                "selection_multitask_loss": 0.0,
                "jepa_loss": 0.0,
            },
            accuracy_correct=1.0,
            accuracy_total=1.0,
        ),
        step=1,
    )

    assert progress == pytest.approx({"loss": 2.0, "task": 3.0, "acc": 2.0 / 3.0})
    epoch_log, _, _, _ = recorder.finish_epoch(
        epoch=0,
        total_epochs=1,
        val_metrics=None,
        current_lr=1e-3,
        optimizer_groups=[],
        model_selection_enabled=False,
    )

    assert epoch_log["train_loss"] == pytest.approx(2.0)
    assert epoch_log["train_task_loss"] == pytest.approx(3.0)
    assert epoch_log["train_beam_soft_loss"] == pytest.approx(4.0)
    assert epoch_log["train_occlusion_loss"] == pytest.approx(50.0 / 3.0)
    assert epoch_log["train_position_loss"] == pytest.approx(7.0)
    assert epoch_log["train_acc"] == pytest.approx(2.0 / 3.0)
    assert epoch_log["train_loss_observation_count"] == 3
    assert epoch_log["train_occlusion_loss_observation_count"] == 3
    assert epoch_log["train_position_loss_observation_count"] == 1
    assert epoch_log["train_accuracy_observation_count"] == 3


def test_epoch_metrics_reject_zero_primary_loss_observations():
    recorder = _recorder()
    recorder.start_epoch(1e-3)

    with pytest.raises(ValueError, match="loss.*zero effective observations"):
        recorder.update_batch(
            _batch_result(
                {"loss": 0.0, "task_loss": 0.0},
                {"loss": 0.0, "task_loss": 0.0},
                accuracy_correct=0.0,
                accuracy_total=0.0,
            ),
            step=0,
        )


def test_batch_scalar_materialization_uses_one_cpu_transfer_and_no_item(monkeypatch: pytest.MonkeyPatch):
    original_cpu = torch.Tensor.cpu
    cpu_calls = 0

    def counted_cpu(tensor: torch.Tensor, *args, **kwargs):
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(tensor, *args, **kwargs)

    def forbidden_item(*_args, **_kwargs):
        raise AssertionError("batch scalar materialization must not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)

    numerators, denominators, diagnostics = materialize_batch_scalars(
        {
            "loss": (torch.tensor(6.0), torch.tensor(3.0)),
            "acc": (torch.tensor(2.0), torch.tensor(3.0)),
        },
        {
            "loss/primary": torch.tensor(2.0),
            "objective/weight_beam": 1.0,
        },
    )

    assert cpu_calls == 1
    assert numerators == pytest.approx({"loss": 6.0, "acc": 2.0})
    assert denominators == pytest.approx({"loss": 3.0, "acc": 3.0})
    assert diagnostics == pytest.approx({"loss/primary": 2.0, "objective/weight_beam": 1.0})


@pytest.mark.parametrize(
    ("objective", "expected_primary"),
    [
        ("beam", 3.0),
        ("occlusion", 2.0),
        ("position", 3.0),
        ("multitask", 3.0),
        ("current_los_classification", 2.0),
        ("current_link_quality", 2.0),
        ("selection_multitask", 3.0),
    ],
)
def test_prediction_observation_counts_are_objective_specific(objective: str, expected_primary: float):
    targets = PredictionTargets(
        labels=torch.tensor([[1, -100], [2, 3]]),
        occlusion_label=torch.zeros(2, 2),
        occlusion_valid=torch.tensor([[True, False], [False, True]]),
        position_target=torch.zeros(2, 2, 2),
        position_valid=torch.tensor([[True, False], [True, True]]),
        los_label=torch.zeros(2, 1),
        link_quality=torch.zeros(2, 1),
    )

    counts = prediction_observation_counts(
        targets,
        {"experiment": {"objective": objective}},
        reference=targets.labels,
    )

    assert float(counts["primary"]) == expected_primary
    assert float(counts["beam"]) == 3.0
    assert float(counts["occlusion"]) == 2.0
    assert float(counts["position"]) == 3.0
    assert float(counts["los"]) == 2.0
    assert float(counts["link_quality"]) == 2.0
