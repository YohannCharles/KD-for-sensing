from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kd_sensing.engine import evaluation_pass


def _config() -> dict:
    return {
        "experiment": {"task": "fusion", "objective": "beam"},
        "model": {"num_pred": 1, "num_classes": 4, "seq_length": 1, "primary": {"modalities": ["image", "radar", "gps", "lidar"]}},
        "evaluation": {"k_values": [1, 3], "dba_delta": 5, "dba_distance_mode": "circular"},
        "training": {"amp": {"enabled": False}},
    }


def _patch_lightweight_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation_pass, "prepare_evaluation_batch", lambda batch: batch)
    monkeypatch.setattr(
        evaluation_pass,
        "prepare_task_labels",
        lambda batch, **_kwargs: batch["labels"],
    )
    monkeypatch.setattr(
        evaluation_pass,
        "run_model_step",
        lambda _model, _task, batch, **_kwargs: SimpleNamespace(logits=batch["logits"], model_output=object()),
    )
    monkeypatch.setattr(
        evaluation_pass,
        "compute_prediction_loss",
        lambda _output, _targets, _cfg, *, beam_total_loss, **_kwargs: SimpleNamespace(total=beam_total_loss),
    )


def test_evaluation_pass_streams_by_default_and_captures_only_on_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lightweight_step(monkeypatch)
    batches = [
        {
            "logits": torch.tensor([[[4.0, 1.0, 0.0, 0.0]], [[0.0, 1.0, 4.0, 0.0]]]),
            "labels": torch.tensor([[0], [2]]),
            "metadata": {"sample_id": ["a", "b"]},
        },
        {
            "logits": torch.tensor([[[0.0, 4.0, 1.0, 0.0]]]),
            "labels": torch.tensor([[1]]),
            "metadata": {"sample_id": ["c"]},
        },
    ]
    criterion = lambda logits, labels: F.cross_entropy(logits, labels)
    model = torch.nn.Identity()

    streamed = evaluation_pass.run_evaluation_pass(
        model, batches, _config(), criterion, torch.device("cpu")
    )
    captured = evaluation_pass.run_evaluation_pass(
        model, batches, _config(), criterion, torch.device("cpu"), capture_outputs=True
    )

    assert streamed.outputs is None
    assert streamed.labels is None
    assert streamed.metadata == []
    assert streamed.metrics["prediction_capture"] is False
    assert captured.outputs is not None and tuple(captured.outputs.shape) == (3, 1, 4)
    assert captured.labels is not None and tuple(captured.labels.shape) == (3, 1)
    assert [row["sample_id"] for row in captured.metadata] == ["a", "b", "c"]
    legacy = evaluation_pass._metrics_from_outputs(
        captured.metrics["loss"],
        captured.outputs,
        captured.labels,
        _config(),
    )
    for key in ("val_loss", "val_acc", "val_atop3", "val_adba"):
        assert streamed.metrics[key] == pytest.approx(captured.metrics[key])
        assert streamed.metrics[key] == pytest.approx(legacy[key])
