from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.evaluation_pass import run_evaluation_pass  # noqa: E402
from kd_sensing.engine.hist_beam_adaptation import (  # noqa: E402
    adapt_hist_beam_target,
    apply_hist_beam_adaptation_strategy,
)
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.hist_beam_history_anchor import history_anchor_run_metadata  # noqa: E402
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss  # noqa: E402
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import (  # noqa: E402
    markov_delta_baseline_metrics,
    source_prior_collapse_diagnostics,
    write_hist_beam_predictions,
)
from kd_sensing.evaluation.hist_beam_residuals import (  # noqa: E402
    circular_residual_labels,
    residual_logits_to_absolute_logits,
    residual_topk_to_absolute,
)
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402


def test_circular_residual_and_reconstruction_wraparound_topk():
    future = torch.tensor([[1, 62]])
    last = torch.tensor([62])
    residual = circular_residual_labels(future, last, num_classes=64, sample_ids=["wrap"])
    logits = torch.full((1, 1, 64), -10.0)
    logits[0, 0, 0] = 3.0
    logits[0, 0, 1] = 2.0
    logits[0, 0, 63] = 1.0

    absolute = residual_logits_to_absolute_logits(logits, last, num_classes=64)
    top = residual_topk_to_absolute(logits, last, k=3)

    assert residual.tolist() == [[3, 0]]
    assert int(absolute.argmax(dim=-1).item()) == 62
    assert top["residual_topk"].tolist() == [[[0, 1, 63]]]
    assert top["absolute_topk"].tolist() == [[[62, 63, 61]]]
    with pytest.raises(ValueError, match="sample_id=bad"):
        circular_residual_labels(torch.tensor([[99]]), torch.tensor([0]), num_classes=64, sample_ids=["bad"])


class _NoHistoryModel(torch.nn.Module):
    def forward(self, gps_batch=None, mmwave_batch=None, **kwargs):  # noqa: ANN001
        assert "input_beam_batch" not in kwargs
        assert "last_beam_batch" not in kwargs
        batch_size = gps_batch.shape[0]
        return {"logits": torch.zeros(batch_size, 1, 4)}


class _HistoryResidualModel(torch.nn.Module):
    def forward(self, gps_batch=None, mmwave_batch=None, input_beam_batch=None, last_beam_batch=None, **kwargs):  # noqa: ANN001, ARG002
        assert input_beam_batch is not None
        assert last_beam_batch is not None
        residual_logits = torch.tensor(
            [[[0.0, 5.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 5.0]]],
            dtype=torch.float32,
            device=gps_batch.device,
        )[: gps_batch.shape[0]]
        absolute_logits = residual_logits_to_absolute_logits(residual_logits, last_beam_batch, num_classes=4)
        return {
            "logits": absolute_logits,
            "residual_logits": residual_logits,
            "last_beam": last_beam_batch,
        }


def _eval_cfg(*, history_anchor: bool) -> dict:
    cfg = {
        "experiment": {"task": "fusion", "objective": "beam"},
        "data": {"dataset": {}},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length_student": 2,
            "num_classes": 4,
            "student": {"modalities": ["gps", "mmwave"], "num_classes": 4, "group_size": 2},
        },
        "training": {"transfer": {"non_blocking": False}, "amp": {"enabled": False}},
        "evaluation": {"k_values": [1, 3], "dba_delta": 5},
    }
    if history_anchor:
        cfg["hist_beam"] = {
            "enabled": True,
            "group_size": 2,
            "history_anchor": {"enabled": True, "mode": "residual_delta", "num_delta_classes": 4},
        }
    return cfg


def _eval_loader():
    return [
        {
            "gps": torch.zeros(2, 2, 3),
            "mmwave": torch.zeros(2, 2, 64),
            "input_beam": torch.tensor([[0, 0], [2, 2]]),
            "target_beam": torch.tensor([[1], [1]]),
        }
    ]


def test_default_evaluation_does_not_pass_input_beam_to_model():
    result = run_evaluation_pass(
        _NoHistoryModel(),
        _eval_loader(),
        _eval_cfg(history_anchor=False),
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )

    assert result.metrics["uses_input_beam_as_model_input"] is False
    assert result.last_beams is None


def test_history_anchor_evaluation_passes_anchor_and_reports_residual_metrics(tmp_path: Path):
    result = run_evaluation_pass(
        _HistoryResidualModel(),
        _eval_loader(),
        _eval_cfg(history_anchor=True),
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    pred_path = write_hist_beam_predictions(
        tmp_path / "predictions.csv",
        result.outputs,
        result.labels,
        last_beams=result.last_beams,
        residual_logits=result.residual_logits,
        residual_labels=result.residual_labels,
        top_k=3,
        group_size=2,
    )

    assert result.metrics["uses_input_beam_as_model_input"] is True
    assert result.metrics["residual_accuracy"] == pytest.approx(1.0)
    assert "val_residual_accuracy" in result.metrics["available_metrics"]
    assert result.metrics["reconstructed_absolute_top1_avg"] == pytest.approx(1.0)
    with pred_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["last_beam"] == "0"
    assert rows[0]["true_residual"] == "1"
    assert rows[0]["pred_residual"] == "1"
    assert json.loads(rows[0]["topk_reconstructed_beam"])[0] == 1


def test_history_anchor_quick_validation_uses_three_main_modalities_and_keeps_anchor():
    cfg = load_config(ROOT / "configs/hist_beam/mmw_history_anchored_quick_validation.yaml")
    metadata = history_anchor_run_metadata(cfg)
    dataloader_cfg = cfg["data"]["dataloader"]
    cache_cfg = cfg["data"]["cache"]
    amp_cfg = cfg["training"]["amp"]

    assert resolve_enabled_modalities(cfg) == ("image", "gps", "lidar")
    assert cfg["data"]["dataset"]["enabled_modalities"] == ["image", "gps", "lidar"]
    assert cfg["model"]["modalities"] == ["image", "gps", "lidar"]
    assert cfg["model"]["student"]["modalities"] == ["image", "gps", "lidar"]
    assert cfg["training"]["epochs"] == 20
    assert cfg["hist_beam"]["adaptation"]["epochs"] == 1
    assert dataloader_cfg["train_batch_size"] == 32
    assert dataloader_cfg["test_batch_size"] == 32
    assert dataloader_cfg["train_num_workers"] == 4
    assert dataloader_cfg["test_num_workers"] == 4
    assert dataloader_cfg["train_persistent_workers"] is True
    assert dataloader_cfg["test_persistent_workers"] is True
    assert dataloader_cfg["train_prefetch_factor"] == 2
    assert dataloader_cfg["test_prefetch_factor"] == 2
    assert cache_cfg["policy"] == "read_only"
    assert cache_cfg["image"]["policy"] == "read_only"
    assert cache_cfg["lidar"]["policy"] == "read_only"
    assert cfg["training"]["transfer"]["non_blocking"] is True
    assert amp_cfg["enabled"] is True
    assert amp_cfg["dtype"] == "bfloat16"
    assert amp_cfg["grad_scaler"] is False
    assert cfg["hist_beam"]["history_anchor"]["enabled"] is True
    assert cfg["model"]["student"]["history_anchor"]["enabled"] is True
    assert metadata["history_anchor_enabled"] is True
    assert metadata["uses_input_beam_as_model_input"] is True


def test_hist_beam_residual_forward_loss_and_private_calibration_freeze():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=8,
        num_pred=1,
        group_size=4,
        variant="v4_adapter",
        history_anchor={"enabled": True, "mode": "residual_delta", "num_delta_classes": 8},
        num_heads=4,
        num_layers=1,
    )
    output = model(gps_batch=torch.randn(2, 2, 3), input_beam_batch=torch.tensor([[6, 6], [1, 1]]))
    loss = compute_hist_beam_loss(
        {"logits": output["logits"], **{k: v for k, v in output.items() if k != "logits"}},
        torch.tensor([[1], [3]]),
        cfg={"hist_beam": {"history_anchor": {"enabled": True, "mode": "residual_delta"}}},
    )
    strategy = apply_hist_beam_adaptation_strategy(model, "v4_adapter")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert output["residual_logits"].shape == (2, 1, 8)
    assert output["logits"].shape == (2, 1, 8)
    assert loss.total.isfinite()
    assert strategy["history_anchor_residual_freeze_strategy"] is True
    assert any(name.startswith("residual_head") for name in trainable)
    assert any(name.startswith("absolute_calibration_bias") for name in trainable)
    assert not any(name.startswith("transformer") for name in trainable)


def test_markov_and_source_prior_diagnostics():
    metrics = markov_delta_baseline_metrics(
        torch.tensor([0, 1, 2]),
        torch.tensor([[1], [2], [3]]),
        torch.tensor([3, 6]),
        torch.tensor([[4], [7]]),
        num_classes=8,
        k_values=[1, 3],
        smoothing=0.0,
    )
    collapse = source_prior_collapse_diagnostics(
        source_histogram=[0, 10, 0, 0],
        target_true_histogram=[0, 0, 9, 0],
        predicted_histogram=[0, 8, 0, 0],
    )

    assert metrics["markov_delta_baseline_available"] is True
    assert metrics["markov_delta_top1"] == pytest.approx(1.0)
    assert metrics["markov_delta_train_samples"] == 3
    assert collapse["source_prior_collapse"] is True


def test_unlabeled_history_anchor_adaptation_does_not_require_future_label():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=8,
        num_pred=1,
        group_size=4,
        variant="v4_adapter",
        history_anchor={"enabled": True, "mode": "residual_delta", "num_delta_classes": 8},
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v4_adapter")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    batch = {
        "gps": torch.randn(2, 2, 3),
        "input_beam": torch.tensor([[0, 0], [1, 1]]),
    }
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 2, "num_pred": 1}},
        "model": {
            "seq_length_student": 2,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {"modalities": ["gps"], "num_classes": 8},
        },
        "hist_beam": {"history_anchor": {"enabled": True, "mode": "residual_delta", "num_delta_classes": 8}},
        "training": {"transfer": {"non_blocking": False}},
    }

    result = adapt_hist_beam_target(
        model,
        labeled_dataloader=None,
        unlabeled_dataloader=torch.utils.data.DataLoader([batch], batch_size=None),
        cfg=cfg,
        device=torch.device("cpu"),
        optimizer=optimizer,
        epochs=1,
        label_budget=0,
    )

    assert result["used_input_beam_as_input"] is True
    assert result["used_target_beam_for_supervised_loss"] is False
    assert result["used_target_beam_for_training"] is False
