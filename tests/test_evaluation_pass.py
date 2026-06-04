from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.evaluation_pass import run_evaluation_pass  # noqa: E402
from kd_sensing.engine.validator import validate  # noqa: E402


class _MaskAwareFusionModel(torch.nn.Module):
    supports_force_modality_mask = True

    def forward(self, gps_batch=None, mmwave_batch=None, force_modality_mask=None, **kwargs):  # noqa: ANN001, ARG002
        batch_size = gps_batch.shape[0]
        logits = torch.tensor([[[4.0, 1.0, 0.0, -1.0]], [[0.0, 1.0, 4.0, -1.0]]], dtype=torch.float32)
        logits = logits[:batch_size].clone()
        if force_modality_mask is not None and not bool(force_modality_mask.to(torch.bool).all()):
            logits = torch.zeros_like(logits)
        return {"logits": logits}


class _SelectionHeadsModel(torch.nn.Module):
    def forward(self, coord_batch=None, **kwargs):  # noqa: ANN001, ARG002
        batch_size = coord_batch.shape[0]
        logits = torch.tensor([[[4.0, 1.0, 0.0, -1.0]], [[0.0, 1.0, 4.0, -1.0]]], dtype=torch.float32)
        los_logits = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)
        link_quality = torch.tensor([[-58.0], [-62.0]], dtype=torch.float32)
        return {
            "logits": logits[:batch_size].clone(),
            "los_logits": los_logits[:batch_size].clone(),
            "link_quality": link_quality[:batch_size].clone(),
        }


class _HardLabelOnlyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.targets: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype != torch.long or targets.ndim != 1:
            raise AssertionError(f"evaluation loss must receive flattened hard labels, got {targets.dtype} {targets.shape}")
        self.targets.append(targets.detach().cpu().clone())
        return torch.nn.functional.cross_entropy(inputs, targets)


def _cfg() -> dict:
    return {
        "experiment": {"task": "fusion", "objective": "beam"},
        "data": {"dataset": {}},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length": 2,
            "num_classes": 4,
            "primary": {"modalities": ["gps", "mmwave"]},
        },
        "training": {"transfer": {"non_blocking": False}, "amp": {"enabled": False}},
        "evaluation": {"k_values": [1, 2], "dba_delta": 5},
    }


def _dataloader():
    return [
        {
            "gps": torch.zeros(2, 2, 3),
            "mmwave": torch.zeros(2, 2, 64),
            "input_beam": torch.tensor([[0, 0], [1, 1]]),
            "target_beam": torch.tensor([[0], [2]]),
        }
    ]


def _dataloader_with_soft_targets():
    return [
        {
            "gps": torch.zeros(2, 2, 3),
            "mmwave": torch.zeros(2, 2, 64),
            "input_beam": torch.tensor([[0, 0], [1, 1]]),
            "target_beam": torch.tensor([[0], [2]]),
            "target_beam_distribution": torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "target_beam_distribution_mask": torch.tensor([[True], [True]]),
        }
    ]


def _selection_cfg(objective: str) -> dict:
    return {
        "experiment": {"task": "fusion", "objective": objective},
        "data": {"dataset": {"type": "raymobtime_s008"}},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length": 1,
            "num_classes": 4,
            "primary": {"modalities": ["coord"]},
        },
        "loss": {"objective": {"weights": {"beam_selection": 1.0, "los": 0.5, "link_quality": 0.25}}},
        "training": {"transfer": {"non_blocking": False}, "amp": {"enabled": False}},
        "evaluation": {"k_values": [1, 3, 5], "dba_delta": 5},
    }


def _selection_dataloader():
    return [
        {
            "coord": torch.zeros(2, 1, 3),
            "target_beam": torch.tensor([[0], [2]]),
            "los_label": torch.tensor([[1.0], [0.0]]),
            "link_quality": torch.tensor([[-60.0], [-61.0]]),
        }
    ]


def test_evaluation_pass_matches_validator_and_records_runtime_metadata():
    cfg = _cfg()
    model = _MaskAwareFusionModel()
    criterion = torch.nn.CrossEntropyLoss()

    direct = run_evaluation_pass(model, _dataloader(), cfg, criterion, torch.device("cpu")).metrics
    wrapped = validate(model, _dataloader(), cfg, criterion, torch.device("cpu"))

    assert wrapped["loss"] == pytest.approx(direct["loss"])
    assert wrapped["topk"]["1"] == pytest.approx(direct["topk"]["1"])
    assert wrapped["available_metrics"] == direct["available_metrics"]
    assert wrapped["objective"]["name"] == "beam"
    assert wrapped["objective"]["primary_metric"] == "val_adba"
    assert wrapped["enabled_modalities"] == ["gps", "mmwave"]


def test_validator_runs_configured_modality_subsets_through_shared_pass():
    cfg = _cfg()
    cfg["evaluation"]["modality_subsets"] = {"enabled": True, "subsets": ["all", "gps"]}
    model = _MaskAwareFusionModel()
    criterion = torch.nn.CrossEntropyLoss()

    metrics = validate(model, _dataloader(), cfg, criterion, torch.device("cpu"))

    subsets = metrics["modality_subsets"]
    assert set(subsets) == {"all", "gps"}
    assert subsets["all"]["topk"] == metrics["topk"]
    assert subsets["all"]["mask"] == [True, True]
    assert subsets["gps"]["modalities"] == ["gps"]
    assert subsets["gps"]["mask"] == [True, False]
    assert "topk" in subsets["gps"]


def test_evaluation_pass_uses_hard_labels_when_soft_targets_are_present():
    cfg = _cfg()
    model = _MaskAwareFusionModel()
    criterion = _HardLabelOnlyCriterion()

    result = run_evaluation_pass(model, _dataloader_with_soft_targets(), cfg, criterion, torch.device("cpu"))

    assert criterion.targets
    assert torch.equal(criterion.targets[0], torch.tensor([0, 2]))
    assert result.labels.dtype == torch.long
    assert result.labels.tolist() == [[0], [2]]
    assert result.metrics["topk"]["1"] == pytest.approx([1.0])


def test_evaluation_pass_force_mask_all_enabled_matches_normal_pass():
    cfg = _cfg()
    model = _MaskAwareFusionModel()
    criterion = torch.nn.CrossEntropyLoss()

    normal = run_evaluation_pass(model, _dataloader(), cfg, criterion, torch.device("cpu")).metrics
    masked = run_evaluation_pass(
        model,
        _dataloader(),
        cfg,
        criterion,
        torch.device("cpu"),
        force_modality_mask=torch.tensor([True, True]),
    ).metrics

    assert masked["loss"] == pytest.approx(normal["loss"])
    assert masked["topk"] == normal["topk"]
    assert masked["available_metrics"] == normal["available_metrics"]


@pytest.mark.parametrize(
    ("objective", "expected", "forbidden"),
    [
        (
            "current_beam_selection",
            {"val_beam_top1", "val_beam_top3", "val_beam_top5", "val_beam_dba"},
            {"val_adba", "val_los_f1", "val_link_mae"},
        ),
        (
            "current_los_classification",
            {"val_los_accuracy", "val_los_f1", "val_los_auc"},
            {"val_beam_top1", "val_beam_dba", "val_link_mae", "val_adba"},
        ),
        (
            "current_link_quality",
            {"val_link_mae", "val_link_rmse", "val_link_r2"},
            {"val_beam_top1", "val_beam_dba", "val_los_f1", "val_adba"},
        ),
        (
            "selection_multitask",
            {
                "val_beam_top1",
                "val_beam_top3",
                "val_beam_top5",
                "val_beam_dba",
                "val_los_accuracy",
                "val_los_f1",
                "val_los_auc",
                "val_link_mae",
                "val_link_rmse",
                "val_link_r2",
                "val_selection_multitask_loss",
            },
            {"val_adba"},
        ),
    ],
)
def test_raymobtime_selection_evaluation_promotes_only_current_objective_metrics(
    objective: str,
    expected: set[str],
    forbidden: set[str],
):
    metrics = run_evaluation_pass(
        _SelectionHeadsModel(),
        _selection_dataloader(),
        _selection_cfg(objective),
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    ).metrics

    assert expected <= set(metrics["available_metrics"])
    assert forbidden.isdisjoint(metrics)
    assert forbidden.isdisjoint(set(metrics["available_metrics"]))
    if objective in {"current_beam_selection", "selection_multitask"}:
        assert set(metrics["los_buckets"]) == {"LOS=0", "LOS=1"}
        assert metrics["los_buckets"]["LOS=0"]["sample_count"] == 1
        assert metrics["los_buckets"]["LOS=0"]["val_beam_top1"] == pytest.approx(1.0)
        assert metrics["los_buckets"]["LOS=1"]["sample_count"] == 1
        assert metrics["los_buckets"]["LOS=1"]["val_beam_top1"] == pytest.approx(1.0)
    else:
        assert "los_buckets" not in metrics
    if objective != "selection_multitask":
        assert "auxiliary" in metrics
