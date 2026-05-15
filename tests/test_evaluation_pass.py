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


def _cfg() -> dict:
    return {
        "experiment": {"task": "fusion", "objective": "beam"},
        "data": {"dataset": {}},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length_student": 2,
            "num_classes": 4,
            "student": {"modalities": ["gps", "mmwave"]},
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
