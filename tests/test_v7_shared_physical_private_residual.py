from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.datasets.mmw import MMWDataset  # noqa: E402
from kd_sensing.data.mmw.physical_labels import (  # noqa: E402
    beamspace_label_from_path_payload,
    beamspace_label_from_power_vector,
    resolve_physical_label_config,
)
from kd_sensing.engine.evaluation_pass import _v7_evaluation_metrics  # noqa: E402
from kd_sensing.engine.hist_beam_adaptation import adapt_hist_beam_target, apply_hist_beam_adaptation_strategy  # noqa: E402
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import write_hist_beam_predictions  # noqa: E402
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402


def test_beamspace_power_vector_and_path_fallback_helpers():
    cfg = resolve_physical_label_config({"enabled": True, "power_unit": "db", "temperature": 2.0, "smoothing_sigma": 1.0})
    vector = np.full(8, -20.0, dtype=np.float32)
    vector[3] = 0.0

    result = beamspace_label_from_power_vector(vector, num_classes=8, config=cfg)
    bad = beamspace_label_from_power_vector(
        np.zeros(8, dtype=np.float32),
        num_classes=8,
        config=resolve_physical_label_config({"enabled": True, "power_unit": "linear"}),
    )
    path = beamspace_label_from_path_payload(
        {
            "coeff": np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex64),
            "dep": np.array([0.0, np.pi / 2], dtype=np.float32),
        },
        num_classes=8,
        config=resolve_physical_label_config(
            {
                "enabled": True,
                "field_map": {"gain": "coeff", "aod_azimuth": "dep"},
                "smoothing_sigma": 0.5,
            }
        ),
    )

    assert result.available
    assert result.label is not None
    assert result.label.shape == (8,)
    assert result.label.sum() == pytest.approx(1.0)
    assert int(result.label.argmax()) == 3
    assert bad.diagnostics["unavailable_reason"] == "non_positive_power_vector"
    assert path.available
    assert path.diagnostics["aod_bin_fallback"] is True
    assert path.label is not None and path.label.sum() == pytest.approx(1.0)


def test_mmw_dataset_returns_beamspace_physical_label_and_cache(tmp_path: Path):
    root = tmp_path / "dataset" / "MMW" / "sunny"
    split_dir = root / "Prepared" / "Town10_skybridge_seed24" / "splits"
    beam_dir = root / "Prepared" / "Town10_skybridge_seed24" / "beam_power" / "cav_0"
    split_dir.mkdir(parents=True)
    beam_dir.mkdir(parents=True)
    _write_power(beam_dir / "000000.txt", peak=2)
    _write_power(beam_dir / "000001.txt", peak=5)
    csv_path = split_dir / "train.csv"
    _write_rows(
        csv_path,
        [
            {
                "beam1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000000.txt",
                "future_beam1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000001.txt",
                "mmwave1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000000.txt",
                "sample_id": "sample-1",
            }
        ],
    )
    dataset = MMWDataset(
        data_root=str(root),
        scene="Town10_skybridge_seed24",
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
        return_metadata=True,
        physical_label={"enabled": True, "cache_dir": str(tmp_path / "cache"), "source": "beam_power"},
    )

    sample = dataset[0]

    assert sample["beamspace_power_label"].shape == (1, 64)
    assert sample["beamspace_power_available"].tolist() == [True]
    assert sample["beamspace_power_source"] == ["beam_power_vector"]
    assert int(sample["beamspace_power_label"][0].argmax().item()) == 5
    assert dataset._physical_label_cache is not None
    assert Path(dataset._physical_label_cache["path"]).exists()
    assert sample["metadata"]["physical_label_stats"]["available_count"] == 1


@pytest.mark.parametrize("modalities", [("image", "gps", "lidar"), ("image", "radar", "gps")])
def test_v7_model_forward_shapes_for_sensor_combinations(modalities: tuple[str, ...]):
    model = HistBeamFusionNet(
        modalities=list(modalities),
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=2,
        group_size=4,
        variant="v7_shared_physical_private_residual",
        num_heads=4,
        num_layers=1,
        image_encoder={"type": "legacy_cnn"},
    )
    kwargs = {"image_batch": torch.randn(2, 2, 3, 224, 224), "gps_batch": torch.randn(2, 2, 3)}
    if "lidar" in modalities:
        kwargs["lidar_batch"] = torch.randn(2, 2, 3, 224, 224)
    if "radar" in modalities:
        kwargs["radar_batch"] = torch.randn(2, 2, 2, 128, 64)

    output = model(**kwargs)

    assert output["logits"].shape == (2, 2, 16)
    assert output["logits_shared"].shape == (2, 2, 16)
    assert output["delta_logits_private"].shape == (2, 2, 16)
    assert output["alpha"].shape == (2, 2, 1)
    assert output["pred_beamspace_power"].shape == (2, 2, 16)
    assert torch.allclose(output["logits"], output["logits_shared"] + output["alpha"] * output["delta_logits_private"])
    assert output["hist_beam"]["uses_input_beam_as_model_input"] is False


def test_v7_loss_valid_missing_warmup_zero_weights_and_small_batch_diff():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=8,
        num_pred=1,
        group_size=2,
        variant="v7_shared_physical_private_residual",
        num_heads=4,
        num_layers=1,
    )
    labels = torch.tensor([[0], [3]])
    target = torch.zeros(2, 1, 8)
    target[0, 0, 0] = 1.0
    target[1, 0, 3] = 1.0
    output = model(gps_batch=torch.randn(2, 1, 3))
    cfg = {"hist_beam": {"variant": "v7_shared_physical_private_residual"}, "training": {"shared_warmup_epochs": 1}}

    valid = compute_hist_beam_loss(output, labels, cfg=cfg, beamspace_power_labels=target, beamspace_power_mask=torch.ones(2, 1, dtype=torch.bool), current_epoch=1)
    balanced = compute_hist_beam_loss(
        model(gps_batch=torch.randn(4, 1, 3)),
        torch.tensor([[0], [0], [0], [3]]),
        cfg={
            "hist_beam": {
                "variant": "v7_shared_physical_private_residual",
                "class_balance": {"enabled": True, "source_training": True, "mode": "inverse_sqrt", "max_weight": 5.0},
            }
        },
        beamspace_power_labels=torch.nn.functional.one_hot(torch.tensor([[0], [0], [0], [3]]), num_classes=8).float(),
        beamspace_power_mask=torch.ones(4, 1, dtype=torch.bool),
        current_epoch=1,
    )
    missing = compute_hist_beam_loss(output, labels, cfg=cfg, current_epoch=1)
    warmup = compute_hist_beam_loss(output, labels, cfg=cfg, beamspace_power_labels=target, beamspace_power_mask=torch.ones(2, 1, dtype=torch.bool), current_epoch=0)
    zero_weights = compute_hist_beam_loss(
        output,
        labels,
        cfg={"hist_beam": {"variant": "v7_shared_physical_private_residual", "loss_weights": {"v7_shared_ce": 0, "v7_final_ce": 0, "v7_bsp_kl": 0, "v7_phys_kl": 0, "v7_res_l2": 0, "v7_gate_l1": 0, "v7_diff": 0}}},
        beamspace_power_labels=target,
        beamspace_power_mask=torch.ones(2, 1, dtype=torch.bool),
        current_epoch=1,
    )
    small = model(gps_batch=torch.randn(1, 1, 3))
    small_loss = compute_hist_beam_loss(
        small,
        torch.tensor([[1]]),
        cfg={"hist_beam": {"variant": "v7_shared_physical_private_residual"}},
        beamspace_power_labels=torch.nn.functional.one_hot(torch.tensor([[1]]), num_classes=8).float(),
        beamspace_power_mask=torch.ones(1, 1, dtype=torch.bool),
        current_epoch=1,
    )

    assert valid.total.isfinite()
    assert balanced.diagnostics["hist/v7/class_balance_enabled"] == 1.0
    assert balanced.diagnostics["hist/v7/class_balance_max_weight"] > balanced.diagnostics["hist/v7/class_balance_min_weight"]
    assert valid.diagnostics["hist/v7/bsp_available"] == 1.0
    assert missing.diagnostics["hist/v7/bsp_available"] == 0.0
    assert warmup.diagnostics["hist/v7/loss_final_ce"] == 0.0
    assert warmup.diagnostics["hist/v7/loss_res_l2"] == 0.0
    assert zero_weights.total.item() == pytest.approx(0.0)
    assert small_loss.diagnostics["hist/v7/loss_diff"] == 0.0


def test_v7_adaptation_freeze_and_target_physical_oracle_not_used():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=8,
        num_pred=1,
        group_size=2,
        variant="v7_shared_physical_private_residual",
        num_heads=4,
        num_layers=1,
    )
    strategy = apply_hist_beam_adaptation_strategy(model, "v7_private_residual")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    batch = {
        "gps": torch.randn(2, 1, 3),
        "target_beam": torch.tensor([[0], [1]]),
        "beamspace_power_label": torch.nn.functional.one_hot(torch.tensor([[0], [1]]), num_classes=8).float(),
        "beamspace_power_available": torch.ones(2, 1, dtype=torch.bool),
    }
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {"seq_length_student": 1, "num_pred": 1, "downsample_ratio": 1, "student": {"type": "hist_beam_fusion", "modalities": ["gps"], "num_classes": 8, "group_size": 2, "variant": "v7_shared_physical_private_residual"}},
        "hist_beam": {"variant": "v7_shared_physical_private_residual", "loss_weights": {"v7_res_l2": 0.01, "v7_gate_l1": 0.001}, "adaptation": {"entropy_weight": 0.0}},
    }

    result = adapt_hist_beam_target(
        model,
        torch.utils.data.DataLoader([batch], batch_size=None),
        None,
        cfg,
        torch.device("cpu"),
        optimizer,
        epochs=1,
        label_budget=2,
    )

    assert strategy["v7_private_residual_freeze_strategy"] is True
    assert any(name.startswith("private_adapter") for name in trainable)
    assert any(name.startswith("private_residual_head") for name in trainable)
    assert any(name.startswith("residual_gate") for name in trainable)
    assert not any(name.startswith("shared_beam_head") for name in trainable)
    assert result["used_target_beam_for_training"] is True
    assert result["used_target_physical_label_for_training"] is False
    assert result["uses_input_beam_as_model_input"] is False


def test_v7_evaluation_metrics_and_prediction_writer(tmp_path: Path):
    final = torch.tensor([[[3.0, 1.0, 0.0, -1.0]], [[0.0, 1.0, 4.0, -1.0]]])
    shared = torch.tensor([[[2.0, 1.0, 0.0, -1.0]], [[0.0, 4.0, 1.0, -1.0]]])
    labels = torch.tensor([[0], [2]])
    bsp = torch.nn.functional.one_hot(labels, num_classes=4).float()

    metrics = _v7_evaluation_metrics(
        final_logits=final,
        shared_logits=shared,
        labels=labels,
        beam_power=torch.tensor([[[1.0, 0.5, 0.1, 0.0]], [[0.2, 0.5, 1.0, 0.0]]]),
        alpha=torch.full((2, 1, 1), 0.25),
        delta_logits_private=final - shared,
        pred_beamspace_power=torch.softmax(final, dim=-1),
        beamspace_power_label=bsp,
        beamspace_power_mask=torch.ones(2, 1, dtype=torch.bool),
        k_values=[1, 3],
    )
    path = write_hist_beam_predictions(
        tmp_path / "predictions.csv",
        final,
        labels,
        metadata=[{"sample_id": "a", "scene_slug": "scene"}, {"sample_id": "b", "scene_slug": "scene"}],
        group_size=2,
        shared_logits=shared,
        variant_metadata={"variant": "v7_shared_physical_private_residual", "split": "target_test"},
    )

    assert metrics["final_top1"] == pytest.approx(1.0)
    assert metrics["shared_top1"] == pytest.approx(0.5)
    assert metrics["alpha_mean"] == pytest.approx(0.25)
    assert metrics["phys_kl_available"] is True
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["final_predicted_beam"] == "0"
    assert rows[0]["shared_predicted_beam"] == "0"
    assert json.loads(rows[0]["final_topk"])[:2] == [0, 1]
    assert json.loads(rows[0]["shared_topk"])[:2] == [0, 1]


def _write_power(path: Path, *, peak: int) -> None:
    vector = np.ones(64, dtype=np.float32)
    vector[int(peak)] = 10.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, vector)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
