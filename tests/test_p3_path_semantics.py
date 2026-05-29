from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd_sensing.data.datasets.mmw import MMWDataset  # noqa: E402
from kd_sensing.data.mmw.path_semantics import (  # noqa: E402
    PathFeatureBuilder,
    PathSemanticLabelBuilder,
    load_path_semantic_artifact,
)
from kd_sensing.engine.batch import assert_sensitive_fields_allowed  # noqa: E402
from kd_sensing.engine.hist_beam_adaptation import (  # noqa: E402
    adapt_hist_beam_target,
    apply_hist_beam_adaptation_strategy,
    path_prototype_assignment,
)
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss  # noqa: E402
from kd_sensing.engine.hist_beam_prototypes import generate_source_prototypes, validate_prototype_artifact  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import path_descriptor_regression_metrics, path_semantic_metrics  # noqa: E402
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402
from scripts.inspect_dataset import inspect_mmw_root  # noqa: E402


def test_path_feature_builder_field_map_descriptor_and_circular_boundary():
    payload = {
        "coeff": np.array([1.0 + 0.0j, 0.8 + 0.1j, 0.05 + 0.0j], dtype=np.complex64),
        "delay_ns": np.array([0.0, 0.1, 2.0], dtype=np.float32),
        "dep": np.array([-np.pi + 0.02, np.pi - 0.02, 0.0], dtype=np.float32),
        "arr": np.array([np.pi - 0.03, -np.pi + 0.03, 0.0], dtype=np.float32),
        "ok": np.array([1, 1, 0], dtype=np.int8),
    }
    builder = PathFeatureBuilder(
        field_map={
            "gain": "coeff",
            "delay": "delay_ns",
            "aod_azimuth": "dep",
            "aoa_azimuth": "arr",
            "mask": "ok",
        }
    )

    result = builder.build_descriptor(payload)

    assert result.available
    assert result.descriptor is not None
    assert result.descriptor.shape == (14,)
    assert result.diagnostics["field_map_used"]["a"] == "coeff"
    assert result.diagnostics["valid_path_count"] == 2
    aod_spread = result.descriptor[builder.descriptor_names.index("weighted_aod_angular_spread")]
    aoa_spread = result.descriptor[builder.descriptor_names.index("weighted_aoa_angular_spread")]
    assert aod_spread < 0.1
    assert aoa_spread < 0.1


def test_path_semantic_label_builder_kmeans_artifact_rule_radio_and_coarse(tmp_path: Path):
    descriptors = np.array(
        [
            [0.0, 0.9, 0.95, 0.1, 1.2, 0.0, 0.05, 0.0, 1.0, 0.0, 1.0, 0.02, 0.02, 0.85],
            [0.1, 0.85, 0.9, 0.2, 1.5, 0.0, 0.08, 0.0, 1.0, 0.0, 1.0, 0.03, 0.03, 0.75],
            [1.0, 0.3, 0.6, 0.8, 4.0, 1.0, 1.5, 1.0, 0.0, 1.0, 0.0, 1.2, 1.1, 0.12],
            [1.1, 0.25, 0.5, 0.9, 5.0, 1.2, 1.8, 1.0, 0.0, 1.0, 0.0, 1.4, 1.3, 0.09],
        ],
        dtype=np.float32,
    )
    artifact_path = tmp_path / "path_kmeans.json"
    builder = PathSemanticLabelBuilder.from_config({"mode": "kmeans_path_descriptor", "num_path_classes": 2, "seed": 3})

    artifact = builder.fit(descriptors, source_domain={"town": "Town10"}, artifact_path=artifact_path)
    loaded = PathSemanticLabelBuilder.from_config(
        {"mode": "kmeans_path_descriptor", "num_path_classes": 2, "artifact_path": str(artifact_path)}
    )
    label = loaded.derive(path_descriptor=descriptors[0])
    mismatch = loaded.derive(path_descriptor=np.zeros(3, dtype=np.float32))
    rule = PathSemanticLabelBuilder.from_config({"mode": "rule_path_pattern", "num_path_classes": 24}).derive(
        path_descriptor=descriptors[0]
    )
    radio_power = np.zeros(64, dtype=np.float32)
    radio_power[17] = 10.0
    radio = PathSemanticLabelBuilder.from_config({"mode": "radio_power", "group_size": 8}).derive(beam_power=radio_power)
    coarse = PathSemanticLabelBuilder.from_config({"mode": "coarse", "group_size": 8}).derive(beam_label=17)

    assert artifact["descriptor_dim"] == 14
    assert load_path_semantic_artifact(artifact_path)["fit_sample_count"] == 4
    assert label.available
    assert mismatch.diagnostics["unavailable_reason"].startswith("descriptor_dim_mismatch")
    assert rule.available
    assert radio.available
    assert coarse.label == 2


def test_inspect_dataset_reports_path_fields_and_field_map(tmp_path: Path):
    root = tmp_path / "MMW" / "sunny"
    channel_dir = root / "Channel_Data" / "Town10" / "Town10_skybridge_seed24" / "cav_0"
    sensor_dir = root / "Sensor_Data" / "Town10" / "Town10_skybridge_seed24" / "cav_0"
    beam_dir = root / "Prepared" / "Town10_skybridge_seed24" / "beam_power" / "cav_0"
    channel_dir.mkdir(parents=True)
    sensor_dir.mkdir(parents=True)
    beam_dir.mkdir(parents=True)
    (sensor_dir / "000000_camera0.png").write_bytes(b"not-an-image")
    (sensor_dir / "000000.pcd").write_text("VERSION .7\n", encoding="utf-8")
    np.savetxt(beam_dir / "000000.txt", np.ones(64, dtype=np.float32))
    np.savez(
        channel_dir / "000000_paths.npz",
        coeff=np.array([1 + 0j, 0.5 + 0j], dtype=np.complex64),
        delay=np.array([0.0, 0.2], dtype=np.float32),
        dep=np.array([0.0, 0.1], dtype=np.float32),
        arr=np.array([0.2, 0.3], dtype=np.float32),
    )

    report = inspect_mmw_root(
        root,
        field_map={"gain": "coeff", "delay": "delay", "aod_azimuth": "dep", "aoa_azimuth": "arr"},
    )

    assert report["dataset_family"] == "MMW"
    assert report["beam_power_file_count"] == 1
    assert report["channel_or_path_file_count"] == 1
    assert report["domains"][0]["path_available"] is True
    assert report["path_field_summaries"][0]["field_map_used"]["a"] == "coeff"


def test_mmw_dataset_returns_path_auxiliary_fields_without_input_modality(tmp_path: Path):
    root = tmp_path / "dataset" / "MMW" / "sunny"
    split_dir = root / "Prepared" / "Town10_skybridge_seed24" / "splits"
    beam_dir = root / "Prepared" / "Town10_skybridge_seed24" / "beam_power" / "cav_0"
    channel_dir = root / "Channel_Data" / "Town10" / "Town10_skybridge_seed24" / "cav_0"
    split_dir.mkdir(parents=True)
    beam_dir.mkdir(parents=True)
    channel_dir.mkdir(parents=True)
    _write_power(beam_dir / "000000.txt", peak=3)
    _write_power(beam_dir / "000001.txt", peak=7)
    np.savez(
        channel_dir / "000001_paths.npz",
        a=np.array([1 + 0j, 0.4 + 0j], dtype=np.complex64),
        tau=np.array([0.0, 0.2], dtype=np.float32),
        aod_azimuth=np.array([0.0, 0.1], dtype=np.float32),
        aoa_azimuth=np.array([0.2, 0.3], dtype=np.float32),
        valid_mask=np.array([1, 1], dtype=np.int8),
    )
    csv_path = split_dir / "train.csv"
    _write_rows(
        csv_path,
        [
            {
                "beam1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000000.txt",
                "future_beam1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000001.txt",
                "mmwave1": "Prepared/Town10_skybridge_seed24/beam_power/cav_0/000000.txt",
                "future_path1": "Channel_Data/Town10/Town10_skybridge_seed24/cav_0/000001_paths.npz",
                "condition": "sunny",
                "town": "Town10",
                "sensor_scenario": "Town10_skybridge_seed24",
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
        path_semantic={"enabled": True, "mode": "rule_path_pattern", "fallback_if_missing": "coarse", "return_path_params": True},
    )

    sample = dataset[0]

    assert sample["mmwave"].shape == (1, 64)
    assert sample["path_descriptor"].shape == (1, 14)
    assert sample["path_semantic_label"].shape == (1,)
    assert sample["path_valid"].tolist() == [True]
    assert sample["metadata"]["path_semantic_available"] is True
    assert "path" not in dataset.enabled_modalities
    assert "csi" not in sample
    assert sample["path_semantic_diagnostics"][0]["path_params"]["file_diagnostics"]["shape"] == ""

    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch["path_descriptor"].shape == (1, 1, 14)
    assert batch["path_valid"].tolist() == [[True]]


def test_v8_hist_beam_model_loss_prototype_adaptation_and_metrics(tmp_path: Path):
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v8_path_proto",
        path_semantic={"enabled": True, "num_path_classes": 4, "use_path_head": True, "use_path_condition_in_beam_head": True, "use_path_regression": True, "descriptor_dim": 14},
        num_path_classes=4,
        use_path_head=True,
        use_path_condition_in_beam_head=True,
        use_path_regression=True,
        path_descriptor_dim=14,
        path_embed_dim=6,
        num_heads=4,
        num_layers=1,
    )
    mu_path = torch.randn(4, 16)
    output = model(
        gps_batch=torch.randn(3, 1, 3),
        path_prototypes=mu_path,
        path_prototype_counts=torch.tensor([2, 1, 0, 3]),
    )
    labels = torch.tensor([[0], [4], [8]])
    path_labels = torch.tensor([[0], [1], [3]])
    descriptors = torch.randn(3, 1, 14)
    loss = compute_hist_beam_loss(
        output,
        labels,
        cfg={"hist_beam": {"group_size": 4, "loss_weights": {"lambda_path": 0.3, "lambda_path_reg": 0.05}}},
        path_semantic_labels=path_labels,
        path_descriptors=descriptors,
        path_descriptor_mask=torch.ones(3, 1, dtype=torch.bool),
    )
    batch = {
        "gps": torch.randn(4, 1, 3),
        "input_beam": torch.tensor([[0], [1], [2], [3]]),
        "target_beam": torch.tensor([[0], [1], [2], [3]]),
        "path_semantic_label": torch.tensor([[0], [1], [1], [3]]),
        "path_valid": torch.ones(4, 1, dtype=torch.bool),
        "path_descriptor": torch.randn(4, 1, 14),
    }
    cfg = _v8_cfg()
    source_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.train()
    source_optimizer.zero_grad()
    train_output = model(gps_batch=batch["gps"])
    train_loss = compute_hist_beam_loss(
        train_output,
        batch["target_beam"],
        cfg=cfg,
        path_semantic_labels=batch["path_semantic_label"],
        path_descriptors=batch["path_descriptor"],
        path_descriptor_mask=batch["path_valid"],
    )
    train_loss.total.backward()
    source_optimizer.step()
    model.eval()
    artifact = generate_source_prototypes(
        model,
        torch.utils.data.DataLoader([batch], batch_size=None),
        cfg,
        torch.device("cpu"),
        output_path=tmp_path / "path_proto.pt",
    )
    alpha, assignment_metrics = path_prototype_assignment(torch.randn(4, 16), artifact["mu_path_c"], counts=artifact["count_path"], tau=0.1)
    apply_hist_beam_adaptation_strategy(model, "v8_path_proto")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    adaptation = adapt_hist_beam_target(
        model,
        None,
        torch.utils.data.DataLoader([batch], batch_size=None),
        cfg,
        torch.device("cpu"),
        optimizer,
        prototypes=artifact,
        epochs=1,
        confidence_threshold=0.0,
        label_budget=0,
    )
    path_metrics = path_semantic_metrics(output["path_logits"], path_labels)
    missing_path_metrics = path_semantic_metrics(output["path_logits"], None)
    reg_metrics = path_descriptor_regression_metrics(output["path_attr_pred"], descriptors, torch.ones(3, 1, dtype=torch.bool))

    assert output["path_logits"].shape == (3, 1, 4)
    assert output["path_attr_pred"].shape == (3, 1, 14)
    assert output["hist_beam"]["path_condition_source"] == "source_path_prototype"
    assert loss.path_semantic.isfinite()
    assert loss.path_regression.isfinite()
    assert train_loss.total.isfinite()
    validate_prototype_artifact(artifact)
    assert (tmp_path / "path_proto.pt").exists()
    assert artifact["metadata"]["prototype_space"] == "shared_path_physical"
    assert artifact["count_path"][1].item() == 2
    assert alpha.shape == (4, 4)
    assert assignment_metrics["path_prototype_available_classes"] >= 3
    assert adaptation["proto_type"] == "path"
    assert adaptation["used_target_beam_for_training"] is False
    assert adaptation["used_target_path_label_for_training"] is False
    assert path_metrics["path_semantic_accuracy"] >= 0.0
    assert missing_path_metrics["path_metrics_unavailable_reason"] == "path_semantic_label_missing"
    assert reg_metrics["path_descriptor_metrics_available"] is True
    with pytest.raises(RuntimeError, match="Target sensitive field access blocked"):
        assert_sensitive_fields_allowed(
            batch,
            split="target_unlabeled",
            label_budget=0,
            fields=("path_semantic_label",),
        )


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


def _v8_cfg() -> dict[str, object]:
    return {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {
            "seq_length_student": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {
                "type": "hist_beam_fusion",
                "modalities": ["gps"],
                "num_classes": 16,
                "group_size": 4,
                "num_path_classes": 4,
                "d_model": 16,
                "variant": "v8_path_proto",
            },
        },
        "hist_beam": {
            "group_size": 4,
            "proto_type": "path",
            "path_semantic": {"enabled": True, "num_path_classes": 4, "mode": "rule_path_pattern"},
            "loss_weights": {"lambda_path": 0.3, "lambda_path_reg": 0.05},
            "prototype": {"proto_type": "path"},
            "adaptation": {
                "proto_type": "path",
                "entropy_weight": 0.01,
                "prototype_weight": 0.1,
                "proto_tau": 0.1,
                "target_proto_momentum": 0.9,
                "proto_warmup_epochs": 0,
                "allow_labeled_target_path_supervision": False,
            },
        },
    }
