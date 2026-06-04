from __future__ import annotations

import csv
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.raymobtime_s008 import RaymobtimeS008SnapshotDataset  # noqa: E402
from kd_sensing.data.layouts import raymobtime_s008_root  # noqa: E402
from kd_sensing.engine.batch import prepare_fusion_inputs  # noqa: E402
from kd_sensing.engine.batch_step import raymobtime_gate_scalar_diagnostics  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output  # noqa: E402
from kd_sensing.engine.prediction_objectives import (  # noqa: E402
    PredictionTargets,
    compute_prediction_loss,
    normalize_objective_metric,
    objective_runtime_metadata,
    objective_spec,
    selection_multitask_loss_weights,
)
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.evaluation.metrics import calculate_current_beam_dba, calculate_link_metrics, calculate_los_metrics  # noqa: E402
from kd_sensing.modalities import (  # noqa: E402
    MODALITY_ORDER,
    batch_input_keys_for_modalities,
    dataset_flags_for_modalities,
    modality_spec,
    normalize_modalities,
)
from kd_sensing.models.raymobtime_s008 import (  # noqa: E402
    RaymobtimeLidar3DCNNEncoder,
    SimpleConcatMultiTaskSelection,
    TaskAwareGatedMultiTaskSelection,
)
from kd_sensing.preprocessing.raymobtime_s008 import (  # noqa: E402
    audit_s008_files,
    build_s008_cache,
    normalize_beam_labels,
)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(1, 1)
        self.bn1 = nn.Linear(1, 1)
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        self.layer3 = nn.Linear(1, 1)
        self.layer4 = nn.Linear(1, 1)

    def forward(self, x):
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 512)


@pytest.fixture()
def tiny_resnet(monkeypatch):
    import kd_sensing.models.image_encoders as image_encoders

    monkeypatch.setattr(
        image_encoders,
        "_build_resnet18_backbone",
        lambda *, pretrained, weights: (_TinyBackbone(), 512),
    )


def test_raymobtime_layout_and_modality_contracts():
    assert raymobtime_s008_root() == "dataset/Raymobtime/s008"
    assert MODALITY_ORDER[:6] == ("image", "radar", "gps", "lidar", "mmwave", "csi")
    assert MODALITY_ORDER[-2:] == ("coord", "ray")
    assert normalize_modalities(["ray", "coord", "lidar", "image"]) == ("image", "lidar", "coord", "ray")
    assert modality_spec("coord").fusion_input_key == "coord_batch"
    assert modality_spec("ray").dataset_flag == "use_ray"
    assert dataset_flags_for_modalities(["coord", "ray"])["use_coord"] is True
    assert batch_input_keys_for_modalities(["coord", "ray"]) == {
        "coord": "coord_batch",
        "ray": "ray_batch",
    }
    with pytest.raises(ValueError, match="duplicates"):
        normalize_modalities(["coord", "coord", "image"])
    with pytest.raises(ValueError, match="ray_path"):
        normalize_modalities(["coord", "ray_path"])


def test_beam_label_normalization_supports_three_raymobtime_formats():
    labels = normalize_beam_labels(np.asarray([0, 7, 8]), num_tx_beams=32, num_rx_beams=8)
    assert labels["beam_label"].tolist() == [0, 7, 8]
    assert labels["beam_tx"].tolist() == [0, 0, 1]
    assert labels["beam_rx"].tolist() == [0, 7, 0]

    pairs = normalize_beam_labels(np.asarray([[1, 2], [3, 4]]), num_tx_beams=4, num_rx_beams=8)
    assert pairs["beam_label"].tolist() == [10, 28]
    assert pairs["num_beam_classes"] == 32

    scores = np.zeros((2, 3, 4), dtype=np.float32)
    scores[0, 1, 3] = 9.0
    scores[1, 2, 0] = 5.0
    matrices = normalize_beam_labels(scores)
    assert matrices["beam_label"].tolist() == [7, 8]
    assert matrices["beam_tx"].tolist() == [1, 2]
    assert matrices["beam_rx"].tolist() == [3, 0]

    reversed_axes = np.zeros((1, 2, 3), dtype=np.complex128)
    reversed_axes[0, 1, 2] = 2.0 + 0.0j
    reversed = normalize_beam_labels(reversed_axes, num_tx_beams=3, num_rx_beams=2)
    assert reversed["beam_label"].tolist() == [5]
    assert reversed["beam_tx"].tolist() == [2]
    assert reversed["beam_rx"].tolist() == [1]


def test_preprocess_cache_and_dataset_contract(tmp_path: Path):
    root = _write_raymobtime_fixture(tmp_path, beam=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64))
    audit = audit_s008_files(data_root=root, output_dir=tmp_path / "audit")
    result = build_s008_cache(
        data_root=root,
        cache_dir=tmp_path / "cache",
        split_seed=1,
        split_ratios=(0.5, 0.25, 0.25),
        num_tx_beams=4,
        num_rx_beams=2,
    )

    assert Path(audit["audit_summary"]).exists()
    assert Path(result["cache_train"]).exists()
    assert Path(result["cache_metadata"]).exists()

    dataset = RaymobtimeS008SnapshotDataset(
        data_root=str(root),
        cache_dir=str(tmp_path / "cache"),
        split="train",
        modalities=["coord", "image", "lidar", "ray"],
    )
    dataset.cache.close()
    dataset.labels.close()
    sample = dataset[0]

    assert set(sample) >= {"coord", "image", "lidar", "ray", "target_beam", "los_label", "link_quality", "meta"}
    assert not {"history", "future", "horizon"} & set(sample)
    assert sample["coord"].shape == (1, 3)
    assert sample["image"].shape == (1, 3, 224, 224)
    assert sample["lidar"].shape == (1, 1, 4, 4, 4)
    assert sample["ray"].shape == (1, 14)
    assert sample["target_beam"].shape == (1,)
    assert {"sample_id", "EpisodeID", "SceneID", "VehicleArrayID", "valid_index", "split"} <= set(sample["meta"])
    assert dataset.raymobtime_metadata()["task_semantics"] == "current_snapshot_beam_selection"

    deferred_resize = RaymobtimeS008SnapshotDataset(
        data_root=str(root),
        cache_dir=str(tmp_path / "cache"),
        split="train",
        modalities=["image"],
        image_resize_in_dataset=False,
    )
    deferred_resize.cache.close()
    deferred_resize.labels.close()
    assert deferred_resize[0]["image"].shape == (1, 3, 4, 4)


def test_preprocess_reads_hdf5_ray_paths_and_quality_summary(tmp_path: Path):
    root = _write_raymobtime_fixture(tmp_path, beam=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64))
    _write_all_episode_hdf5_ray_zip(root)

    result = build_s008_cache(
        data_root=root,
        cache_dir=tmp_path / "cache",
        split_seed=1,
        split_ratios=(0.5, 0.25, 0.25),
        num_tx_beams=4,
        num_rx_beams=2,
    )

    train_cache = np.load(tmp_path / "cache" / "cache_train.npz", allow_pickle=True)
    report = json.loads(Path(result["unmatched_report"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(result["cache_metadata"]).read_text(encoding="utf-8"))
    assert not np.allclose(train_cache["link_quality"], -120.0)
    assert train_cache["ray"].shape[1] == 14
    assert report["summary"]["matched_ray_paths"] > 0
    assert report["summary"]["fallback_link_targets"] < report["summary"]["num_samples"]
    assert metadata["ray_quality"]["status"] == "ok"
    assert metadata["hdf5_schema"]["dataset"] == "allEpisodeData"
    assert "raymobtime_path_flag" not in train_cache.files


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "all samples are missing ray paths"),
        ("fallback", "all link_quality targets equal fallback"),
        ("constant", "train split link_quality standard deviation is 0"),
    ],
)
def test_preprocess_cache_quality_gate_rejects_bad_link_targets(tmp_path: Path, mode: str, message: str):
    root = _write_raymobtime_fixture(tmp_path, beam=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64))
    if mode == "missing":
        with zipfile.ZipFile(root / "raw_data" / "ray_tracing_data_s008_carrier60GHz.zip", "w"):
            pass
    elif mode == "fallback":
        _write_csv_ray_zip(root, power_by_episode=lambda _episode: -120.0)
    else:
        _write_csv_ray_zip(root, power_by_episode=lambda _episode: -55.0)

    with pytest.raises(ValueError, match=message):
        build_s008_cache(
            data_root=root,
            cache_dir=tmp_path / "cache",
            split_seed=1,
            split_ratios=(0.5, 0.25, 0.25),
            num_tx_beams=4,
            num_rx_beams=2,
        )


def test_preprocess_cache_aligns_official_split_npz_files(tmp_path: Path):
    root = _write_raymobtime_fixture(tmp_path, beam=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64))
    _rewrite_fixture_los_as_strings(root)
    _write_official_split_npz_files(root, train_len=4, val_len=2)

    build_s008_cache(
        data_root=root,
        cache_dir=tmp_path / "cache",
        split_seed=999,
        split_ratios=(0.8, 0.1, 0.1),
        num_tx_beams=4,
        num_rx_beams=8,
    )

    train_index = pd.read_csv(tmp_path / "cache" / "index_train.csv")
    val_index = pd.read_csv(tmp_path / "cache" / "index_val.csv")
    test_index = pd.read_csv(tmp_path / "cache" / "index_test.csv")
    assert len(train_index) == 4
    assert len(val_index) == 2
    assert len(test_index) == 2
    assert test_index["source_split"].tolist() == ["validation", "validation"]
    assert test_index["source_split_index"].tolist() == [0, 1]

    labels_train = np.load(tmp_path / "cache" / "labels_train.npz", allow_pickle=True)
    labels_val = np.load(tmp_path / "cache" / "labels_val.npz", allow_pickle=True)
    labels_test = np.load(tmp_path / "cache" / "labels_test.npz", allow_pickle=True)
    assert labels_train["beam_label"].tolist() == [10, 11, 12, 13]
    assert labels_val["beam_label"].tolist() == [20, 21]
    assert labels_test["beam_label"].tolist() == [20, 21]
    assert labels_test["los_label"].tolist() == [1.0, 1.0]

    cache_train = np.load(tmp_path / "cache" / "cache_train.npz", allow_pickle=True)
    cache_test = np.load(tmp_path / "cache" / "cache_test.npz", allow_pickle=True)
    assert cache_train["image"][0].mean() == pytest.approx(101.0)
    assert cache_test["image"][0].mean() == pytest.approx(201.0)


def test_runtime_batch_helpers_prepare_coord_and_ray_inputs():
    batch = {
        "coord": torch.randn(2, 1, 3),
        "ray": torch.randn(2, 1, 14),
    }
    prepared = prepare_fusion_inputs(
        batch,
        seq_length=1,
        num_pred=1,
        device=torch.device("cpu"),
        modalities=["ray", "coord"],
    )

    assert prepared["coord_batch"].shape == (2, 1, 3)
    assert prepared["ray_batch"].shape == (2, 1, 14)
    image_prepared = prepare_fusion_inputs(
        {"image": torch.randn(2, 1, 3, 4, 4)},
        seq_length=1,
        num_pred=1,
        device=torch.device("cpu"),
        modalities=["image"],
        image_profile="rgb_imagenet",
    )
    assert image_prepared["image_batch"].shape == (2, 1, 3, 224, 224)
    lidar_prepared = prepare_fusion_inputs(
        {"lidar": torch.randn(2, 1, 1, 4, 4, 4)},
        seq_length=1,
        num_pred=1,
        device=torch.device("cpu"),
        modalities=["lidar"],
    )
    assert lidar_prepared["lidar_batch"].shape == (2, 1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="ray"):
        prepare_fusion_inputs(
            {"coord": torch.randn(2, 1, 3)},
            seq_length=1,
            num_pred=1,
            device=torch.device("cpu"),
            modalities=["coord", "ray"],
        )


def test_selection_objective_loss_and_metrics_contract():
    logits = torch.tensor([[[2.0, 0.1, 0.0]], [[0.0, 0.2, 2.5]]])
    output = adapt_model_output(
        {
            "logits": logits,
            "los_logits": torch.tensor([[0.5], [-0.5]]),
            "link_quality": torch.tensor([[1.0], [3.0]]),
        }
    )
    labels = torch.tensor([[0], [2]])
    targets = PredictionTargets(
        labels=labels,
        los_label=torch.tensor([[1.0], [0.0]]),
        link_quality=torch.tensor([[1.5], [2.0]]),
    )
    beam = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
    cfg = {
        "experiment": {"objective": "selection_multitask"},
        "loss": {"objective": {"weights": {"beam_selection": 1.0, "los": 0.5, "link_quality": 0.25}}},
    }
    bundle = compute_prediction_loss(output, targets, cfg, reference=logits, beam_total_loss=beam, beam_task_loss=beam)

    assert torch.isfinite(bundle.total)
    assert bundle.diagnostics["loss/selection_multitask_total"] == pytest.approx(float(bundle.total.detach()))
    assert selection_multitask_loss_weights(cfg) == {"beam_selection": 1.0, "los": 0.5, "link_quality": 0.25}
    assert objective_spec("current_beam_selection").default_metric == "val_beam_top1"
    assert normalize_objective_metric("beam_top1", objective="current_beam_selection") == "val_beam_top1"
    assert normalize_objective_metric("beam_dba", objective="current_beam_selection") == "val_beam_dba"
    assert normalize_objective_metric("beam/val_dba_current", objective="current_beam_selection") == "val_beam_dba"
    assert calculate_current_beam_dba(logits, labels, delta=5) == pytest.approx(1.0)
    runtime = objective_runtime_metadata(cfg)
    assert runtime["enabled_heads"] == ["beam_selection", "los", "link_quality"]

    los_only = compute_prediction_loss(
        output,
        targets,
        {"experiment": {"objective": "current_los_classification"}, "loss": {"objective": {"los": {"pos_weight": None}}}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    expected_los = F.binary_cross_entropy_with_logits(output.diagnostics["los_logits"], targets.los_label)
    assert torch.allclose(los_only.total, expected_los)
    assert objective_spec("current_los_classification").default_metric == "val_los_f1"
    assert normalize_objective_metric("los", objective="current_los_classification") == "val_los_f1"

    link_only = compute_prediction_loss(
        output,
        targets,
        {"experiment": {"objective": "current_link_quality"}, "loss": {"objective": {"link_quality": {"beta": 1.0}}}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    expected_link = F.smooth_l1_loss(output.diagnostics["link_quality"], targets.link_quality, beta=1.0)
    assert torch.allclose(link_only.total, expected_link)
    assert objective_spec("current_link_quality").default_metric == "val_link_mae"
    assert normalize_objective_metric("link_quality", objective="current_link_quality") == "val_link_mae"

    los = calculate_los_metrics(torch.tensor([[0.1], [0.2]]), torch.tensor([[1.0], [1.0]]))
    assert los["los_auc"] is None
    assert los["los_auc_available"] is False
    link = calculate_link_metrics(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.5], [1.5]]))
    assert link["link_mae"] == pytest.approx(0.5)


def test_raymobtime_snapshot_models_forward_gates_and_reject_temporal_core(tiny_resnet):
    lidar_encoder = RaymobtimeLidar3DCNNEncoder(
        output_dim=8,
        lidar_channels=1,
        stem_channels=4,
        block_channels=[4, 8],
        hidden_size=16,
    )
    lidar_features = lidar_encoder(torch.randn(2, 1, 1, 4, 4, 4))
    assert lidar_features.shape == (2, 1, 8)
    assert any(isinstance(module, nn.Conv3d) for module in lidar_encoder.modules())
    assert any(isinstance(module, nn.AdaptiveAvgPool3d) for module in lidar_encoder.modules())
    assert any(isinstance(module, nn.AdaptiveMaxPool3d) for module in lidar_encoder.modules())

    simple = SimpleConcatMultiTaskSelection(
        modalities=["coord", "ray"],
        feature_size=8,
        hidden_size=16,
        num_classes=6,
        coord_input_size=3,
        ray_input_size=14,
    )
    output = simple(coord_batch=torch.randn(2, 1, 3), ray_batch=torch.randn(2, 1, 14))
    assert output["logits"].shape == (2, 1, 6)
    assert output["los_logits"].shape == (2, 1)
    assert output["link_quality"].shape == (2, 1)

    gated = TaskAwareGatedMultiTaskSelection(
        modalities=["coord", "image", "lidar", "ray"],
        feature_size=8,
        hidden_size=16,
        num_classes=6,
        coord_input_size=3,
        ray_input_size=14,
        lidar_channels=1,
        encoders={
            "image": {"type": "resnet18_imagenet_rgb", "output_dim": 8, "pretrained": False},
            "lidar": {
                "type": "raymobtime_lidar_3d_cnn",
                "output_dim": 8,
                "lidar_channels": 1,
                "stem_channels": 4,
                "block_channels": [4, 8],
                "hidden_size": 16,
            },
        },
    )
    gated_output = gated(
        coord_batch=torch.randn(2, 1, 3),
        image_batch=torch.randn(2, 1, 3, 224, 224),
        lidar_batch=torch.randn(2, 1, 1, 4, 4, 4),
        ray_batch=torch.randn(2, 1, 14),
    )
    assert gated_output["logits"].shape == (2, 1, 6)
    assert gated_output["gates"]["beam_selection"].shape == (2, 4)
    assert torch.allclose(gated_output["gates"]["los"].sum(dim=1), torch.ones(2), atol=1e-6)
    masked_output = gated(
        coord_batch=torch.randn(2, 1, 3),
        image_batch=torch.randn(2, 1, 3, 224, 224),
        lidar_batch=torch.randn(2, 1, 1, 4, 4, 4),
        ray_batch=torch.randn(2, 1, 14),
        force_modality_mask=torch.tensor([False, True, True, False]),
    )
    assert gated.supports_force_modality_mask is True
    assert masked_output["effective_modality_mask"].tolist() == [[False, True, True, False]] * 2
    assert torch.allclose(masked_output["gates"]["beam_selection"][:, [0, 3]], torch.zeros(2, 2), atol=1e-6)
    gate_scalars = raymobtime_gate_scalar_diagnostics(adapt_model_output(masked_output).diagnostics)
    assert "raymobtime/gate/beam_selection/coord" in gate_scalars
    assert gate_scalars["raymobtime/gate/beam_selection/image"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="no available Raymobtime modality"):
        gated(
            coord_batch=torch.randn(2, 1, 3),
            image_batch=torch.randn(2, 1, 3, 224, 224),
            lidar_batch=torch.randn(2, 1, 1, 4, 4, 4),
            ray_batch=torch.randn(2, 1, 14),
            force_modality_mask=torch.tensor([False, False, False, False]),
        )
    assert not any(isinstance(module, (nn.GRU, nn.RNN, nn.LSTM)) for module in gated.modules())
    with pytest.raises(ValueError, match="current snapshot"):
        simple(coord_batch=torch.randn(2, 2, 3), ray_batch=torch.randn(2, 1, 14))


def test_raymobtime_config_loads_and_rejects_future_terms():
    cfg = load_config(ROOT / "configs/raymobtime/s008_multitask_selection.yaml")

    assert cfg["data"]["dataset"]["type"] == "raymobtime_s008"
    assert cfg["experiment"]["task_semantics"] == "current_snapshot_beam_selection"
    assert cfg["training"]["early_stopping_metric"] == "val_selection_multitask_loss"
    assert "ray" in cfg["model"]["primary"]["modalities"]
    assert cfg["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert cfg["model"]["primary"]["encoders"]["lidar"]["type"] == "raymobtime_lidar_3d_cnn"
    assert cfg["data"]["dataset"]["image_resize_in_dataset"] is False
    assert cfg["training"]["cpu_threads"] == {"intra_op": 2, "inter_op": 1}

    los_cfg = load_config(
        ROOT / "configs/raymobtime/s008_multitask_selection.yaml",
        ["experiment.objective=current_los_classification", "model.modalities=[coord,image,lidar]"],
    )
    assert los_cfg["training"]["early_stopping_metric"] == "val_los_f1"
    assert los_cfg["model"]["primary"]["modalities"] == ["image", "lidar", "coord"]

    link_cfg = load_config(
        ROOT / "configs/raymobtime/s008_multitask_selection.yaml",
        ["experiment.objective=current_link_quality", "model.modalities=[coord,image,lidar]"],
    )
    assert link_cfg["training"]["early_stopping_metric"] == "val_link_mae"
    assert link_cfg["training"]["early_stopping_mode"] == "min"

    with pytest.raises(ValueError, match="current snapshot beam selection"):
        load_config(
            ROOT / "configs/raymobtime/s008_multitask_selection.yaml",
            ["data.dataset.beam_prediction_horizon=3"],
        )


def test_raymobtime_minimal_train_smoke(tmp_path: Path):
    root = _write_raymobtime_fixture(tmp_path, beam=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64))
    build_s008_cache(
        data_root=root,
        cache_dir=tmp_path / "cache",
        split_seed=1,
        split_ratios=(0.5, 0.25, 0.25),
        num_tx_beams=4,
        num_rx_beams=2,
    )
    cfg = load_config(ROOT / "configs/raymobtime/s008_smoke_selection.yaml")
    cfg["data"]["dataset"]["data_root"] = str(root)
    cfg["data"]["dataset"]["cache_dir"] = str(tmp_path / "cache")
    cfg["output"]["dir"] = str(tmp_path / "outputs")
    cfg["output"]["run_name"] = "smoke"
    cfg["model"]["num_classes"] = 8
    cfg["model"]["primary"]["num_classes"] = 8
    cfg["training"]["epochs"] = 1
    cfg["training"]["use_early_stopping"] = False

    result = train(cfg)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "checkpoints" / "last.pth").exists()
    assert result["prediction_objective"]["name"] == "selection_multitask"
    assert result["history"]["val_selection_multitask_loss"][-1] is not None
    assert result["history"]["val_beam_dba"][-1] is not None
    assert "val_adba" not in result["history"]
    assert "val_adba" not in result["epoch_logs"][0]
    outputs = np.load(run_dir / "training_outputs.npz")
    assert "val_beam_dba" in outputs.files
    assert "val_adba" not in outputs.files


def _write_raymobtime_fixture(root: Path, *, beam: np.ndarray) -> Path:
    data_root = root / "Raymobtime_s008_fixture"
    for rel in (
        "baseline_data/beam_output",
        "baseline_data/coord_input",
        "baseline_data/lidar_input",
        "baseline_data/image_v2_input",
        "raw_data",
    ):
        (data_root / rel).mkdir(parents=True, exist_ok=True)
    rows = []
    valid_idx = 0
    for idx in range(8):
        val = "I" if idx in {2, 6} else "V"
        los = idx % 2
        rows.append(
            {
                "Val": val,
                "EpisodeID": idx,
                "SceneID": 0,
                "VehicleArrayID": idx + 10,
                "VehicleName": f"vehicle_{idx}",
                "x": float(idx),
                "y": float(idx + 1),
                "z": float(idx + 2),
                "LOS": los,
            }
        )
        if val == "V":
            valid_idx += 1
    with (data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(data_root / "baseline_data" / "beam_output" / "beam_output.npz", beam=beam)
    valid_count = int((np.asarray([row["Val"] for row in rows]) == "V").sum())
    image = np.arange(valid_count * 3 * 4 * 4, dtype=np.float32).reshape(valid_count, 3, 4, 4) / 100.0
    lidar = np.arange(valid_count * 1 * 4 * 4 * 4, dtype=np.float32).reshape(valid_count, 1, 4, 4, 4) / 50.0
    np.savez(data_root / "baseline_data" / "image_v2_input" / "image.npz", image=image)
    np.savez(data_root / "baseline_data" / "lidar_input" / "lidar.npz", lidar=lidar)
    ray_rows = []
    for row in rows:
        if row["Val"] != "V":
            continue
        sample_id = f"e{row['EpisodeID']}_s{row['SceneID']}_v{row['VehicleArrayID']}"
        for ray_idx in range(2):
            ray_rows.append(
                {
                    "sample_id": sample_id,
                    "power_dbm": -50.0 + float(row["EpisodeID"]) - ray_idx,
                    "toa": 1.0 + ray_idx,
                    "elev_aod": 0.1,
                    "az_aod": 0.2,
                    "elev_aoa": 0.3,
                    "az_aoa": 0.4,
                    "phase": 0.5,
                }
            )
    ray_csv = root / "ray_paths.csv"
    with ray_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ray_rows[0]))
        writer.writeheader()
        writer.writerows(ray_rows)
    with zipfile.ZipFile(data_root / "raw_data" / "ray_tracing_data_s008_carrier60GHz.zip", "w") as archive:
        archive.write(ray_csv, "ray_paths.csv")
    return data_root


def _write_all_episode_hdf5_ray_zip(data_root: Path) -> None:
    h5py = pytest.importorskip("h5py")
    csv_path = data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv"
    frame = pd.read_csv(csv_path)
    valid_episodes = sorted(frame.loc[frame["Val"].astype(str).str.upper() == "V", "EpisodeID"].unique())
    zip_path = data_root / "raw_data" / "ray_tracing_data_s008_carrier60GHz.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for episode in valid_episodes:
            episode_rows = frame.loc[frame["EpisodeID"] == episode]
            vehicle_count = int(frame["VehicleArrayID"].max()) + 1
            all_episode = np.full((1, vehicle_count, 3, 9), np.nan, dtype=np.float32)
            for row in episode_rows.itertuples(index=False):
                vehicle = int(row.VehicleArrayID)
                for path_idx in range(3):
                    all_episode[0, vehicle, path_idx, 0] = -60.0 + float(episode) - float(path_idx)
                    all_episode[0, vehicle, path_idx, 1] = 1e-7 * float(path_idx + 1)
                    all_episode[0, vehicle, path_idx, 2] = 10.0 + path_idx
                    all_episode[0, vehicle, path_idx, 3] = 20.0 + path_idx
                    all_episode[0, vehicle, path_idx, 4] = 30.0 + path_idx
                    all_episode[0, vehicle, path_idx, 5] = 40.0 + path_idx
                    all_episode[0, vehicle, path_idx, 8] = 180.0 + path_idx
            buffer = BytesIO()
            with h5py.File(buffer, "w") as h5:
                h5.create_dataset("allEpisodeData", data=all_episode)
            archive.writestr(f"ray_tracing_data_s008_carrier60GHz/fixture_e{int(episode)}.hdf5", buffer.getvalue())


def _write_csv_ray_zip(data_root: Path, *, power_by_episode) -> None:
    csv_path = data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv"
    frame = pd.read_csv(csv_path)
    ray_rows = []
    for row in frame.itertuples(index=False):
        if str(row.Val).upper() != "V":
            continue
        sample_id = f"e{int(row.EpisodeID)}_s{int(row.SceneID)}_v{int(row.VehicleArrayID)}"
        ray_rows.append(
            {
                "sample_id": sample_id,
                "power_dbm": float(power_by_episode(int(row.EpisodeID))),
                "toa": 1.0,
                "phase": 0.0,
            }
        )
    ray_csv = data_root.parent / "bad_ray_paths.csv"
    with ray_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ray_rows[0]))
        writer.writeheader()
        writer.writerows(ray_rows)
    with zipfile.ZipFile(data_root / "raw_data" / "ray_tracing_data_s008_carrier60GHz.zip", "w") as archive:
        archive.write(ray_csv, "ray_paths.csv")


def _rewrite_fixture_los_as_strings(data_root: Path) -> None:
    csv_path = data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv"
    frame = pd.read_csv(csv_path)
    frame["LOS"] = [f"LOS={int(value)}" for value in frame["LOS"]]
    frame.to_csv(csv_path, index=False)


def _write_official_split_npz_files(data_root: Path, *, train_len: int, val_len: int) -> None:
    csv_path = data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv"
    frame = pd.read_csv(csv_path)
    valid = frame.loc[frame["Val"].astype(str).str.upper() == "V"].reset_index(drop=True)
    coords = valid.loc[:, ["x", "y"]].to_numpy(dtype=np.float64)
    np.savez(data_root / "baseline_data" / "coord_input" / "coord_train.npz", coordinates=coords[:train_len])
    np.savez(
        data_root / "baseline_data" / "coord_input" / "coord_validation.npz",
        coordinates=coords[train_len : train_len + val_len],
    )

    np.savez(
        data_root / "baseline_data" / "beam_output" / "beams_output_train.npz",
        output_classification=np.asarray([10, 11, 12, 13], dtype=np.int64),
    )
    np.savez(
        data_root / "baseline_data" / "beam_output" / "beams_output_validation.npz",
        output_classification=np.asarray([20, 21], dtype=np.int64),
    )

    image_train = np.full((train_len, 3, 4, 4), 101.0, dtype=np.float32)
    image_val = np.full((val_len, 3, 4, 4), 201.0, dtype=np.float32)
    lidar_train = np.full((train_len, 1, 4, 4, 4), 102.0, dtype=np.float32)
    lidar_val = np.full((val_len, 1, 4, 4, 4), 202.0, dtype=np.float32)
    np.savez(data_root / "baseline_data" / "image_v2_input" / "img_input_train_20.npz", inputs=image_train)
    np.savez(data_root / "baseline_data" / "image_v2_input" / "img_input_validation_20.npz", inputs=image_val)
    np.savez(data_root / "baseline_data" / "lidar_input" / "lidar_train.npz", input=lidar_train)
    np.savez(data_root / "baseline_data" / "lidar_input" / "lidar_validation.npz", input=lidar_val)
