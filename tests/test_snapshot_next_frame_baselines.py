import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.config import load_config
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.engine.batch import prepare_labels
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.normalization_artifacts import _fingerprint, validate_normalization_artifact_fingerprint
from kd_sensing.engine.prediction_objectives import PredictionTargets, compute_prediction_loss
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.preprocessing.sequences import generate_sequence_data
from kd_sensing.registries import ENCODERS


@ENCODERS.register("snapshot_test_identity", force=True)
class SnapshotTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"snapshot_test_identity expects [B, T, D], got {tuple(batch.shape)}.")
        if int(batch.shape[-1]) != self.output_dim:
            raise ValueError(f"expected D={self.output_dim}, got {tuple(batch.shape)}.")
        return batch


def test_snapshot_frame_single_and_five_modality_forward_shapes_and_no_rnn():
    single = _snapshot_model(["gps"], num_classes=5)
    single_output = single(gps_batch=torch.randn(2, 1, 8))

    assert single_output["logits"].shape == (2, 1, 5)
    assert not _contains_recurrent_module(single)

    fusion = _snapshot_model(["image", "radar", "gps", "lidar", "mmwave"], num_classes=7)
    kwargs = {key: torch.randn(2, 1, 8) for key in ("image_batch", "radar_batch", "gps_batch", "lidar_batch", "mmwave_batch")}
    fusion_output = fusion(**kwargs)

    assert fusion_output["logits"].shape == (2, 1, 7)
    assert not _contains_recurrent_module(fusion)

    with pytest.raises(ValueError, match="seq_len=1.*num_pred=1"):
        single(gps_batch=torch.randn(2, 2, 8))


def test_snapshot_auxiliary_heads_feed_objective_aware_loss_path():
    model = _snapshot_model(
        ["gps"],
        num_classes=4,
        auxiliary_heads={"enabled": True, "occlusion": True, "position": True},
    )
    output = adapt_model_output(model(gps_batch=torch.randn(2, 1, 8)))
    logits = select_prediction_slots(output.logits, num_pred=1)

    assert output.diagnostics["occlusion_logits"].shape == (2, 1)
    assert output.diagnostics["position"].shape == (2, 1, 2)

    labels = torch.tensor([[1], [2]])
    beam_loss = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
    targets = PredictionTargets(
        labels=labels,
        occlusion_label=torch.tensor([[1.0], [0.0]]),
        occlusion_valid=torch.tensor([[True], [True]]),
        position_target=torch.zeros(2, 1, 2),
        position_valid=torch.tensor([[True], [True]]),
    )
    cfg = {
        "experiment": {"objective": "multitask"},
        "loss": {"objective": {"weights": {"beam": 1.0, "occlusion": 1.0, "position": 1.0}}},
    }

    bundle = compute_prediction_loss(
        output,
        targets,
        cfg,
        reference=logits,
        beam_total_loss=beam_loss,
        beam_task_loss=beam_loss,
    )

    assert torch.isfinite(bundle.total)
    assert bundle.diagnostics["loss/occlusion"] >= 0.0
    assert bundle.diagnostics["loss/position"] >= 0.0


def test_snapshot_preprocessing_writes_validation_split_and_metadata(tmp_path: Path):
    source = _write_snapshot_source(tmp_path, seq_count=5, rows_per_seq=3)

    train_path, val_path = generate_sequence_data(
        source,
        tmp_path,
        "_SNAPSHOT_NEXT_FRAME",
        in_len=1,
        out_len=1,
        training_set_pct=0.8,
        split_strategy="snapshot_next_frame_balanced_seq",
        split_seed=3,
        include_gps=True,
        include_lidar=True,
        include_mmwave=True,
        include_position_targets=True,
    )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    metadata = json.loads((tmp_path / "split_metadata_SNAPSHOT_NEXT_FRAME.json").read_text(encoding="utf-8"))

    assert train_path.name == "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
    assert val_path.name == "val_seqs_SNAPSHOT_NEXT_FRAME.csv"
    assert train["seq_index"].nunique() == 4
    assert val["seq_index"].nunique() == 1
    assert set(train["seq_index"]).isdisjoint(set(val["seq_index"]))
    assert {"camera1", "radar1", "gps1", "bs_gps1", "lidar1", "mmwave1", "beam1", "future_beam1"} <= set(train)
    assert {"future_gps1", "future_bs_gps1"} <= set(train)
    assert metadata["split_protocol"] == "snapshot_next_frame_balanced_seq"
    assert metadata["in_len"] == 1
    assert metadata["out_len"] == 1
    assert metadata["training_set_pct"] == 0.8
    assert metadata["window_counts"]["validation"] == len(val)
    assert metadata["splits"]["validation"]["csv_path"] == str(val_path)


@pytest.mark.parametrize("modality", ["image", "radar", "gps", "lidar", "mmwave"])
def test_single_modality_snapshot_configs_load(modality: str):
    cfg = load_config(ROOT / f"configs/{modality}/snapshot_next_frame_supervised.yaml")

    assert cfg["experiment"]["variant"] == "snapshot_next_frame"
    assert cfg["experiment"]["task"] == modality
    assert "distillation" not in cfg
    assert cfg["data"]["dataset"]["seq_len"] == 1
    assert cfg["data"]["dataset"]["num_pred"] == 1
    assert cfg["data"]["dataset"]["train_csv_name"] == "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
    assert cfg["data"]["dataset"]["val_csv_name"] == "val_seqs_SNAPSHOT_NEXT_FRAME.csv"
    assert cfg["model"]["primary"]["representation_core"]["type"] == "snapshot_frame"
    assert cfg["model"]["primary"]["num_pred"] == 1


def test_snapshot_fusion_configs_and_slug_validation_load():
    cfg = load_config(ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_snapshot_next_frame_supervised.yaml")
    alias = load_config(ROOT / "configs/fusion/all_modalities_snapshot_next_frame_supervised.yaml")

    assert cfg["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert alias["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert alias["output"]["run_name"] == "all_modalities_snapshot_next_frame_supervised"
    with pytest.raises(ValueError, match="gps_mmwave_snapshot_next_frame_supervised.yaml"):
        load_config(ROOT / "configs/fusion/mmwave_gps_snapshot_next_frame_supervised.yaml")


def test_snapshot_config_rejects_history_window_override():
    with pytest.raises(ValueError, match="data.dataset.seq_len=1"):
        load_config(ROOT / "configs/gps/snapshot_next_frame_supervised.yaml", ["data.dataset.seq_len=2"])


def test_snapshot_labels_use_future_beam_not_current_input(tmp_path: Path):
    beam_now = _beam_file(tmp_path, "beam_now.txt", 3)
    beam_future = _beam_file(tmp_path, "beam_future.txt", 11)
    gps = tmp_path / "gps.txt"
    bs = tmp_path / "bs.txt"
    gps.write_text("34.0 -112.0\n", encoding="utf-8")
    bs.write_text("34.1 -112.1\n", encoding="utf-8")
    csv_path = tmp_path / "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
    csv_path.write_text(
        "gps1,bs_gps1,beam1,future_beam1,seq_index\n"
        f"{gps.name},{bs.name},{beam_now.name},{beam_future.name},1\n",
        encoding="utf-8",
    )
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=csv_path.name,
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["gps"],
        use_gps=True,
        gps_normalize=False,
    )
    sample = dataset[0]

    labels = prepare_labels(
        {"target_beam": sample["target_beam"].unsqueeze(0)},
        num_pred=1,
        downsample_ratio=1,
        device=torch.device("cpu"),
    )

    assert sample["input_beam"].tolist() == [3]
    assert sample["target_beam"].tolist() == [11]
    assert labels.tolist() == [[11]]


def test_snapshot_artifact_fingerprint_rejects_history_window_split():
    cfg = load_config(ROOT / "configs/mmwave/snapshot_next_frame_supervised.yaml")
    artifact_metadata = {
        "fit_split": "train",
        "effective_sample_count": 10,
        "domain_policy": "shared",
        "normalization_modalities": ["mmwave"],
        "feature_mode": None,
        "split_protocol": "balanced_seq",
        "seq_len": 8,
        "num_pred": 3,
    }
    artifact_metadata["normalization_fingerprint"] = _fingerprint(artifact_metadata)

    with pytest.raises(ValueError, match="non-snapshot split"):
        validate_normalization_artifact_fingerprint(
            cfg,
            {
                "normalization_artifacts": {
                    "mmwave_scaler": "/tmp/mmwave_scaler.npz",
                    "metadata": artifact_metadata,
                }
            },
        )


def _snapshot_model(
    modalities: list[str],
    *,
    num_classes: int = 5,
    auxiliary_heads: dict | None = None,
) -> ModularSequenceModel:
    encoders = {modality: {"type": "snapshot_test_identity", "output_dim": 8} for modality in modalities}
    projectors = {modality: {"type": "identity", "input_dim": 8, "d_model": 8} for modality in modalities}
    return ModularSequenceModel(
        modalities=modalities,
        encoders=encoders,
        projectors=projectors,
        representation_core={"type": "snapshot_frame", "d_model": 8, "hidden_size": 8},
        feature_size=8,
        d_model=8,
        num_classes=num_classes,
        num_pred=1,
        auxiliary_heads=auxiliary_heads,
    )


def _contains_recurrent_module(model: nn.Module) -> bool:
    return any(isinstance(module, (nn.GRU, nn.RNN, nn.LSTM)) for module in model.modules())


def _write_snapshot_source(root: Path, *, seq_count: int, rows_per_seq: int) -> Path:
    rows = []
    for seq_idx in range(seq_count):
        for row_idx in range(rows_per_seq):
            beam = _beam_file(root, f"beam_s{seq_idx}_r{row_idx}.txt", (seq_idx + row_idx) % 8)
            rows.append(
                [
                    f"camera_s{seq_idx}_r{row_idx}.jpg",
                    f"radar_s{seq_idx}_r{row_idx}.npy",
                    beam.name,
                    f"gps_s{seq_idx}_r{row_idx}.txt",
                    f"bs_gps_s{seq_idx}_r{row_idx}.txt",
                    f"lidar_s{seq_idx}_r{row_idx}.ply",
                    str(seq_idx),
                ]
            )
    source = root / "scenario31_RA.csv"
    source.write_text(
        "unit1_rgb,unit1_radar,unit1_pwr_60ghz,unit2_loc_cal,unit1_loc,unit1_lidar,seq_index\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return source


def _beam_file(root: Path, name: str, label: int) -> Path:
    values = np.zeros(64, dtype=np.float32)
    values[int(label)] = 1.0
    path = root / name
    np.savetxt(path, values)
    return path
