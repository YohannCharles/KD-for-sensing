from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kd_sensing.baselines.beambench.dataset_check import check_dataset
from kd_sensing.baselines.beambench.metrics import beambench_metric_summary_from_logits


@dataclass(frozen=True)
class MockTrainingConfig:
    data_root: str
    csv: str
    output_dir: str
    num_beams: int = 64
    beam_shift: int = 0
    epochs: int = 3
    batch_size: int = 4
    lr: float = 0.02
    seed: int = 42
    device: str = "cpu"


class BeamBenchCsvFeatureDataset(Dataset):
    def __init__(self, csv_path: str | Path, *, beam_shift: int = 0):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        self.feature_columns = [column for column in self.frame.columns if str(column).startswith("mock_feature_")]
        if not self.feature_columns:
            self.feature_columns = [
                column
                for column in self.frame.columns
                if pd.api.types.is_numeric_dtype(self.frame[column])
                and column not in {"label", "beam_label", "target_label", "target_beam", "timestamp", "seq"}
                and not str(column).startswith("future_beam_label")
            ]
        if not self.feature_columns:
            raise ValueError(f"No numeric feature columns found in {self.csv_path}.")
        self.label_column = _find_label_column(self.frame)
        self.beam_shift = int(beam_shift)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[int(index)]
        features = row[self.feature_columns].astype("float32").to_numpy()
        label = int(round(float(row[self.label_column]))) - self.beam_shift
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class TinyBeamBenchClassifier(nn.Module):
    def __init__(self, input_dim: int, num_beams: int):
        super().__init__()
        hidden = max(16, min(128, int(num_beams)))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(num_beams)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_mock_baseline(config: MockTrainingConfig | dict[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, MockTrainingConfig) else MockTrainingConfig(**config)
    _seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.csv)
    if not csv_path.is_absolute():
        csv_path = Path(cfg.data_root) / csv_path
    csv_path = csv_path.resolve()
    dataset_report = check_dataset(
        cfg.data_root,
        csv_path,
        num_beams=cfg.num_beams,
        beam_shift=cfg.beam_shift,
    )
    dataset = BeamBenchCsvFeatureDataset(csv_path, beam_shift=cfg.beam_shift)
    loader = DataLoader(dataset, batch_size=max(1, int(cfg.batch_size)), shuffle=True)
    device = torch.device(cfg.device)
    model = TinyBeamBenchClassifier(len(dataset.feature_columns), cfg.num_beams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.CrossEntropyLoss()
    epoch_logs = []
    for epoch in range(int(cfg.epochs)):
        total_loss = 0.0
        seen = 0
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        epoch_logs.append({"epoch": int(epoch + 1), "loss": float(total_loss / max(seen, 1))})
    metrics, logits_path = _evaluate_dataset(model, dataset, device=device, output_dir=output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mock_beambench_baseline.pt"
    torch.save(
        {
            "mock_data": True,
            "config": asdict(cfg),
            "feature_columns": dataset.feature_columns,
            "num_beams": int(cfg.num_beams),
            "beam_shift": int(cfg.beam_shift),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    report = {
        "mock_data": True,
        "mode": "mock_train_eval",
        "dataset_report": dataset_report,
        "config": asdict(cfg),
        "feature_columns": list(dataset.feature_columns),
        "checkpoint_path": str(checkpoint_path),
        "logits_path": str(logits_path),
        "metrics": metrics,
        "epoch_logs": epoch_logs,
        "warning": "MOCK smoke only; metrics are not real BeamBench reproduction results.",
    }
    (output_dir / "train_log.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    data_root: str | Path,
    csv: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    csv_path = Path(csv)
    if not csv_path.is_absolute():
        csv_path = Path(data_root) / csv_path
    csv_path = csv_path.resolve()
    dataset = BeamBenchCsvFeatureDataset(csv_path, beam_shift=int(ckpt.get("beam_shift", 0)))
    model = TinyBeamBenchClassifier(len(ckpt["feature_columns"]), int(ckpt["num_beams"]))
    model.load_state_dict(ckpt["model_state_dict"])
    metrics, logits_path = _evaluate_dataset(model.to(device), dataset, device=torch.device(device), output_dir=Path(output_dir))
    report = {
        "mock_data": bool(ckpt.get("mock_data", False)),
        "mode": "mock_eval",
        "checkpoint_path": str(checkpoint),
        "csv": str(csv_path),
        "metrics": metrics,
        "logits_path": str(logits_path),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "eval_log.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_dataset(
    model: nn.Module,
    dataset: BeamBenchCsvFeatureDataset,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = torch.stack([dataset[index][0] for index in range(len(dataset))]).to(device)
    labels = torch.stack([dataset[index][1] for index in range(len(dataset))]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(features).detach().cpu()
    metrics = beambench_metric_summary_from_logits(
        logits,
        labels.detach().cpu(),
        num_beams=int(logits.shape[-1]),
        label_beam_shift=0,
    )
    logits_path = output_dir / "mock_logits.npy"
    np.save(logits_path, logits.numpy())
    return metrics, logits_path


def _find_label_column(frame: pd.DataFrame) -> str:
    for name in ("label", "beam_label", "target_label", "target_beam", "future_beam_label1", "future_beam_label"):
        if name in frame.columns:
            return name
    raise ValueError("No label column found; expected label/beam_label/target_label/target_beam.")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
