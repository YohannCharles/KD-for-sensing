from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.deepsense6g_camera_residual import (
    CameraResidualManifestDataset,
    build_camera_residual_manifest,
    collate_camera_residual_batch,
)
from kd_sensing.data.deepsense6g_residual import ratio_tag
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DeepSense6G tiny camera autoencoder.")
    parser.add_argument("--config", "-c", default="configs/deepsense6g_camera_residual.yaml")
    parser.add_argument("--support-ratio", type=float, default=None)
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_config(args.config, overrides)
    return train_camera_ae(
        cfg,
        support_ratio=args.support_ratio,
        label_space=args.label_space,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )


def train_camera_ae(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    image_cfg = _mapping(cfg.get("image"))
    ae_cfg = _mapping(cfg.get("ae"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    if manifest_path is None:
        manifest_result = build_camera_residual_manifest(cfg, support_ratio=ratio, label_space=selected_label_space)
        manifest_path = manifest_result["manifest_path"]
    manifest = Path(manifest_path)
    result_dir = Path(
        output_dir
        or outputs_cfg.get("default_ae_dir")
        or (Path(str(outputs_cfg.get("ae_training_root", "outputs/training/deepsense6g_camera_ae"))) / tag / selected_label_space)
    )

    dataset = CameraResidualManifestDataset(
        manifest,
        stage="ae_training",
        image_size=int(image_cfg.get("size", 64)),
        use_target_query_unlabeled=bool(ae_cfg.get("use_target_query_unlabeled", False)),
    )
    seed = int(_mapping(cfg.get("train")).get("seed", 42))
    torch.manual_seed(seed)
    train_dataset, val_dataset = _split_dataset(dataset, val_fraction=float(ae_cfg.get("val_fraction", 0.15)), seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(ae_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(ae_cfg.get("num_workers", 0)),
        collate_fn=collate_camera_residual_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(ae_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(ae_cfg.get("num_workers", 0)),
        collate_fn=collate_camera_residual_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CameraAutoEncoder(
        latent_dim=int(ae_cfg.get("latent_dim", 128)),
        image_channels=int(image_cfg.get("channels", 3)),
        image_size=int(image_cfg.get("size", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ae_cfg.get("lr", 1e-3)),
        weight_decay=float(ae_cfg.get("weight_decay", 1e-4)),
    )
    checkpoint_dir = result_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    metrics_path = result_dir / "metrics.csv"
    start_epoch = 0
    best_val = float("inf")
    if bool(ae_cfg.get("resume", True)) and best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_val_loss", best_val))

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    patience = int(ae_cfg.get("patience", 5))
    stale = 0
    epochs = int(ae_cfg.get("epochs", 20))
    for epoch in range(start_epoch, epochs):
        train_loss = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, device=device, optimizer=None)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "model_metadata": model.metadata(),
                    "manifest_path": str(manifest),
                    "query_label_used_for_training": False,
                    "use_target_query_unlabeled": bool(ae_cfg.get("use_target_query_unlabeled", False)),
                },
                best_path,
            )
        else:
            stale += 1
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val})
        _write_metrics(metrics_path, metrics)
        if stale >= patience:
            break
    if not best_path.exists():
        raise RuntimeError(f"Camera AE training did not produce a checkpoint for manifest: {manifest}")
    _save_reconstruction_examples(
        model,
        val_loader if len(val_dataset) else train_loader,
        result_dir / "recon_examples",
        device=device,
        limit=int(ae_cfg.get("reconstruction_examples", 8)),
    )
    metadata = {
        "workflow": "deepsense6g_camera_ae_train",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest),
        "checkpoint_path": str(best_path),
        "metrics_path": str(metrics_path),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "query_label_used_for_training": False,
        "use_target_query_unlabeled": bool(ae_cfg.get("use_target_query_unlabeled", False)),
        "transductive_unlabeled_query": bool(ae_cfg.get("use_target_query_unlabeled", False)),
    }
    (result_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


def _run_epoch(
    model: CameraAutoEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(train):
            out = model(image)
            loss = F.mse_loss(out["reconstruction"], image)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.detach().cpu()) * int(image.shape[0])
        count += int(image.shape[0])
    return total / max(count, 1)


def _split_dataset(dataset: CameraResidualManifestDataset, *, val_fraction: float, seed: int):
    length = len(dataset)
    if length == 1:
        return dataset, Subset(dataset, [0])
    val_count = max(1, int(round(length * max(float(val_fraction), 0.0))))
    val_count = min(val_count, length - 1)
    train_count = length - val_count
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_count, val_count], generator=generator)


def _save_reconstruction_examples(
    model: CameraAutoEncoder,
    loader: DataLoader,
    output_dir: Path,
    *,
    device: torch.device,
    limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if limit <= 0:
        return
    from PIL import Image

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            recon = model(image)["reconstruction"].detach().cpu()
            original = image.detach().cpu()
            for idx in range(int(original.shape[0])):
                canvas = torch.cat([original[idx], recon[idx]], dim=-1)
                array = (((canvas.clamp(-1, 1) * 0.5) + 0.5) * 255.0).byte().permute(1, 2, 0).numpy()
                Image.fromarray(array).save(output_dir / f"recon_{saved:04d}.png")
                saved += 1
                if saved >= limit:
                    return


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Camera residual config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
