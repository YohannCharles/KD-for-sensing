from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.deepsense6g_camera_residual import (
    CameraResidualManifestDataset,
    build_camera_residual_manifest,
    collate_camera_residual_batch,
    manifest_fingerprint,
    write_manifest_with_ae_features,
)
from kd_sensing.data.deepsense6g_residual import ratio_tag
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen DeepSense6G camera AE features.")
    parser.add_argument("--config", "-c", default="configs/deepsense6g_camera_residual.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--support-ratio", type=float, default=None)
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_config(args.config, overrides)
    return extract_camera_ae_features(
        cfg,
        checkpoint=args.checkpoint,
        support_ratio=args.support_ratio,
        label_space=args.label_space,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        force=bool(args.force),
    )


def extract_camera_ae_features(
    cfg: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    support_ratio: float | None = None,
    label_space: str | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
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
    ae_dir = Path(outputs_cfg.get("default_ae_dir") or Path(str(outputs_cfg.get("ae_training_root", "outputs/training/deepsense6g_camera_ae"))) / tag / selected_label_space)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else ae_dir / "checkpoints" / "best.pt"
    feature_dir = Path(
        output_dir
        or outputs_cfg.get("default_feature_dir")
        or (Path(str(outputs_cfg.get("feature_root", "outputs/features/deepsense6g_camera_ae"))) / tag / selected_label_space)
    )
    features_path = feature_dir / "features.npy"
    index_path = feature_dir / "features_index.csv"
    metadata_path = feature_dir / "feature_metadata.json"
    fingerprint = manifest_fingerprint(
        manifest,
        extra={"checkpoint": str(checkpoint_path), "latent_dim": int(ae_cfg.get("latent_dim", 128))},
    )
    if not force and features_path.exists() and index_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            manifest_result = write_manifest_with_ae_features(
                manifest,
                features_path=features_path,
                features_index_path=index_path,
            )
            metadata["skipped_reason"] = "existing_features_fingerprint_match"
            metadata.update(manifest_result)
            return metadata

    try:
        dataset = CameraResidualManifestDataset(
            manifest,
            stage="ae_training",
            image_size=int(image_cfg.get("size", 64)),
            use_target_query_unlabeled=True,
        )
    except ValueError:
        feature_dir.mkdir(parents=True, exist_ok=True)
        empty = np.zeros((0, int(ae_cfg.get("latent_dim", 128))), dtype=np.float32)
        np.save(features_path, empty)
        _write_csv(index_path, [])
        manifest_result = write_manifest_with_ae_features(
            manifest,
            features_path=features_path,
            features_index_path=index_path,
        )
        metadata = {
            "workflow": "deepsense6g_camera_ae_feature_extraction",
            "feature_count": 0,
            "features_shape": list(empty.shape),
            "fingerprint": fingerprint,
            "skipped_reason": "no_available_images",
            "checkpoint_path": str(checkpoint_path),
            **manifest_result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return metadata

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Camera AE checkpoint is required for feature extraction: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    model_meta = dict(payload.get("model_metadata") or {})
    model = CameraAutoEncoder(
        latent_dim=int(model_meta.get("latent_dim", ae_cfg.get("latent_dim", 128))),
        image_channels=int(model_meta.get("image_channels", image_cfg.get("channels", 3))),
        image_size=int(model_meta.get("image_size", image_cfg.get("size", 64))),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(ae_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(ae_cfg.get("num_workers", 0)),
        collate_fn=collate_camera_residual_batch,
    )
    features: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    row_index = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            latent = model.encode(image).detach().cpu().numpy().astype(np.float32)
            features.append(latent)
            for local_idx, sample_id in enumerate(batch["sample_id"]):
                index_rows.append(
                    {
                        "row_index": row_index,
                        "scene": batch["scene"][local_idx],
                        "sample_id": sample_id,
                        "split_role": batch["split_role"][local_idx],
                    }
                )
                row_index += 1
    feature_array = np.concatenate(features, axis=0) if features else np.zeros((0, model.latent_dim), dtype=np.float32)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_path, feature_array)
    _write_csv(index_path, index_rows)
    manifest_result = write_manifest_with_ae_features(
        manifest,
        features_path=features_path,
        features_index_path=index_path,
    )
    metadata = {
        "workflow": "deepsense6g_camera_ae_feature_extraction",
        "checkpoint_path": str(checkpoint_path),
        "features_path": str(features_path),
        "features_index_path": str(index_path),
        "feature_count": int(feature_array.shape[0]),
        "features_shape": list(feature_array.shape),
        "fingerprint": fingerprint,
        "query_label_used_for_training": False,
        **manifest_result,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    if not fieldnames:
        fieldnames = ["row_index", "scene", "sample_id", "split_role"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
