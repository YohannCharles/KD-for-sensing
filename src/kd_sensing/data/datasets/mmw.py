from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.registries import DATASETS


@DATASETS.register("mmw")
class MMWDataset(DeepSense6GDataset):
    """Prepared MMW sequence dataset using the existing beam/mmWave sample contract."""

    def __init__(
        self,
        condition: str = "sunny",
        scene: str | None = "town10_skybridge_seed24",
        scene_id: str | None = None,
        scene_slug: str | None = None,
        data_root: str | None = None,
        train_csv_name: str | None = None,
        test_csv_name: str | None = None,
        val_csv_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        scenario = str(scene or scene_slug or scene_id or "town10_skybridge_seed24")
        layout = mmw_condition_layout(condition)
        root = data_root or layout.root
        prepared_prefix = Path("Prepared") / scenario / "splits"
        csi_enabled = bool(kwargs.get("use_csi", False)) or "csi" in set(kwargs.get("enabled_modalities") or ())
        if csi_enabled:
            if kwargs.get("csv_name"):
                kwargs["csv_name"] = _ensure_csi_columns(root, str(kwargs["csv_name"]), scenario)
            if kwargs.get("root_csv"):
                kwargs["root_csv"] = _ensure_csi_columns(root, str(kwargs["root_csv"]), scenario)
            train_csv_name = _ensure_csi_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
            test_csv_name = _ensure_csi_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
            if val_csv_name:
                val_csv_name = _ensure_csi_columns(root, val_csv_name, scenario)
        super().__init__(
            data_root=root,
            train_csv_name=train_csv_name or str(prepared_prefix / "train.csv"),
            test_csv_name=test_csv_name or str(prepared_prefix / "test.csv"),
            val_csv_name=val_csv_name,
            scene=31,
            **kwargs,
        )
        self.condition = str(condition).strip().lower()
        self.scene_slug = scenario
        self.scene_id = scenario


__all__ = ["MMWDataset"]


def _ensure_csi_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("csi") for col in frame.columns):
        return str(csv_path.resolve())
    beam_cols = _numbered_columns(frame.columns, "beam")
    if not beam_cols:
        return str(csv_path.resolve())
    manifest_path = root / "Prepared" / scenario / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise ValueError(
            f"CSI is enabled for MMW dataset but {csv_path} has no csi columns and manifest is missing: "
            f"{manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    if "beam_power_path" not in manifest.columns or "channel_path" not in manifest.columns:
        raise ValueError(
            f"Cannot derive CSI columns from {manifest_path}; expected beam_power_path and channel_path columns."
        )
    channel_by_beam = {
        _norm_path(row["beam_power_path"]): str(row["channel_path"])
        for _, row in manifest.iterrows()
        if str(row.get("beam_power_path", "")).strip() and str(row.get("channel_path", "")).strip()
    }
    missing: list[str] = []
    for idx, beam_col in enumerate(beam_cols, start=1):
        csi_values = []
        for value in frame[beam_col].tolist():
            key = _norm_path(value)
            channel_path = channel_by_beam.get(key)
            if channel_path is None:
                missing.append(str(value))
                channel_path = "-99"
            csi_values.append(channel_path)
        frame[f"csi{idx}"] = csi_values
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            f"Could not derive CSI paths for {len(missing)} beam paths in {csv_path}; examples: {examples}."
        )
    output_path = csv_path.with_name(f"{csv_path.stem}_with_csi{csv_path.suffix}")
    frame.to_csv(output_path, index=False)
    return str(output_path.resolve())


def _numbered_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            selected.append(text)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def _norm_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("/")
