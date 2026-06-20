import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kd_sensing.data.beam_label_calibration import BeamLabelMapping, resolve_beam_label_mapping


LABEL_SPACE_FIELDS = ("label_space", "beam_label_space", "beam_label_mapping_fingerprint")


def resolve_label_space_mapping(
    data_cfg: Mapping[str, Any],
    label_space: str,
    *,
    num_beams: int = 64,
    scene: str | None = None,
) -> BeamLabelMapping:
    return resolve_beam_label_mapping(
        _resolve_label_space_config(data_cfg, label_space, num_beams=num_beams),
        scene=scene,
        default_num_classes=int(num_beams),
    )


def label_space_metadata(
    data_cfg: Mapping[str, Any],
    label_space: str,
    *,
    num_beams: int = 64,
    scene: str | None = None,
) -> dict[str, Any]:
    mapping = resolve_label_space_mapping(data_cfg, label_space, num_beams=num_beams, scene=scene)
    payload = mapping.metadata()
    payload["label_space"] = str(label_space)
    payload["raw_to_mapped_mapping_source"] = _mapping_source(data_cfg, label_space)
    return payload


def attach_label_space_metadata(row: dict[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    row["label_space"] = str(metadata.get("label_space") or "")
    row["beam_label_space"] = str(metadata.get("beam_label_space") or "")
    row["beam_label_mapping_fingerprint"] = str(metadata.get("beam_label_mapping_fingerprint") or "")
    if metadata.get("raw_to_mapped_mapping_source") is not None:
        row["raw_to_mapped_mapping_source"] = str(metadata.get("raw_to_mapped_mapping_source") or "")
    return row


def validate_label_space_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: Mapping[str, Any],
    source_path: str | Path,
    artifact_name: str,
    require_fields: bool | None = None,
) -> None:
    strict = bool(require_fields) if require_fields is not None else str(expected.get("label_space")) != "mapping_disabled"
    expected_label = str(expected.get("label_space") or "")
    expected_beam_space = str(expected.get("beam_label_space") or "")
    expected_fingerprint = str(expected.get("beam_label_mapping_fingerprint") or "")
    for idx, row in enumerate(rows):
        row_label = str(row.get("label_space") or "")
        row_beam_space = str(row.get("beam_label_space") or "")
        row_fingerprint = str(row.get("beam_label_mapping_fingerprint") or "")
        missing = [
            field
            for field, value in (
                ("label_space", row_label),
                ("beam_label_space", row_beam_space),
                ("beam_label_mapping_fingerprint", row_fingerprint),
            )
            if not value
        ]
        if missing and strict:
            raise ValueError(
                f"{artifact_name} label-space metadata missing in {source_path}: "
                f"row={idx}, missing={missing}, expected_label_space={expected_label}, "
                f"expected_fingerprint={expected_fingerprint}."
            )
        if row_label and row_label != expected_label:
            raise ValueError(
                f"{artifact_name} label-space mismatch in {source_path}: row={idx}, "
                f"source label_space={row_label}, target label_space={expected_label}, "
                f"mapping fingerprint={row_fingerprint or '<missing>'}."
            )
        if row_beam_space and row_beam_space != expected_beam_space:
            raise ValueError(
                f"{artifact_name} beam label-space mismatch in {source_path}: row={idx}, "
                f"source beam_label_space={row_beam_space}, target beam_label_space={expected_beam_space}, "
                f"source fingerprint={row_fingerprint or '<missing>'}, target fingerprint={expected_fingerprint}."
            )
        if row_fingerprint and row_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{artifact_name} mapping fingerprint mismatch in {source_path}: row={idx}, "
                f"source fingerprint={row_fingerprint}, target fingerprint={expected_fingerprint}, "
                f"source label_space={row_label or '<missing>'}, target label_space={expected_label}."
            )


def validate_label_space_metadata(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    source_path: str | Path,
    artifact_name: str,
    require_fields: bool | None = None,
) -> None:
    validate_label_space_rows(
        [metadata],
        expected=expected,
        source_path=source_path,
        artifact_name=artifact_name,
        require_fields=require_fields,
    )


def _resolve_label_space_config(data_cfg: Mapping[str, Any], label_space: str, *, num_beams: int) -> dict[str, Any]:
    label_spaces = data_cfg.get("label_spaces")
    if isinstance(label_spaces, Mapping) and label_space in label_spaces:
        spec = dict(label_spaces[label_space] or {})
    elif label_space == "mapping_disabled":
        spec = {"enabled": False}
    else:
        raise ValueError(f"data.label_space must be one of {sorted(label_spaces) if isinstance(label_spaces, Mapping) else ['mapping_disabled']}, got {label_space}.")
    if not bool(spec.get("enabled", False)):
        return {"enabled": False, "label_space": "raw", "num_classes": int(num_beams)}
    for key in ("mapping_file", "fallback_mapping_file"):
        value = str(spec.get(key) or "").strip()
        if value and Path(value).exists():
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
            payload["enabled"] = True
            payload.setdefault("fit_source", value)
            payload.setdefault("num_classes", int(num_beams))
            return payload
    payload = dict(spec)
    payload["enabled"] = True
    payload.setdefault("label_space", str(label_space))
    payload.setdefault("num_classes", int(num_beams))
    return payload


def _mapping_source(data_cfg: Mapping[str, Any], label_space: str) -> str:
    label_spaces = data_cfg.get("label_spaces")
    if not isinstance(label_spaces, Mapping) or label_space not in label_spaces:
        return "mapping_disabled_builtin" if label_space == "mapping_disabled" else "missing_label_space_config"
    spec = label_spaces[label_space]
    if not isinstance(spec, Mapping) or not bool(spec.get("enabled", False)):
        return "mapping_disabled_builtin"
    for key in ("mapping_file", "fallback_mapping_file"):
        value = str(spec.get(key) or "").strip()
        if value:
            return value
    return "inline_label_space_config"


__all__ = [
    "LABEL_SPACE_FIELDS",
    "attach_label_space_metadata",
    "label_space_metadata",
    "resolve_label_space_mapping",
    "validate_label_space_metadata",
    "validate_label_space_rows",
]
