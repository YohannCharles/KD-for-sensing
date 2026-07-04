import json
from pathlib import Path
from typing import Any, Mapping


def resolve_label_space_config(data_cfg: Mapping[str, Any], label_space: str) -> dict[str, Any]:
    label_spaces = _mapping(data_cfg.get("label_spaces"))
    if label_space not in label_spaces:
        if label_space == "mapping_disabled":
            return {
                "enabled": False,
                "label_space": "raw",
                "num_classes": int(data_cfg.get("num_beams", 64)),
            }
        raise ValueError(f"data.label_space must be one of {sorted(label_spaces)}, got {label_space}.")
    spec = _mapping(label_spaces[label_space])
    if not bool(spec.get("enabled", False)):
        return {"enabled": False, "label_space": "raw", "num_classes": int(data_cfg.get("num_beams", 64))}
    for key in ("mapping_file", "fallback_mapping_file"):
        path = spec.get(key)
        if path and Path(str(path)).exists():
            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            payload["enabled"] = True
            payload.setdefault("fit_source", str(path))
            return payload
    raise FileNotFoundError(
        "mapping_enabled requires an existing mapping_file or fallback_mapping_file. "
        f"Checked: {spec.get('mapping_file')}, {spec.get('fallback_mapping_file')}"
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["resolve_label_space_config"]
