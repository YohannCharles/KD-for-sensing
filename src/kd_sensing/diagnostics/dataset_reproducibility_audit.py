import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

from kd_sensing.data.layouts import deepsense6g_scene_layout, mmw_condition_layout


DEFAULT_OUTPUT_DIR = Path("outputs/analysis/dataset_audit")
MODALITY_PREFIXES = {
    "camera": ("camera", "image", "rgb", "unit1_rgb"),
    "lidar": ("lidar", "unit1_lidar"),
    "radar": ("radar", "unit1_radar"),
    "gps": ("gps", "bs_gps", "unit1_loc", "unit2_loc", "future_gps", "future_bs_gps"),
    "mmwave": ("mmwave", "beam_power", "power", "unit1_pwr"),
    "csi": ("csi", "channel", "h_path"),
}
LABEL_NAMES = ("label", "beam_label", "target_label", "target_beam", "true_beam", "optimal_beam")
IDENTIFIER_ALIASES = {
    "scene": ("scene", "scene_id", "scenario", "scenario_id"),
    "sample": ("sample", "sample_id", "sample_index", "frame", "frame_id", "index"),
    "sequence": ("seq", "seq_id", "sequence", "sequence_id"),
    "timestamp": ("timestamp", "time", "utc_time", "frame_time"),
}
SPLIT_NAMES = ("train", "validation", "val", "test")


def run_dataset_audit(
    *,
    dataset_family: str,
    data_root: str | Path | None = None,
    csv_path: str | Path | None = None,
    scene: str | int | None = None,
    condition: str | None = None,
    num_beams: int = 64,
    beam_shift: int = 0,
    split_metadata: str | Path | None = None,
    official_artifacts: dict[str, str | Path | None] | None = None,
    local_config: str | Path | None = None,
    local_checkpoint_provenance: str | None = None,
    max_missing_examples: int = 5,
) -> dict[str, Any]:
    descriptor = dataset_layout_descriptor(dataset_family, scene=scene, condition=condition, data_root=data_root)
    root = Path(descriptor["data_root"])
    csv_report = audit_csv(
        root,
        Path(csv_path) if csv_path is not None else None,
        num_beams=int(num_beams),
        beam_shift=int(beam_shift),
        max_missing_examples=max_missing_examples,
    )
    split_report = audit_split_leakage(csv_report.get("rows", []), split_metadata=split_metadata)
    field_summary = csv_report.get("field_summary", {})
    file_summary = csv_report.get("file_reference_summary", {})
    label_summary = csv_report.get("label_summary", {})
    warnings = list(csv_report.get("warnings", []))
    warnings.extend(split_report.get("warnings", []))

    official = official_reproduction_status(official_artifacts or {})
    local = local_substitute_status(
        descriptor=descriptor,
        csv_report=csv_report,
        local_config=local_config,
        local_checkpoint_provenance=local_checkpoint_provenance,
    )
    blocked_reasons = list(official["missing_items"])
    if field_summary and not field_summary.get("has_label"):
        blocked_reasons.append("CSV label column unavailable")
    if label_summary.get("invalid_count", 0):
        blocked_reasons.append("invalid beam labels")

    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "ok": not blocked_reasons and csv_report.get("csv_exists", False),
        "dataset_family": descriptor["dataset_family"],
        "data_root": str(root),
        "csv_path": csv_report.get("csv_path"),
        "scene_scope": descriptor.get("scene_scope"),
        "layout": descriptor,
        "field_summary": field_summary,
        "file_reference_summary": file_summary,
        "label_summary": label_summary,
        "split_summary": split_report,
        "official_reproduction": official,
        "local_substitute": local,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "notes": [
            "dataset audit is read-only; it does not move, delete, copy, or rewrite data",
            "local substitute readiness is not official reproduction",
        ],
    }
    report.pop("rows", None)
    return report


def dataset_layout_descriptor(
    dataset_family: str,
    *,
    scene: str | int | None = None,
    condition: str | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    key = _normalize_family(dataset_family)
    if key in {"deepsense6g", "beambench"}:
        scene_scope = str(scene or 31)
        primary_scene = scene_scope.replace(",", "-").split("-", 1)[0]
        try:
            layout = deepsense6g_scene_layout(primary_scene)
        except ValueError:
            layout = deepsense6g_scene_layout(31)
        canonical_root = Path(layout.canonical_root)
        legacy_root = Path(layout.legacy_root)
        root = Path(data_root) if data_root is not None else canonical_root
        return {
            "dataset_family": "BeamBench/DeepSense6G" if key == "beambench" else "DeepSense6G",
            "layout_key": key,
            "data_root": str(root),
            "canonical_root": str(canonical_root),
            "legacy_roots": [str(legacy_root), "dataset/DeepSense6G/raw_data/test"],
            "required_subdirectories": [],
            "root_status": _root_status(root, canonical_root, [legacy_root]),
            "scene_scope": f"scenario{layout.scene_id}" if scene_scope == str(layout.scene_id) else scene_scope,
            "artifact_boundary": "dataset input only; generated cache/output/checkpoint ignored",
        }
    if key == "mmw":
        layout = mmw_condition_layout(condition or "sunny")
        canonical_root = Path(layout.root)
        root = Path(data_root) if data_root is not None else canonical_root
        return {
            "dataset_family": "MMW",
            "layout_key": "mmw",
            "data_root": str(root),
            "canonical_root": str(canonical_root),
            "legacy_roots": [],
            "required_subdirectories": list(layout.required_subdirs) + ["Prepared"],
            "root_status": _root_status(root, canonical_root, []),
            "required_subdirectory_status": {
                name: (root / name).exists() for name in list(layout.required_subdirs) + ["Prepared"]
            },
            "scene_scope": condition or "sunny",
            "artifact_boundary": "prepared dataset input only; generated cache/output/checkpoint ignored",
        }
    raise ValueError(f"Unsupported dataset family: {dataset_family}")


def audit_csv(
    data_root: Path,
    csv_path: Path | None,
    *,
    num_beams: int,
    beam_shift: int,
    max_missing_examples: int,
) -> dict[str, Any]:
    if csv_path is None:
        return {
            "csv_exists": False,
            "csv_path": None,
            "rows": [],
            "warnings": ["csv path not provided"],
            "field_summary": {},
            "file_reference_summary": {},
            "label_summary": {},
        }
    resolved = csv_path if csv_path.is_absolute() else data_root / csv_path
    if not resolved.exists():
        return {
            "csv_exists": False,
            "csv_path": str(resolved),
            "rows": [],
            "warnings": [f"csv path missing: {resolved}"],
            "field_summary": {},
            "file_reference_summary": {},
            "label_summary": {},
        }
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    columns = list(rows[0]) if rows else []
    fields = resolve_csv_fields(columns)
    file_summary = {
        modality: _file_reference_summary(rows, data_root, fields[f"{modality}_columns"], max_missing_examples)
        for modality in MODALITY_PREFIXES
    }
    label_summary = _label_summary(rows, fields["label_columns"], num_beams=num_beams, beam_shift=beam_shift)
    return {
        "csv_exists": True,
        "csv_path": str(resolved),
        "row_count": len(rows),
        "rows": rows,
        "warnings": [],
        "field_summary": {
            **fields,
            "has_label": bool(fields["label_columns"]),
            "row_count": len(rows),
        },
        "file_reference_summary": file_summary,
        "label_summary": label_summary,
    }


def resolve_csv_fields(columns: list[str]) -> dict[str, Any]:
    lowered = {column: column.lower() for column in columns}
    fields: dict[str, Any] = {}
    for modality, prefixes in MODALITY_PREFIXES.items():
        fields[f"{modality}_columns"] = [
            column for column, lower in lowered.items() if any(lower == prefix or lower.startswith(prefix) for prefix in prefixes)
        ]
    fields["label_columns"] = [
        column
        for column, lower in lowered.items()
        if lower in LABEL_NAMES or lower.startswith("beam") and lower.replace("beam", "").isdigit()
    ]
    for name, aliases in IDENTIFIER_ALIASES.items():
        fields[f"{name}_columns"] = [
            column
            for column, lower in lowered.items()
            if lower in aliases or any(lower.startswith(alias + "_") for alias in aliases)
        ]
    fields["split_columns"] = [column for column, lower in lowered.items() if lower in {"split", "fold", "partition"}]
    return fields


def audit_split_leakage(rows: list[dict[str, str]], *, split_metadata: str | Path | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    row_sets = _split_sets_from_rows(rows)
    metadata_sets: dict[str, dict[str, set[str]]] = {}
    if split_metadata is not None:
        path = Path(split_metadata)
        if path.exists():
            metadata_sets = _split_sets_from_metadata(json.loads(path.read_text(encoding="utf-8")))
        else:
            warnings.append(f"split metadata missing: {path}")
    split_sets = metadata_sets or row_sets
    if not split_sets:
        return {
            "status": "unavailable",
            "source": None if split_metadata is None else str(split_metadata),
            "overlaps": [],
            "warnings": warnings or ["split leakage check unavailable; no split metadata or split columns"],
        }
    overlaps = _overlaps(split_sets)
    return {
        "status": "blocked" if overlaps else "ok",
        "source": "metadata" if metadata_sets else "csv_split_column",
        "overlaps": overlaps,
        "warnings": warnings,
    }


def official_reproduction_status(artifacts: dict[str, str | Path | None]) -> dict[str, Any]:
    required = ("official_data", "official_weights", "official_source", "official_environment")
    availability: dict[str, bool] = {}
    missing: list[str] = []
    for name in required:
        value = artifacts.get(name)
        available = bool(value and Path(value).exists())
        availability[name] = available
        if not available:
            missing.append(name)
    return {
        "status": "ready" if not missing else "blocked",
        "availability": availability,
        "missing_items": missing,
        "next_step": "provide official data, weights, source/config, and environment before reporting official metrics"
        if missing
        else "official artifacts are present; run official evaluation separately",
    }


def local_substitute_status(
    *,
    descriptor: dict[str, Any],
    csv_report: dict[str, Any],
    local_config: str | Path | None,
    local_checkpoint_provenance: str | None,
) -> dict[str, Any]:
    root_exists = Path(descriptor["data_root"]).exists()
    csv_ok = bool(csv_report.get("csv_exists")) and not csv_report.get("label_summary", {}).get("invalid_count", 0)
    config_ok = local_config is None or Path(local_config).exists()
    ready = root_exists and csv_ok and config_ok
    missing = []
    if not root_exists:
        missing.append("data_root")
    if not csv_ok:
        missing.append("csv_or_labels")
    if not config_ok:
        missing.append("local_config")
    return {
        "status": "ready" if ready else "incomplete",
        "ready": ready,
        "missing_items": missing,
        "local_config": None if local_config is None else str(local_config),
        "checkpoint_provenance": local_checkpoint_provenance or "unavailable",
        "claim_boundary": "local substitute readiness is not official reproduction",
    }


def write_audit_report(report: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_audit_report.json"
    md_path = out_dir / "dataset_audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Reproducibility Audit",
        "",
        f"- dataset_family: {report['dataset_family']}",
        f"- data_root: {report['data_root']}",
        f"- csv_path: {report.get('csv_path') or 'unavailable'}",
        f"- official_reproduction: {report['official_reproduction']['status']}",
        f"- local_substitute: {report['local_substitute']['status']}",
        f"- split_leakage_check: {report['split_summary']['status']}",
        "",
        "## Blocked Reasons",
        "",
    ]
    reasons = report.get("blocked_reasons", [])
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _file_reference_summary(
    rows: list[dict[str, str]],
    data_root: Path,
    columns: list[str],
    max_missing_examples: int,
) -> dict[str, Any]:
    total = existing = missing = 0
    examples: list[str] = []
    for row in rows:
        for column in columns:
            value = _clean(row.get(column))
            if not _looks_like_path(value):
                continue
            total += 1
            path = Path(value) if Path(value).is_absolute() else data_root / value
            if path.exists():
                existing += 1
            else:
                missing += 1
                if len(examples) < max_missing_examples:
                    examples.append(str(path))
    return {
        "columns": columns,
        "total_references": total,
        "existing_count": existing,
        "missing_count": missing,
        "missing_ratio": 0.0 if total == 0 else missing / total,
        "missing_examples": examples,
    }


def _label_summary(rows: list[dict[str, str]], columns: list[str], *, num_beams: int, beam_shift: int) -> dict[str, Any]:
    raw_values: list[int] = []
    for row in rows:
        for column in columns:
            value = _coerce_int(row.get(column))
            if value is not None:
                raw_values.append(value)
    shifted = [value - beam_shift for value in raw_values]
    invalid = [value for value in shifted if value < 0 or value >= num_beams]
    return {
        "columns": columns,
        "count": len(raw_values),
        "raw_min": min(raw_values) if raw_values else None,
        "raw_max": max(raw_values) if raw_values else None,
        "shifted_min": min(shifted) if shifted else None,
        "shifted_max": max(shifted) if shifted else None,
        "invalid_count": len(invalid),
        "num_beams": num_beams,
        "beam_shift": beam_shift,
        "label_space": _label_space(raw_values, num_beams=num_beams),
    }


def _split_sets_from_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, set[str]]]:
    if not rows:
        return {}
    fields = resolve_csv_fields(list(rows[0]))
    split_column = next(iter(fields["split_columns"]), None)
    id_columns = fields["sample_columns"] + fields["sequence_columns"]
    if split_column is None or not id_columns:
        return {}
    split_sets: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        split = _clean(row.get(split_column)).lower()
        if not split:
            continue
        bucket = split_sets.setdefault(split, {"ids": set()})
        for column in id_columns:
            value = _clean(row.get(column))
            if value:
                bucket["ids"].add(value)
    return split_sets


def _split_sets_from_metadata(payload: Any) -> dict[str, dict[str, set[str]]]:
    source = payload.get("splits", payload) if isinstance(payload, dict) else {}
    split_sets: dict[str, dict[str, set[str]]] = {}
    if not isinstance(source, dict):
        return split_sets
    for split in SPLIT_NAMES:
        value = source.get(split)
        if value is None:
            continue
        split_sets[split] = {"ids": _collect_ids(value)}
    return split_sets


def _collect_ids(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, dict):
        ids: set[str] = set()
        for key in ("ids", "sample_ids", "sequence_ids", "group_ids", "groups"):
            nested = value.get(key)
            if isinstance(nested, list):
                ids.update(str(item) for item in nested)
        return ids
    return set()


def _overlaps(split_sets: dict[str, dict[str, set[str]]]) -> list[dict[str, Any]]:
    names = sorted(split_sets)
    out: list[dict[str, Any]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(split_sets[left].get("ids", set()) & split_sets[right].get("ids", set()))
            if shared:
                out.append({"left": left, "right": right, "count": len(shared), "examples": shared[:5]})
    return out


def _root_status(root: Path, canonical_root: Path, legacy_roots: list[Path]) -> dict[str, Any]:
    return {
        "exists": root.exists(),
        "canonical_exists": canonical_root.exists(),
        "is_canonical": root == canonical_root,
        "legacy_compatible": any(root == legacy for legacy in legacy_roots),
        "legacy_existing_roots": [str(path) for path in legacy_roots if path.exists()],
    }


def _normalize_family(value: str) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _looks_like_path(value: str) -> bool:
    if not value or value.lower() in {"nan", "none", "-99"}:
        return False
    try:
        float(value)
        return False
    except ValueError:
        return "/" in value or "\\" in value or "." in Path(value).name


def _coerce_int(value: Any) -> int | None:
    text = _clean(value)
    if not text or text.lower() in {"nan", "none", "-99"}:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _label_space(values: list[int], *, num_beams: int) -> str:
    if not values:
        return "unavailable"
    if min(values) >= 0 and max(values) < num_beams:
        return "0-based-like"
    if min(values) >= 1 and max(values) <= num_beams:
        return "1-based-like"
    return "unknown"
