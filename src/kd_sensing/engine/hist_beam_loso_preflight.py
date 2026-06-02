from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.data.loso import SUPPORTED_LABEL_BUDGETS
from kd_sensing.engine.hist_beam_loso_artifacts import _csv_header, _csv_records, _numbered_columns, _resolve_csv_path, _resolve_resource_path, _write_json
from kd_sensing.engine.hist_beam_loso_config import _cfg_for_scene, _cpu_thread_config, _enabled_modalities, _excluded_sensitive_fields, _modality_profile_metadata, validate_loso_variant
from kd_sensing.engine.hist_beam_loso_matrix import matrix_summary
from kd_sensing.engine.hist_beam_loso_records import _run_identity, _utc_now
from kd_sensing.preprocessing.mmw_radar import materialize_mmw_radar_split_csv
from kd_sensing.utils.paths import resolve_path


def ensure_mmw_radar_csv_for_preflight(data_root: Path, csv_path: Path, scene: str) -> Path:
    result = materialize_mmw_radar_split_csv(data_root, csv_path, scene, require_maps=True)
    return Path(result["path"])


def preflight_error(scene: Any, resource_type: str, path: str | None, message: str, runs: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {
        "scene": scene,
        "resource_type": resource_type,
        "path": path,
        "message": message,
        "runs": [_run_identity(run) for run in (runs or [])],
    }

def run_loso_execute_preflight(plan: dict[str, Any], cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    from kd_sensing.engine.run_metadata import cache_run_metadata
    from kd_sensing.engine.runtime import configure_torch_runtime_threads

    out_dir = Path(output_dir)
    errors: list[dict[str, Any]] = []
    checked_paths: list[dict[str, Any]] = []
    matrix = matrix_summary(plan)
    try:
        cpu_threads = {
            "configured": _cpu_thread_config(cfg),
            "applied": configure_torch_runtime_threads(cfg),
        }
    except Exception as exc:  # noqa: BLE001 - preflight should report thread config errors.
        cpu_threads = {"configured": _cpu_thread_config(cfg), "applied": {}, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(preflight_error("runtime", "cpu_threads", None, str(cpu_threads["error"]), None))
    runs = list(plan.get("runs", []))
    if not runs:
        errors.append(preflight_error("matrix", "runs", None, "LOSO execute matrix contains no runs.", None))
    for variant in matrix["variants"]:
        try:
            validate_loso_variant(variant)
        except ValueError as exc:
            errors.append(preflight_error("matrix", "variant", None, str(exc), None))
    for budget in matrix["budgets"]:
        if int(budget) not in SUPPORTED_LABEL_BUDGETS:
            errors.append(
                preflight_error(
                    "matrix",
                    "budget",
                    None,
                    f"Unsupported label budget '{budget}'. Supported budgets: {list(SUPPORTED_LABEL_BUDGETS)}.",
                    None,
                )
            )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".loso_preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checked_paths.append({"resource_type": "output_dir", "path": str(out_dir), "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        errors.append(
            preflight_error(
                "output",
                "output_dir",
                str(out_dir),
                f"Output directory is not writable: {exc}",
                None,
            )
        )

    enabled_modalities = _enabled_modalities(plan, cfg)
    scene_ids = sorted(
        {
            scene
            for run in runs
            for scene in [run.get("target_scene"), *list(run.get("source_scenes", []))]
            if scene is not None
        },
        key=str,
    )
    for scene in scene_ids:
        scene_cfg = _cfg_for_scene(cfg, scene)
        dataset_cfg = scene_cfg.get("data", {}).get("dataset", {})
        data_root = resolve_path(dataset_cfg.get("data_root", "."))
        if not data_root.exists() or not data_root.is_dir():
            errors.append(
                preflight_error(
                    scene,
                    "data_root",
                    str(data_root),
                    f"Scene {scene} data root is missing.",
                    _runs_for_scene(runs, scene),
                )
            )
            continue
        checked_paths.append({"scene": scene, "resource_type": "data_root", "path": str(data_root), "status": "ok"})
        for split_key, csv_name in (
            ("train_csv", dataset_cfg.get("train_csv_name")),
            ("test_csv", dataset_cfg.get("test_csv_name")),
        ):
            csv_path = _resolve_csv_path(data_root, csv_name)
            if csv_path is None or not csv_path.exists():
                errors.append(
                    preflight_error(
                        scene,
                        split_key,
                        str(csv_path) if csv_path is not None else None,
                        f"Scene {scene} required CSV is missing.",
                        _runs_for_scene(runs, scene),
                    )
                )
                continue
            if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw" and "radar" in enabled_modalities:
                try:
                    csv_path = ensure_mmw_radar_csv_for_preflight(data_root, csv_path, str(dataset_cfg.get("scene", scene)))
                except Exception as exc:  # noqa: BLE001 - report before training starts.
                    errors.append(
                        preflight_error(
                            scene,
                            "radar_derived_csv",
                            str(csv_path),
                            f"Could not materialize MMW radar columns before training: {exc}",
                            _runs_for_scene(runs, scene),
                        )
                    )
                    continue
            checked_paths.append({"scene": scene, "resource_type": split_key, "path": str(csv_path), "status": "ok"})
            errors.extend(
                _preflight_csv_resources(
                    scene=scene,
                    csv_path=csv_path,
                    data_root=data_root,
                    enabled_modalities=enabled_modalities,
                    cfg=scene_cfg,
                    runs=_runs_for_scene(runs, scene),
                )
            )

    return {
        "status": "passed" if not errors else "failed",
        "checked_at": _utc_now(),
        "checked_scenes": scene_ids,
        "checked_paths": checked_paths,
        "enabled_modalities": list(enabled_modalities),
        "modality_profile": _modality_profile_metadata(plan, cfg),
        "excluded_sensitive_fields": list(_excluded_sensitive_fields(cfg)),
        "cache": cache_run_metadata(cfg),
        "dataloader": _dataloader_preflight_metadata(cfg),
        "cpu_threads": cpu_threads,
        "output_dir": str(out_dir),
        "matrix": matrix,
        "errors": errors,
    }


def _preflight_csv_resources(
    *,
    scene: Any,
    csv_path: Path,
    data_root: Path,
    enabled_modalities: tuple[str, ...],
    cfg: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    rows = _csv_records(csv_path)
    header = list(rows[0].keys()) if rows else _csv_header(csv_path)
    required = _required_column_prefixes(enabled_modalities)
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        required = [prefix for prefix in required if prefix != "bs_gps"]
    seq_len = int(cfg.get("data", {}).get("dataset", {}).get("seq_len", 1))
    num_pred = int(cfg.get("data", {}).get("dataset", {}).get("num_pred", 1))
    minimum_by_prefix = {prefix: seq_len for prefix in required}
    minimum_by_prefix["future_beam"] = num_pred
    for prefix in required:
        cols = _numbered_columns(header, prefix)
        minimum = minimum_by_prefix[prefix]
        if len(cols) < minimum:
            errors.append(
                preflight_error(
                    scene,
                    "csv_columns",
                    str(csv_path),
                    f"{csv_path} contains {len(cols)} {prefix} columns; expected at least {minimum}.",
                    runs,
                )
            )
            continue
        for row_index, row in enumerate(rows):
            for col in cols[:minimum]:
                value = str(row.get(col, "")).strip()
                if not value or value == "-99":
                    continue
                path = _resolve_resource_path(data_root, value)
                if path is not None and not path.exists():
                    errors.append(
                        preflight_error(
                            scene,
                            f"{prefix}_resource",
                            str(path),
                            f"Scene {scene} enabled resource '{prefix}' referenced by {csv_path}:{row_index + 2} is missing.",
                            runs,
                        )
                    )
                    break
                if prefix == "radar":
                    doppler_path = _resolve_resource_path(data_root, str(value).replace("_RA", "_DA"))
                    if doppler_path is not None and not doppler_path.exists():
                        errors.append(
                            preflight_error(
                                scene,
                                "radar_doppler_resource",
                                str(doppler_path),
                                f"Scene {scene} radar Doppler resource derived from {csv_path}:{row_index + 2} is missing.",
                                runs,
                            )
                        )
                        break
            if errors and errors[-1].get("resource_type") == f"{prefix}_resource":
                break
            if errors and errors[-1].get("resource_type") == "radar_doppler_resource":
                break
    return errors


def _required_column_prefixes(enabled_modalities: tuple[str, ...]) -> list[str]:
    prefixes = ["beam", "future_beam"]
    if "image" in enabled_modalities:
        prefixes.append("camera")
    if "radar" in enabled_modalities:
        prefixes.append("radar")
    if "gps" in enabled_modalities:
        prefixes.extend(["gps", "bs_gps"])
    if "lidar" in enabled_modalities:
        prefixes.append("lidar")
    if "mmwave" in enabled_modalities:
        prefixes.append("mmwave")
    if "csi" in enabled_modalities:
        prefixes.append("csi")
    return prefixes


def _dataloader_preflight_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    from kd_sensing.engine.data_factory import resolve_dataloader_split_config

    loader_cfg = cfg.get("data", {}).get("dataloader", {}) if isinstance(cfg.get("data"), dict) else {}
    return {
        "train": resolve_dataloader_split_config(loader_cfg, split="train"),
        "test": resolve_dataloader_split_config(loader_cfg, split="test"),
    }


def _runs_for_scene(runs: list[dict[str, Any]], scene: Any) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if str(run.get("target_scene", "")) == str(scene)
        or str(scene) in {str(item) for item in run.get("source_scenes", [])}
    ]

__all__ = ["ensure_mmw_radar_csv_for_preflight", "preflight_error", "run_loso_execute_preflight"]
