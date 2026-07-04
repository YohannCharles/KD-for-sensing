"""TII VLRG Transformer external reproduction adapter."""


import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from kd_sensing.baselines.beambench.metrics import official_dba_score


DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/tii_vlrg_transformer_reproduction")
DEFAULT_MODEL_ID = "tii_vlrg_transformer"
DEFAULT_MODALITIES = ("camera", "lidar", "radar", "gps")
DEFAULT_STRICT_PROTOCOL = {
    "split": "deepsense6g_s32_s34_train_s31_s34_eval",
    "scene_set": [31, 32, 33, 34],
    "label_space": "64-beam",
    "metric_profile": "beambench_linear_topk",
    "history_window": 0,
    "gps_source_window": "current",
    "prediction_horizon": 1,
    "seed": 42,
    "difficulty_digest": "clean",
}
MANIFEST_SCHEMA = {
    "required": (
        "model_id",
        "source_repo",
        "source_commit",
        "enabled_modalities",
        "scene_set",
        "split",
        "metric_profile",
        "output_root",
        "status",
        "warnings",
    )
}


def load_manifest_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"TII manifest config must be a mapping: {path}")
    return payload


def output_root_from(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return Path(override or config.get("output_root") or DEFAULT_OUTPUT_ROOT)


def run_reproduction(
    config_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = True,
    execute: bool = False,
    manifest_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    command_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    config = load_manifest_config(config_path)
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    root = output_root_from(config, output_root)
    manifest = build_manifest(config, output_root=root, dry_run=dry_run, execute=execute, command_args=command_args)
    if execute:
        manifest["execution"] = execute_external_commands(manifest)
        manifest["artifacts"] = _artifact_status(manifest)
        manifest["status"] = _status_after_execution(manifest)
        manifest["warnings"] = [item["warning"] for item in manifest["artifacts"] if item.get("warning")]
    manifest_path = write_json(manifest, manifest_output or root / "manifests" / "manifest.json")
    result: dict[str, Any] = {"manifest": manifest, "manifest_path": str(manifest_path)}
    if manifest["status"] != "blocked" and (_existing_path(config.get("metrics_path")) or _existing_path(config.get("prediction_path"))):
        summary = build_summary_row(manifest)
        summary_path = write_json(summary, summary_output or root / "summaries" / "summary_row.json")
        result.update({"summary_row": summary, "summary_path": str(summary_path)})
    return result


def build_manifest(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
    dry_run: bool = True,
    execute: bool = False,
    command_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = output_root_from(config, output_root)
    payload = _base_payload(config, root)
    artifacts = _artifact_status(payload)
    warnings = [item["warning"] for item in artifacts if item.get("warning")]
    status = _status_from_artifacts(artifacts)
    payload.update(
        {
            "status": status,
            "warnings": warnings,
            "artifacts": artifacts,
            "manifest_schema": MANIFEST_SCHEMA,
            "dry_run": {
                "enabled": bool(dry_run and not execute),
                "will_execute": bool(execute),
                "commands": build_dry_run_commands(payload),
            },
            "command_args": list(command_args),
        }
    )
    return payload


def build_dry_run_commands(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    configured = manifest.get("external_commands")
    if configured:
        return [_normalize_command_entry(item) for item in configured]
    source_repo = Path(str(manifest["source_repo"]))
    output_root = Path(str(manifest["output_root"]))
    scene_set = ",".join(str(item) for item in manifest.get("scene_set", []))
    modalities = ",".join(str(item) for item in manifest.get("enabled_modalities", []))
    prediction_path = manifest.get("prediction_path") or str(output_root / "predictions" / "tii_predictions.csv")
    return [
        {
            "stage": "preprocess",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "python",
                str(source_repo / "preprocess.py"),
                "--scene-set",
                scene_set,
                "--modalities",
                modalities,
                "--output-root",
                str(output_root / "cache"),
            ],
        },
        {
            "stage": "infer",
            "command": [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "python",
                str(source_repo / "test.py"),
                "--checkpoint",
                str(manifest.get("checkpoint_path") or output_root / "checkpoints" / "best.pth"),
                "--prediction-output",
                str(prediction_path),
            ],
        },
    ]


def execute_external_commands(manifest: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(str(manifest["output_root"]))
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_repo = Path(str(manifest["source_repo"]))
    if not source_repo.is_dir():
        return {"status": "blocked", "records": [], "reason": f"source_repo missing: {source_repo}"}

    records = []
    for entry in manifest["dry_run"]["commands"]:
        command = [str(item) for item in entry["command"]]
        _validate_external_command(command)
        result = subprocess.run(
            command,
            cwd=source_repo,
            text=True,
            capture_output=True,
            check=False,
        )
        stage = _safe_stage_name(str(entry.get("stage", "command")))
        stdout_path = logs_dir / f"{stage}.stdout.log"
        stderr_path = logs_dir / f"{stage}.stderr.log"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        records.append(
            {
                "stage": stage,
                "command": command,
                "returncode": int(result.returncode),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        if result.returncode != 0:
            return {"status": "blocked", "records": records}
    return {"status": "complete", "records": records}


def build_summary_row(
    manifest: dict[str, Any],
    *,
    strict_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_source = _existing_path(manifest.get("metrics_path"))
    prediction_source = _existing_path(manifest.get("prediction_path"))
    metrics = _read_metrics_csv(metric_source) if metric_source else _read_prediction_csv(prediction_source, manifest)
    protocol = {key: manifest.get(key) for key in DEFAULT_STRICT_PROTOCOL}
    mismatches = _protocol_mismatches(protocol, strict_protocol or DEFAULT_STRICT_PROTOCOL)
    row = {
        "model": manifest.get("model_id", DEFAULT_MODEL_ID),
        "source": "external_tii_vlrg_transformer",
        "overall_clean": metrics.get("overall_clean"),
        "P0": metrics.get("P0"),
        "P1": metrics.get("P1"),
        "P2": metrics.get("P2"),
        "P3": metrics.get("P3"),
        "P4": metrics.get("P4"),
        "P5": metrics.get("P5"),
        "overall_p0_p5_mean": _mean([metrics.get(f"P{index}") for index in range(6)]),
        "strict_comparability": "not_comparable" if mismatches else "strict",
        "strict_ranking_eligible": not mismatches,
        "comparison_scope": "external_reference" if mismatches else "strict_ranking",
        "comparability_mismatches": mismatches,
        "source_artifact_path": str(metric_source or prediction_source),
        "source_repo": manifest.get("source_repo"),
        "source_commit": manifest.get("source_commit"),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "checkpoint_sha256": _fingerprint(manifest.get("checkpoint_path")),
        "metric_profile": manifest.get("metric_profile"),
        "sample_count": metrics.get("sample_count"),
    }
    row.update(protocol)
    return row


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _base_payload(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        "model_id": config.get("model_id", DEFAULT_MODEL_ID),
        "source_repo": str(config.get("source_repo", output_root / "external_repo")),
        "source_commit": str(config.get("source_commit", "unknown")),
        "checkpoint_path": _optional_str(config.get("checkpoint_path")),
        "prediction_path": _optional_str(config.get("prediction_path")),
        "metrics_path": _optional_str(config.get("metrics_path")),
        "external_commands": config.get("external_commands"),
        "enabled_modalities": list(config.get("enabled_modalities") or DEFAULT_MODALITIES),
        "scene_set": list(config.get("scene_set") or [31, 32, 33, 34]),
        "split": config.get("split", "tii_deepsense6g_challenge_scene31_34"),
        "label_space": config.get("label_space", "64-beam"),
        "metric_profile": config.get("metric_profile", "tii_challenge_dba"),
        "history_window": config.get("history_window", 0),
        "gps_source_window": config.get("gps_source_window", "tii_official"),
        "prediction_horizon": config.get("prediction_horizon", 1),
        "seed": config.get("seed"),
        "difficulty_digest": config.get("difficulty_digest", "external_tii_clean"),
        "output_root": str(output_root),
        "dba_delta": float(config.get("dba_delta", 5.0)),
        "prediction_beam_shift": int(config.get("prediction_beam_shift", 0)),
        "label_beam_shift": int(config.get("label_beam_shift", 0)),
    }


def _artifact_status(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _artifact("source_repo", manifest["source_repo"], must_be_dir=True),
        _artifact("checkpoint", manifest.get("checkpoint_path")),
        _artifact("prediction", manifest.get("prediction_path")),
        _artifact("metrics", manifest.get("metrics_path")),
    ]


def _artifact(name: str, value: Any, *, must_be_dir: bool = False) -> dict[str, Any]:
    path = Path(str(value)) if value else None
    exists = bool(path and (path.is_dir() if must_be_dir else path.exists()))
    record = {
        "name": name,
        "path": str(path) if path else None,
        "available": exists,
        "sha256": _fingerprint(path) if exists and path and path.is_file() else None,
    }
    if not exists:
        record["warning"] = f"{name} unavailable: {path or 'not configured'}"
    return record


def _status_from_artifacts(artifacts: list[dict[str, Any]]) -> str:
    available = {item["name"]: bool(item["available"]) for item in artifacts}
    if available.get("metrics") or available.get("prediction"):
        return "imported"
    if not available.get("source_repo"):
        return "pending"
    if not available.get("checkpoint"):
        return "unavailable"
    return "pending"


def _status_after_execution(manifest: dict[str, Any]) -> str:
    execution = manifest.get("execution", {})
    if execution.get("status") == "blocked":
        return "blocked"
    return _status_from_artifacts(manifest.get("artifacts", []))


def _normalize_command_entry(item: dict[str, Any]) -> dict[str, Any]:
    command = item.get("command") if isinstance(item, dict) else None
    if not isinstance(command, list):
        raise ValueError("external_commands entries must use list-form command values.")
    return {"stage": str(item.get("stage", "command")), "command": [str(part) for part in command]}


def _validate_external_command(command: list[str]) -> None:
    if command[:4] != ["conda", "run", "-n", "kd_mm_beam"]:
        raise ValueError("TII external commands must start with: conda run -n kd_mm_beam")


def _safe_stage_name(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_")
    return clean or "command"


def _read_metrics_csv(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    row = _first_csv_row(path)
    values = {
        "overall_clean": _float_from(row, "overall_clean", "overall_dba", "clean_dba", "official_top3_dba", "dba", "DBA"),
        "sample_count": _int_from(row, "sample_count", "samples", "n"),
    }
    for index in range(6):
        values[f"P{index}"] = _float_from(row, f"P{index}", f"p{index}", f"P{index}_dba", f"p{index}_dba")
    if values["overall_clean"] is None:
        values["overall_clean"] = _mean([values[f"P{index}"] for index in range(6)])
    return values


def _read_prediction_csv(path: Path | None, manifest: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return {}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    labels = [_int_from(row, "label", "target", "target_beam", "beam_label") for row in rows]
    topk = [
        [
            _int_from(row, "top1", "prediction", "predicted_beam", "beam_pred"),
            _int_from(row, "top2"),
            _int_from(row, "top3"),
        ]
        for row in rows
    ]
    filtered = [(label, [item for item in preds if item is not None]) for label, preds in zip(labels, topk) if label is not None and preds]
    if not filtered:
        return {"overall_clean": _mean([_float_from(row, "dba", "score", "correct") for row in rows]), "sample_count": len(rows)}
    truth = [item[0] for item in filtered]
    preds = [item[1] for item in filtered]
    return {
        "overall_clean": official_dba_score(
            preds,
            truth,
            max_k=min(3, max(len(item) for item in preds)),
            delta=float(manifest.get("dba_delta", 5.0)),
            prediction_beam_shift=int(manifest.get("prediction_beam_shift", 0)),
            label_beam_shift=int(manifest.get("label_beam_shift", 0)),
        ),
        "sample_count": len(filtered),
    }


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty metrics CSV: {path}") from exc


def _protocol_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"actual": actual.get(key), "expected": expected.get(key)}
        for key in expected
        if _normalize(actual.get(key)) != _normalize(expected.get(key))
    }


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _fingerprint(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_from(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    return None


def _int_from(row: dict[str, Any], *names: str) -> int | None:
    value = _float_from(row, *names)
    return None if value is None else int(round(value))


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value
