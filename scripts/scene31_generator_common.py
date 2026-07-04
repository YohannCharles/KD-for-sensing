#!/usr/bin/env python3

import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = "configs/scene31/templates/main_v3_proto_es20_base.yaml"
DEFAULT_MANIFEST_FIELDNAMES = ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"]


def write_scene31_manifest_configs(
    *,
    specs: list[dict[str, Any]],
    base_config: Path,
    out_dir: Path,
    output_dir: str | None,
    overwrite: bool,
    expected_epochs: int,
    fieldnames: list[str] | None = None,
    skip_config_modes: set[str] | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    columns = list(fieldnames or DEFAULT_MANIFEST_FIELDNAMES)
    skipped_modes = skip_config_modes or set()

    for spec in specs:
        run_name = _run_name(spec)
        execution_mode = spec.get("execution_mode", "train")
        config_path = out_dir / f"{run_name}.yaml"
        should_write_config = execution_mode not in skipped_modes
        if should_write_config and (overwrite or not config_path.exists()):
            payload = config_payload(base_config, config_path, run_name, int(spec.get("seed") or 1), spec)
            if output_dir is not None:
                payload.setdefault("output", {})["dir"] = str(output_dir)
            for key, value in spec.get("extra", {}).items():
                payload[key] = value
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append(
            _row_for_spec(
                spec,
                run_name=run_name,
                config_path="" if not should_write_config else rel(config_path),
                expected_epochs=expected_epochs,
                include_execution_mode="execution_mode" in columns,
            )
        )

    write_manifest(out_dir, rows, columns)
    return rows


def write_manifest(out_dir: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with (out_dir / "experiment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def config_payload(base_config: Path, config_path: Path, run_name: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    base_text = os.path.relpath((ROOT / base_config).resolve(), (ROOT / config_path.parent).resolve())
    payload: dict[str, Any] = {
        "_base_": base_text,
        "experiment": {"name": run_name, "seed": int(seed)},
        "model": {"primary": {"ablation_id": run_name}},
        "training": dict(spec.get("training", {})),
        "loss": {"u_mask_beam_jepa": {}},
        "evaluation": {"beam_distance_circular": True},
        "output": {"run_name": run_name},
    }
    if spec.get("model"):
        payload["model"]["primary"].update(spec["model"])
    if spec.get("loss"):
        payload["loss"].update(spec["loss"])
    payload["loss"].setdefault("u_mask_beam_jepa", {})
    payload["loss"]["u_mask_beam_jepa"].update(payload["training"])
    return payload


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_name(spec: dict[str, Any]) -> str:
    if spec.get("run_name"):
        return str(spec["run_name"])
    seed = spec.get("seed")
    return str(spec["name"]) if seed is None else f"{spec['name']}_seed{seed}"


def _row_for_spec(
    spec: dict[str, Any],
    *,
    run_name: str,
    config_path: str,
    expected_epochs: int,
    include_execution_mode: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": run_name,
        "group": spec["group"],
        "config_path": config_path,
        "seed": "" if spec.get("seed") is None else spec["seed"],
        "method_tags": ",".join(spec["tags"]),
        "expected_epochs": spec.get("expected_epochs", expected_epochs),
        "priority": spec.get("priority", "medium"),
    }
    if include_execution_mode:
        row["execution_mode"] = spec.get("execution_mode", "train")
    return row
