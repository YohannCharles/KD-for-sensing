#!/usr/bin/env python3
"""Launch the fixed-budget MMW T2 hyperparameter development screen."""

import argparse
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Any

import yaml

from launch_mmw_all_weather_matrix import MODALITIES, ROOT, T2_BASE_CONFIG, build_config, domains, preflight


DEFAULT_OUTPUT_ROOT = "outputs/mmw_t2_hyperparameter_screening_v1"
BASELINE_RECIPE = ROOT / T2_BASE_CONFIG
INNER_SPLIT_SEED = 1
INNER_VALIDATION_FRACTION = 0.10
INNER_SPLIT_STRATEGY = "group_safe_time_block"
VARIANTS = (
    "H0-base",
    "H1-BPA+",
    "H2-BPA-sharp",
    "H3-mask-tail",
    "H4-optimizer",
    "H5-KL+",
)
SELECTION_RULE = {
    "score": "0.20*clean + 0.20*mean(drop1,drop2,drop3) + 0.25*temporal_auc + 0.35*temporal_drop80",
    "minimum_delta_pp": {"clean": -0.5, "modality_missing_mean": -0.5, "temporal_drop80": -0.5},
    "development_only": True,
}
VARIANT_PROTOCOL = {
    "H0-base": {"matched_control": None, "allowed_effective_fields": []},
    "H1-BPA+": {
        "matched_control": "H0-base",
        "allowed_effective_fields": ["bpa.outer_weight", "bpa.modality_weight"],
    },
    "H2-BPA-sharp": {
        "matched_control": "H1-BPA+",
        "allowed_effective_fields": ["bpa.prototype_temperature", "bpa.gaussian_sigma"],
    },
    "H3-mask-tail": {
        "matched_control": "H0-base",
        "allowed_effective_fields": ["temporal_missing.train_temporal_missing_rates", "temporal_missing.train_missing_drop_counts"],
    },
    "H4-optimizer": {
        "matched_control": "H0-base",
        "allowed_effective_fields": ["optimizer.type", "optimizer.weight_decay", "scheduler"],
    },
    "H5-KL+": {"matched_control": "H0-base", "allowed_effective_fields": ["superset_consistency.kl_weight"]},
}


def validate_batch_size(value: int) -> int:
    batch_size = int(value)
    if batch_size <= 0 or batch_size % 16:
        raise ValueError("batch_size must be a positive multiple of 16.")
    return batch_size


def select_highest_common_safe_batch(
    probe_results_by_gpu: dict[int, list[dict[str, Any]]],
    *,
    memory_fraction_limit: float = 0.90,
) -> int:
    """Return the largest 16-multiple that passed on every requested physical GPU."""
    if not probe_results_by_gpu:
        raise ValueError("probe_results_by_gpu must not be empty.")
    if not 0.0 < float(memory_fraction_limit) <= 1.0:
        raise ValueError("memory_fraction_limit must be in (0, 1].")

    common: set[int] | None = None
    for physical_gpu, rows in probe_results_by_gpu.items():
        safe_batches: set[int] = set()
        for row in rows:
            try:
                batch_size = validate_batch_size(int(row.get("requested_batch_size", 0)))
                actual_batch_size = int(row.get("actual_batch_size", -1))
                peak_fraction = float(row.get("peak_reserved_fraction", float("inf")))
                logical_device = int(row.get("logical_cuda_device", -1))
                visible_count = int(row.get("visible_cuda_device_count", -1))
                reported_gpu = int(row.get("physical_gpu", -1))
            except (TypeError, ValueError):
                continue
            if (
                row.get("status") == "safe"
                and reported_gpu == int(physical_gpu)
                and row.get("cuda_visible_devices") == str(int(physical_gpu))
                and logical_device == 0
                and visible_count == 1
                and actual_batch_size == batch_size
                and peak_fraction <= float(memory_fraction_limit)
            ):
                safe_batches.add(batch_size)
        if not safe_batches:
            raise RuntimeError(f"GPU{physical_gpu} has no safe, trusted batch probe.")
        common = safe_batches if common is None else common & safe_batches

    if not common:
        raise RuntimeError("No common safe 16-multiple batch exists across the requested GPUs.")
    return max(common)


def collect_probe_results(
    roots: list[Path],
    *,
    gpus: tuple[int, ...],
    memory_fraction_limit: float,
) -> dict[str, Any]:
    records_by_gpu: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    sources: list[str] = []
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"Batch-probe root is missing: {resolved_root}")
        sources.append(str(resolved_root))
        for path in sorted(resolved_root.glob("**/result.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                physical_gpu = int(payload.get("physical_gpu", -1))
            except (TypeError, ValueError):
                continue
            if physical_gpu not in records_by_gpu:
                continue
            payload["report_path"] = str(path)
            records_by_gpu[physical_gpu].append(payload)
    selected = select_highest_common_safe_batch(
        records_by_gpu,
        memory_fraction_limit=memory_fraction_limit,
    )
    compact_records = {
        str(gpu): [
            {
                key: row.get(key)
                for key in (
                    "report_path",
                    "physical_gpu",
                    "requested_batch_size",
                    "actual_batch_size",
                    "status",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "peak_reserved_fraction",
                    "cuda_visible_devices",
                    "visible_cuda_device_count",
                    "logical_cuda_device",
                    "probe_domain_id",
                    "error",
                )
                if key in row
            }
            for row in rows
        ]
        for gpu, rows in sorted(records_by_gpu.items())
    }
    return {
        "source_roots": sources,
        "memory_fraction_limit": float(memory_fraction_limit),
        "selected_common_batch_size": int(selected),
        "records": compact_records,
    }


def _force_single_probe_gpu(physical_gpu: int) -> str:
    """Bind a probe subprocess before torch is imported."""
    visible = str(int(physical_gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def build_screening_config(
    variant: str,
    output_root: Path,
    *,
    seed: int,
    batch_size: int,
    epochs: int = 40,
    domain_inventory: list[dict[str, str]] | None = None,
    split_fingerprint: str | None = None,
    baseline_fingerprint: str | None = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown screening variant {variant!r}.")
    validate_batch_size(batch_size)
    payload = build_config(
        "T2",
        output_root,
        seed=int(seed),
        smoke=False,
        epochs=int(epochs),
        batch_size=int(batch_size),
        umask_training_profile="umask_h4_v1" if variant == "H4-optimizer" else "legacy_h0_v1",
    )
    training = payload.setdefault("training", {})
    loss = payload.setdefault("loss", {})
    u_mask = loss.setdefault("u_mask_beam_jepa", {})
    primary = payload.setdefault("model", {}).setdefault("primary", {})

    payload.setdefault("experiment", {}).update(
        {
            "name": variant,
            "hyperparameter_screening_variant": variant,
        }
    )
    training.update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "validation": {"interval_epochs": 5},
        }
    )
    if domain_inventory is not None:
        payload.setdefault("data", {}).setdefault("dataset", {})["domains"] = deepcopy(domain_inventory)
    payload["output"] = {
        "dir": str(output_root / variant),
        "run_name": f"seed{int(seed)}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    _apply_variant(payload, variant)
    payload["mmw_t2_hyperparameter_screening"] = {
        "protocol": "mmw_t2_hyperparameter_screening_v1",
        "variant": variant,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "validation_interval_epochs": 5,
        "development_only": True,
        "claim_eligible": False,
        "screening_consumed_test": True,
        "matched_control": VARIANT_PROTOCOL[variant]["matched_control"],
        "allowed_effective_fields": deepcopy(VARIANT_PROTOCOL[variant]["allowed_effective_fields"]),
        "baseline_recipe_sha256": baseline_fingerprint,
        "inner_split_fingerprint": split_fingerprint,
        "selection_rule": deepcopy(SELECTION_RULE),
    }
    payload.setdefault("mmw_all_weather_protocol", {}).update(
        {
            "screening_role": "development_hyperparameter_screening",
            "checkpoint_policy": "fixed_epoch_last_pth",
        }
    )
    return payload


def _apply_variant(payload: dict[str, Any], variant: str) -> None:
    u_mask = payload.setdefault("loss", {}).setdefault("u_mask_beam_jepa", {})
    primary = payload.setdefault("model", {}).setdefault("primary", {})
    if variant == "H0-base":
        return
    if variant in {"H1-BPA+", "H2-BPA-sharp"}:
        _set_bpa_strength(payload, outer_weight=0.25, modality_weight=0.15)
    if variant == "H2-BPA-sharp":
        u_mask["beam_label_sigma"] = 1.5
        primary["beam_proto_temperature"] = 0.08
    elif variant == "H3-mask-tail":
        temporal = payload.setdefault("temporal_missing", {})
        temporal["train_temporal_missing_rates"] = "0.0,0.2,0.4,0.6,0.8,0.8"
        temporal["train_missing_drop_counts"] = "0,1,2,3,3"
    elif variant == "H4-optimizer":
        # H4 is now materialized by the explicit profile before variant overlays run.
        return
    elif variant == "H5-KL+":
        superset = deepcopy(u_mask.get("superset_consistency", {}))
        superset["enabled"] = True
        superset["confidence_gated_kl"] = True
        superset["kl_weight"] = 0.5
        superset["temperature"] = 2.0
        u_mask["superset_consistency"] = superset


def _set_bpa_strength(payload: dict[str, Any], *, outer_weight: float, modality_weight: float) -> None:
    u_mask = payload.setdefault("loss", {}).setdefault("u_mask_beam_jepa", {})
    u_mask.update(
        {
            "use_beam_prototype_alignment": True,
            "lambda_proto": float(outer_weight),
            "lambda_modality_proto": float(modality_weight),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _domain_csv_path(domain: dict[str, str], key: str) -> Path:
    raw = Path(str(domain[key]))
    return raw if raw.is_absolute() else ROOT / str(domain["data_root"]) / raw


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"MMW split CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _label_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("beam_label", "")).strip()
        counts[label] = counts.get(label, 0) + 1
    return {label: counts[label] for label in sorted(counts)}


def _validate_baseline_snapshot() -> dict[str, Any]:
    if not BASELINE_RECIPE.exists():
        raise FileNotFoundError(f"T2 baseline recipe is missing: {BASELINE_RECIPE}")
    payload = build_config(
        "T2",
        Path("outputs/mmw_t2_hyperparameter_baseline"),
        seed=1,
        smoke=False,
        epochs=40,
        batch_size=32,
        umask_training_profile="legacy_h0_v1",
    )
    primary = payload.get("model", {}).get("primary", {})
    dataset = payload.get("data", {}).get("dataset", {})
    training = payload.get("training", {})
    if primary.get("head_type") != "prototype":
        raise ValueError("The T2 baseline snapshot is not a prototype-head configuration.")
    if tuple(primary.get("modalities", ())) != MODALITIES:
        raise ValueError("The T2 baseline snapshot does not use the required four sensing modalities.")
    if len(dataset.get("domains", ())) != 15 or int(dataset.get("seq_len", 0)) != 5 or int(dataset.get("num_pred", 0)) != 1:
        raise ValueError("The T2 baseline snapshot does not match the 15-domain 5-to-1 MMW protocol.")
    if int(training.get("epochs", 0)) != 40 or int(training.get("max_epochs", 0)) != 40:
        raise ValueError("The T2 baseline snapshot does not match the fixed 40-epoch protocol.")
    return {
        "path": str(BASELINE_RECIPE),
        "sha256": _sha256_file(BASELINE_RECIPE),
        "architecture": {
            "head_type": primary.get("head_type"),
            "modalities": list(primary.get("modalities", ())),
            "seq_len": int(dataset.get("seq_len", 0)),
            "num_pred": int(dataset.get("num_pred", 0)),
            "domain_ids": [str(item.get("id")) for item in dataset.get("domains", ())],
        },
    }


def _inner_split_dir(output_root: Path, domain: dict[str, str]) -> Path:
    safe_id = str(domain["id"]).replace("/", "__").replace(" ", "_")
    return output_root / "inner_splits" / safe_id


def _outer_split_metadata_path(domain: dict[str, str]) -> Path:
    return _domain_csv_path(domain, "train_csv_name").parent / "split_metadata.json"


def _validate_outer_split_metadata(metadata: dict[str, Any], *, domain_id: str) -> tuple[int, int]:
    if metadata.get("split_strategy") != INNER_SPLIT_STRATEGY or not bool(metadata.get("strict_validation_eligible", False)):
        raise ValueError(f"{domain_id} outer split is not an eligible group-safe MMW split.")
    if int(metadata.get("seq_len", 0)) != 5 or int(metadata.get("pred_len", metadata.get("num_pred", 0))) != 1:
        raise ValueError(f"{domain_id} outer split does not have the required 5-to-1 temporal window.")
    block_size = int(metadata.get("block_size_frames", 0))
    guard_band = int(metadata.get("guard_band_frames", 0))
    if block_size <= 0 or guard_band < 5:
        raise ValueError(f"{domain_id} outer split has invalid group-safe block or guard-band metadata.")
    return block_size, guard_band


def _screening_time_axis_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Route source windows through one shared time axis for the inner split."""
    routed: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["_screening_row_id"] = str(index)
        item["contiguous_segment_id"] = "__screening_time_axis__"
        routed.append(item)
    return routed


def _rows_from_routed_split(
    source_rows: list[dict[str, str]],
    routed_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in routed_rows:
        try:
            source_index = int(row["_screening_row_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The routed inner split lost its source-row identity.") from exc
        result.append(dict(source_rows[source_index]))
    return result


def _resource_overlap_counts(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]]) -> dict[str, int]:
    prefixes = ("camera", "radar", "gps", "lidar", "beam", "future_beam")
    counts: dict[str, int] = {}
    for prefix in prefixes:
        def values(rows: list[dict[str, str]]) -> set[str]:
            return {
                str(value).strip()
                for row in rows
                for key, value in row.items()
                if key.startswith(prefix)
                and key[len(prefix) :].isdigit()
                and str(value).strip() not in {"", "-99"}
            }

        left = values(left_rows)
        right = values(right_rows)
        counts[prefix] = len(left & right)
    return counts


def build_inner_validation_domains(output_root: Path, *, seed: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Derive a shared group-safe development validation split from each outer train CSV."""
    from kd_sensing.data.mmw.preparation_splits import compute_split_leakage_diagnostics, split_sequence_rows

    split_root = output_root / "inner_splits"
    if split_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing inner split artifacts: {split_root}")
    rewritten_domains: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for outer_domain in domains():
        domain = {key: str(value) for key, value in outer_domain.items()}
        source_train = _domain_csv_path(domain, "train_csv_name")
        source_test = _domain_csv_path(domain, "test_csv_name")
        metadata_path = _outer_split_metadata_path(domain)
        if not source_train.exists() or not source_test.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"{domain['id']} is missing an outer MMW split input or metadata file.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        block_size, guard_band = _validate_outer_split_metadata(metadata, domain_id=domain["id"])
        fieldnames, outer_train_rows = _read_csv_rows(source_train)
        if not outer_train_rows:
            raise ValueError(f"{domain['id']} outer train CSV is empty.")
        split = split_sequence_rows(
            _screening_time_axis_rows(outer_train_rows),
            seed=int(seed),
            train_ratio=1.0 - INNER_VALIDATION_FRACTION,
            strategy=INNER_SPLIT_STRATEGY,
            seq_len=5,
            pred_len=1,
            block_size_frames=block_size,
            guard_band_frames=guard_band,
        )
        train_rows = _rows_from_routed_split(outer_train_rows, list(split["train_rows"]))
        validation_rows = _rows_from_routed_split(outer_train_rows, list(split["test_rows"]))
        if not train_rows or not validation_rows:
            raise ValueError(f"{domain['id']} group-safe inner split produced an empty train or validation role.")
        train_validation = compute_split_leakage_diagnostics(
            train_rows,
            validation_rows,
            seq_len=5,
            pred_len=1,
            guard_band_frames=guard_band,
        )
        if any(
            int(train_validation.get(key, 0)) != 0
            for key in ("train_test_frame_overlap_count", "adjacent_window_cross_split_count", "guard_band_violations")
        ):
            raise ValueError(f"{domain['id']} group-safe inner split failed its train/validation leakage audit.")
        destination = _inner_split_dir(output_root, domain)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite inner split directory: {destination}")
        train_path = destination / "train.csv"
        validation_path = destination / "validation.csv"
        _write_csv_rows(train_path, fieldnames, train_rows)
        _write_csv_rows(validation_path, fieldnames, validation_rows)
        rewritten = deepcopy(domain)
        rewritten["train_csv_name"] = str(train_path.resolve())
        rewritten["val_csv_name"] = str(validation_path.resolve())
        rewritten_domains.append(rewritten)
        records.append(
            {
                "id": domain["id"],
                "source_train_csv": str(source_train),
                "source_train_sha256": _sha256_file(source_train),
                "outer_test_csv": str(source_test),
                "outer_test_sha256": _sha256_file(source_test),
                "outer_split_metadata": str(metadata_path),
                "outer_split_metadata_sha256": _sha256_file(metadata_path),
                "inner_train_csv": str(train_path.resolve()),
                "inner_train_sha256": _sha256_file(train_path),
                "inner_validation_csv": str(validation_path.resolve()),
                "inner_validation_sha256": _sha256_file(validation_path),
                "seed": int(seed),
                "validation_fraction": INNER_VALIDATION_FRACTION,
                "split_strategy": INNER_SPLIT_STRATEGY,
                "block_size_frames": block_size,
                "guard_band_frames": guard_band,
                "source_train_rows": len(outer_train_rows),
                "inner_train_rows": len(train_rows),
                "inner_validation_rows": len(validation_rows),
                "label_histograms": {
                    "source_train": _label_histogram(outer_train_rows),
                    "inner_train": _label_histogram(train_rows),
                    "inner_validation": _label_histogram(validation_rows),
                },
                "group_assignments": split.get("group_assignments", []),
                "train_validation_leakage": train_validation,
                "train_validation_identity_audit": {
                    "status": "passed",
                    "source": "group_safe_csv_leakage_diagnostic",
                    "diagnostics": train_validation,
                },
                "outer_test_resource_overlap": _resource_overlap_counts(
                    [*train_rows, *validation_rows],
                    _read_csv_rows(source_test)[1],
                ),
                "outer_test_identity_policy": "outer_test_preserved_not_claim_eligible",
            }
        )
    if len(rewritten_domains) != 15:
        raise ValueError(f"Expected 15 MMW domains, wrote {len(rewritten_domains)} inner split domains.")
    payload = {
        "protocol": "mmw_t2_hyperparameter_screening_inner_split_v1",
        "seed": int(seed),
        "validation_fraction": INNER_VALIDATION_FRACTION,
        "split_strategy": INNER_SPLIT_STRATEGY,
        "domains": records,
    }
    payload["fingerprint"] = _sha256_payload(payload)
    return rewritten_domains, payload


def _assert_screening_config_matches_baseline(
    config: dict[str, Any],
    *,
    baseline: dict[str, Any],
) -> None:
    primary = config.get("model", {}).get("primary", {})
    dataset = config.get("data", {}).get("dataset", {})
    training = config.get("training", {})
    if primary.get("head_type") != baseline["architecture"]["head_type"]:
        raise ValueError("Screening config changed the frozen T2 head type.")
    if list(primary.get("modalities", ())) != baseline["architecture"]["modalities"]:
        raise ValueError("Screening config changed the frozen T2 modality inventory.")
    if int(dataset.get("seq_len", 0)) != baseline["architecture"]["seq_len"] or int(dataset.get("num_pred", 0)) != baseline["architecture"]["num_pred"]:
        raise ValueError("Screening config changed the frozen temporal window.")
    domain_ids = [str(item.get("id")) for item in dataset.get("domains", ())]
    if domain_ids != baseline["architecture"]["domain_ids"]:
        raise ValueError("Screening config changed the frozen 15-domain inventory.")
    if int(training.get("epochs", 0)) != 40 or int(training.get("max_epochs", 0)) != 40:
        raise ValueError("Screening config changed the fixed 40-epoch budget.")


def _effective_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config

    resolved = u_mask_beam_jepa_config(config)
    superset = resolved.get("superset_consistency", {})
    primary = config.get("model", {}).get("primary", {})
    return {
        "bpa": {
            "outer_weight": float(resolved.get("lambda_proto", 0.0)),
            "modality_weight": float(resolved.get("lambda_modality_proto", 0.0)),
            "prototype_temperature": float(primary.get("beam_proto_temperature", 0.0)),
            "gaussian_sigma": float(resolved.get("beam_label_sigma", 0.0)),
        },
        "superset_consistency": {"kl_weight": float(superset.get("kl_weight", 0.0))},
        "optimizer": config.get("training", {}).get("optimizer", {"type": "adam"}),
        "weight_decay": float(config.get("training", {}).get("weight_decay", 0.0)),
        "scheduler": deepcopy(config.get("scheduler", {})),
    }


def build_jobs(
    variants: tuple[str, ...],
    gpus: tuple[int, ...],
    output_root: Path,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be non-empty and unique.")
    if len(gpus) != len(variants) or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Each variant requires one unique non-negative GPU.")
    return [
        {
            "variant": variant,
            "seed": int(seed),
            "gpu": int(gpu),
            "config_path": str(output_root / "generated_configs" / f"{variant}_seed{seed}.yaml"),
            "log_path": str(output_root / "logs" / f"{variant}_seed{seed}.log"),
            "run_dir": str(output_root / variant / f"seed{seed}"),
            "status": "planned",
        }
        for variant, gpu in zip(variants, gpus)
    ]


def write_screening_plan(
    output_root: Path,
    *,
    variants: tuple[str, ...],
    gpus: tuple[int, ...],
    seed: int,
    batch_size: int,
    epochs: int,
    batch_probe: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    validate_batch_size(batch_size)
    manifest_path = output_root / "screening_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing screening manifest: {manifest_path}")
    jobs = build_jobs(variants, gpus, output_root, seed=seed)
    conflicts = [Path(job[key]) for job in jobs for key in ("config_path", "log_path", "run_dir") if Path(job[key]).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing screening artifacts:\n" + "\n".join(map(str, conflicts)))
    baseline = _validate_baseline_snapshot()
    inner_domains, inner_split = build_inner_validation_domains(output_root, seed=seed)
    report = preflight(inner_domains, enabled_modalities=MODALITIES)
    if report.get("status") != "ready":
        raise RuntimeError(f"MMW preflight failed: {report.get('failures', [])}")
    (output_root / "generated_configs").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_json(output_root / "inner_split_manifest.json", inner_split)
    for job in jobs:
        config = build_screening_config(
            job["variant"],
            output_root,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            domain_inventory=inner_domains,
            split_fingerprint=str(inner_split["fingerprint"]),
            baseline_fingerprint=str(baseline["sha256"]),
        )
        _assert_screening_config_matches_baseline(config, baseline=baseline)
        config_path = Path(job["config_path"])
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        job.update(
            {
                "matched_control": VARIANT_PROTOCOL[job["variant"]]["matched_control"],
                "allowed_effective_fields": deepcopy(VARIANT_PROTOCOL[job["variant"]]["allowed_effective_fields"]),
                "effective_hyperparameters": _effective_hyperparameters(config),
                "config_sha256": _sha256_file(config_path),
                "inner_split_fingerprint": str(inner_split["fingerprint"]),
                "baseline_recipe_sha256": str(baseline["sha256"]),
            }
        )
    manifest = {
        "protocol": "mmw_t2_hyperparameter_screening_v1",
        "development_only": True,
        "created_at": _now(),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "claim_eligible": False,
        "screening_consumed_test": True,
        "baseline": baseline,
        "inner_split": inner_split,
        "batch_probe": deepcopy(batch_probe) if batch_probe is not None else None,
        "selection_rule": deepcopy(SELECTION_RULE),
        "preflight_path": str(output_root / "preflight.json"),
        "jobs": jobs,
    }
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def probe_training_step(
    config_path: Path,
    report_path: Path,
    *,
    physical_gpu: int,
    memory_fraction_limit: float,
) -> dict[str, Any]:
    import torch

    from kd_sensing.config import load_config
    from kd_sensing.engine.batch_step import BatchStepRunner
    from kd_sensing.engine.data_factory import build_dataloaders
    from kd_sensing.engine.optim import build_device, build_model, build_optimizer, build_task_criterion
    from kd_sensing.engine.runtime import (
        configure_cuda_performance_settings,
        configure_torch_runtime_threads,
        make_grad_scaler,
        resolve_amp_settings,
        transfer_non_blocking,
    )
    from kd_sensing.engine.trainer import _build_training_extensions
    from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
    from kd_sensing.engine.training_extensions import ExtensionContext
    from kd_sensing.utils.seed import set_seed

    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataloaders = None
    optimizer = None
    primary_model = None
    batch_runner = None
    extension_context = None
    extensions = None
    extension_states = None
    task_criterion = None
    raw_batch = None
    batch_result = None
    train_iterator = None
    result: dict[str, Any] = {
        "physical_gpu": int(physical_gpu),
        "config_path": str(config_path),
        "memory_fraction_limit": float(memory_fraction_limit),
        "requested_batch_size": None,
        "status": "failed",
    }
    try:
        expected_visible = str(int(physical_gpu))
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
            raise RuntimeError(
                "Probe GPU binding is untrusted: set CUDA_VISIBLE_DEVICES to the requested physical GPU "
                "before importing torch."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("batch probe requires CUDA.")
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("batch probe requires exactly one visible CUDA device at logical index 0.")

        cfg = load_config(config_path)
        cfg.setdefault("experiment", {})["device"] = "cuda:0"
        probe_protocol = cfg.get("mmw_t2_hyperparameter_screening", {})
        if probe_protocol.get("probe_scope") == "single_representative_domain":
            probe_domains = cfg.get("data", {}).get("dataset", {}).get("domains", [])
            if not isinstance(probe_domains, list) or not probe_domains:
                raise ValueError("Single-domain batch probe requires a non-empty MMW domain inventory.")
            cfg["data"]["dataset"]["domains"] = [deepcopy(probe_domains[0])]
            result["probe_domain_id"] = str(probe_domains[0].get("id"))
        requested_batch_size = validate_batch_size(cfg["data"]["dataloader"]["train_batch_size"])
        result["requested_batch_size"] = requested_batch_size
        configure_torch_runtime_threads(cfg)
        set_seed(cfg.get("experiment", {}).get("seed", 0))
        dataloaders = build_dataloaders(cfg)
        device = build_device(cfg)
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"batch probe resolved unexpected device {device}.")
        configure_cuda_performance_settings(cfg, device)
        non_blocking = transfer_non_blocking(cfg)
        amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
        primary_model = build_model(cfg["model"]["primary"]).to(device)
        primary_model.train()
        task_criterion = build_task_criterion(cfg)
        optimizer = build_optimizer(cfg, primary_model)
        model_cfg = cfg["model"]
        extension_context = ExtensionContext(
            cfg=cfg,
            task=cfg["experiment"].get("task", "image"),
            model_cfg=model_cfg,
            training_cfg=cfg["training"],
            primary_model=primary_model,
            task_criterion=task_criterion,
            run_dir=report_path.parent,
            device=device,
            num_pred=model_cfg.get("num_pred", 3),
            num_classes=model_cfg.get("num_classes", 64),
            seq_length=model_cfg.get("seq_length", 8),
            non_blocking=non_blocking,
        )
        extensions = _build_training_extensions(cfg)
        extension_states = [extension.setup(extension_context) for extension in extensions]
        for extension, state in zip(extensions, extension_states):
            extension.before_epoch(extension_context, state, epoch=0)
        batch_runner = BatchStepRunner(
            cfg=cfg,
            task=extension_context.task,
            model_cfg=model_cfg,
            training_cfg=cfg["training"],
            optimizer=optimizer,
            grad_scaler=make_grad_scaler(cfg, amp_enabled),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            extension_context=extension_context,
            extensions=extensions,
            extension_states=extension_states,
        )

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        train_iterator = iter(dataloaders["train"])
        raw_batch = next(train_iterator)
        batch_result = batch_runner.run(raw_batch, epoch=0, step=0)
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        total = int(props.total_memory)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        actual_batch_size = int(batch_result.labels.shape[0])
        peak_fraction = peak_reserved / max(total, 1)
        status = "safe"
        if actual_batch_size != requested_batch_size:
            status = "unexpected_batch_size"
        elif peak_fraction > float(memory_fraction_limit):
            status = "unsafe_memory_fraction"
        result.update(
            {
                "status": status,
                "device_name": str(props.name),
                "total_memory_bytes": total,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_fraction": peak_fraction,
                "actual_batch_size": actual_batch_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "visible_cuda_device_count": int(torch.cuda.device_count()),
                "logical_cuda_device": int(torch.cuda.current_device()),
                "loss": float(batch_result.total_loss.detach().cpu().item()),
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "oom", "error": str(exc), "traceback": traceback.format_exc()})
    except Exception as exc:  # noqa: BLE001 - retain probe evidence for launch decisions.
        result.update(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del batch_result, raw_batch, train_iterator, batch_runner, extension_states, extensions
        del extension_context, task_criterion, primary_model, optimizer
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass
        _write_json(report_path, result)
    return result


def launch_jobs(manifest_path: Path, manifest: dict[str, Any]) -> int:
    jobs = manifest["jobs"]
    running: list[tuple[subprocess.Popen, dict[str, Any], Any]] = []
    for job in jobs:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
            }
        )
        command = [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "--no-capture-output",
            "kd-sensing-train",
            "--config",
            job["config_path"],
        ]
        handle = Path(job["log_path"]).open("w", encoding="utf-8")
        job.update({"status": "running", "start_time": _now(), "command": command})
        running.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    _write_json(manifest_path, manifest)
    failed = False
    for process, job, handle in running:
        code = process.wait()
        handle.close()
        job.update({"status": "done" if code == 0 else "failed", "return_code": code, "end_time": _now()})
        failed = failed or code != 0
        _write_json(manifest_path, manifest)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the MMW T2 hyperparameter development screen.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--memory-fraction-limit", type=float, default=0.90)
    parser.add_argument(
        "--probe-results-root",
        action="append",
        default=[],
        help="Ignored output root containing prior probe result.json artifacts; repeatable.",
    )
    args = parser.parse_args(argv)
    try:
        batch_size = validate_batch_size(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs != 40:
        parser.error("This development screen is fixed to 40 epochs.")
    if args.seed <= 0:
        parser.error("seed must be positive.")
    if not 0.0 < args.memory_fraction_limit <= 1.0:
        parser.error("memory-fraction-limit must be in (0, 1].")
    output_root = ROOT / args.output_root
    if args.probe_only:
        if args.physical_gpu is None or args.physical_gpu < 0:
            parser.error("--probe-only requires --physical-gpu.")
        _force_single_probe_gpu(args.physical_gpu)
        probe_root = output_root / "batch_probes" / f"gpu{args.physical_gpu}_batch{batch_size}"
        probe_root.mkdir(parents=True, exist_ok=True)
        baseline = _validate_baseline_snapshot()
        probe_domains = domains()
        probe_preflight = preflight(probe_domains, enabled_modalities=MODALITIES)
        if probe_preflight.get("status") != "ready":
            parser.error(f"MMW probe preflight failed: {probe_preflight.get('failures', [])}")
        config_path = probe_root / "config.yaml"
        config = build_screening_config(
            "H0-base",
            probe_root,
            seed=args.seed,
            batch_size=batch_size,
            epochs=args.epochs,
            domain_inventory=probe_domains,
            split_fingerprint="probe_train_shape_only",
            baseline_fingerprint=str(baseline["sha256"]),
        )
        config["mmw_t2_hyperparameter_screening"].update(
            {
                "probe_scope": "single_representative_domain",
                "probe_scope_reason": "per-step GPU memory depends on model and batch tensor shapes, not the 15-domain sampler inventory",
            }
        )
        _assert_screening_config_matches_baseline(config, baseline=baseline)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result = probe_training_step(
            config_path,
            probe_root / "result.json",
            physical_gpu=args.physical_gpu,
            memory_fraction_limit=args.memory_fraction_limit,
        )
        result.update(
            {
                "baseline_recipe_sha256": baseline["sha256"],
                "inner_split_fingerprint": "probe_train_shape_only",
                "preflight_status": probe_preflight["status"],
            }
        )
        _write_json(probe_root / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "safe" else 2
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")
    try:
        gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
        batch_probe = None
        if args.probe_results_root:
            batch_probe = collect_probe_results(
                [ROOT / item if not Path(item).is_absolute() else Path(item) for item in args.probe_results_root],
                gpus=gpus,
                memory_fraction_limit=args.memory_fraction_limit,
            )
            if int(batch_probe["selected_common_batch_size"]) != batch_size:
                raise ValueError(
                    "Requested batch size does not equal the highest trusted common probe result: "
                    f"requested={batch_size}, selected={batch_probe['selected_common_batch_size']}."
                )
        elif not args.dry_run:
            raise ValueError("Launching training requires --probe-results-root evidence for the common batch.")
        manifest_path, manifest = write_screening_plan(
            output_root,
            variants=variants,
            gpus=gpus,
            seed=args.seed,
            batch_size=batch_size,
            epochs=args.epochs,
            batch_probe=batch_probe,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(manifest_path), "jobs": manifest["jobs"]}, indent=2))
    return 0 if args.dry_run else launch_jobs(manifest_path, manifest)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
