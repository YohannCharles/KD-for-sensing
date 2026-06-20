#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kd_sensing.config.io import safe_load_yaml  # noqa: E402
from kd_sensing.engine.debug_diagnostics import evaluate_pilot_noise_validity  # noqa: E402


FIXED_PILOT_SCALING_VERSION = "fixed_estimation_snr_v1"
LEGACY_INVALID_SWEEP_MARKERS = ("csi_hardening_matrix_20260520_164406",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze CSI hardening sweep training runs.")
    parser.add_argument("--runs_root", required=True, help="Directory containing run subdirectories.")
    parser.add_argument("--pattern", default="csi_*", help="Glob pattern under runs_root.")
    parser.add_argument("--clean_teacher_run", required=True, help="Clean reference run name, path, or run directory name.")
    parser.add_argument("--out", required=True, help="Output directory for summary CSVs and plots.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    runs = discover_runs(Path(args.runs_root), args.pattern)
    rows = [analyze_run(run) for run in runs]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    ranked_path = out_dir / "ranked_candidates.csv"
    summary_rows = add_reference_metrics_rows(rows, args.clean_teacher_run)
    summary_rows = add_debug_decision_rows(summary_rows)
    _write_csv(summary_path, summary_rows)
    ranked_rows = rank_candidate_rows(summary_rows)
    _write_csv(ranked_path, ranked_rows)
    write_plots_rows(summary_rows, out_dir)
    _write_json(out_dir / "analysis_metadata.json", build_analysis_metadata(summary_rows, Path(args.runs_root)))
    run_count = len(summary_rows)
    return {
        "runs": run_count,
        "summary": str(summary_path),
        "ranked_candidates": str(ranked_path),
        "out": str(out_dir),
    }


def discover_runs(runs_root: Path, pattern: str) -> list[Path]:
    root = Path(runs_root)
    candidates = [Path(item) for item in glob.glob(str(root / pattern))]
    runs = []
    for candidate in candidates:
        if candidate.is_dir():
            runs.append(candidate)
        elif candidate.name in {"train_log.json", "final_config.yaml", "metrics.json"}:
            runs.append(candidate.parent)
    return sorted(set(runs), key=lambda path: path.name)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    train_log = _read_json(run_dir / "train_log.json")
    final_config = _read_yaml(run_dir / "final_config.yaml")
    metrics = _read_metrics_artifact(run_dir)
    csi_debug_records = _read_csi_debug_records(run_dir, train_log)
    pilot_validity = _read_pilot_noise_validity(run_dir, final_config, train_log, csi_debug_records)
    val_acc = _series_from_sources(train_log, metrics, ("val_acc", "beam/accuracy_val", "accuracy_val", "val_top1"))
    val_adba = _series_from_sources(train_log, metrics, ("val_adba", "beam/adba_val", "adba_val"))
    epochs = _epoch_series(train_log, len(val_acc))
    final_acc = _last_n_mean(val_acc, 10)
    best_acc = max(val_acc) if val_acc else None
    e50 = _first_epoch_at_fraction(val_acc, epochs, final_acc, 0.50)
    e80 = _first_epoch_at_fraction(val_acc, epochs, final_acc, 0.80)
    e90 = _first_epoch_at_fraction(val_acc, epochs, final_acc, 0.90)
    metadata = _run_metadata(run_dir, final_config, train_log)
    hardening = _find_csi_hardening(final_config)
    degradation = _find_csi_degradation(final_config)
    csi_estimation = _find_csi_estimation(final_config)
    config_diff = _read_json(run_dir / "config_diff.json")
    debug_cfg = final_config.get("debug") if isinstance(final_config.get("debug"), dict) else {}
    return {
        "run_dir": str(run_dir),
        "run_name": metadata["run_name"],
        "experiment_name": metadata["experiment_name"],
        "seed": metadata["seed"],
        "matrix_role": debug_cfg.get("matrix_role"),
        "modalities": ",".join(metadata["modalities"]),
        "num_epochs": len(val_acc),
        "final_acc": final_acc,
        "best_acc": best_acc,
        "final_adba": _last_n_mean(val_adba, 10),
        "best_adba": max(val_adba) if val_adba else None,
        "E50": e50,
        "E80": e80,
        "E90": e90,
        "csi_hardening_enabled": bool(_mapping_enabled(hardening)),
        "csi_degradation_enabled": bool(_mapping_enabled(degradation)),
        "csi_degradation_profile": degradation.get("profile") if isinstance(degradation, dict) else None,
        "csi_estimation_mode": _pilot_mode(csi_estimation),
        "csi_estimation_snr_db": _snr_db_from_estimation(csi_estimation),
        "csi_estimation_train_snr_min_db": csi_estimation.get("train_snr_min_db") if isinstance(csi_estimation, dict) else None,
        "csi_estimation_train_snr_max_db": csi_estimation.get("train_snr_max_db") if isinstance(csi_estimation, dict) else None,
        "has_debug_diagnostics": bool(csi_debug_records),
        "pilot_noise_scale_valid": pilot_validity.get("valid"),
        "pilot_noise_invalid_reason": None if pilot_validity.get("valid") is not False else pilot_validity.get("reason"),
        "pilot_noise_signal_ratio": pilot_validity.get("noise_power_signal_ratio"),
        "pilot_noise_expected_ratio_min": pilot_validity.get("expected_ratio_min"),
        "pilot_noise_expected_ratio_max": pilot_validity.get("expected_ratio_max"),
        "pilot_scaling_config_version": debug_cfg.get("pilot_scaling_config_version"),
        "uses_fixed_pilot_scaling_config": _uses_fixed_pilot_scaling_config(final_config, pilot_validity),
        "is_mild_pilot_estimation": bool(pilot_validity.get("is_mild_pilot_estimation")),
        "is_destructive_control": bool(pilot_validity.get("is_destructive_control")),
        "a0_parity_passed": config_diff.get("parity_passed") if config_diff else None,
        "a0_parity_status": config_diff.get("status") if config_diff else None,
        "curve_epochs": json.dumps(epochs),
        "curve_val_acc": json.dumps(val_acc),
    }


def add_reference_metrics_rows(rows: list[dict[str, Any]], clean_teacher_run: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    clean = _select_clean_reference_row(rows, clean_teacher_run)
    clean_final = _to_float(clean.get("final_acc"))
    clean_e90 = _to_float(clean.get("E90"))
    result = []
    for row in rows:
        item = dict(row)
        final_acc = _to_float(item.get("final_acc"))
        e90 = _to_float(item.get("E90"))
        gap = clean_final - final_acc if clean_final is not None and final_acc is not None else None
        ratio = e90 / clean_e90 if e90 is not None and clean_e90 not in (None, 0.0) else None
        item["clean_final_acc"] = clean_final
        item["clean_E90"] = clean_e90
        item["ceiling_gap_acc"] = gap
        item["E90_ratio"] = ratio
        item["is_destructive"] = bool(gap is not None and gap > 0.05)
        item["is_slow_high_ceiling"] = bool(gap is not None and ratio is not None and gap <= 0.03 and ratio >= 1.5)
        result.append(item)
    return result


def add_debug_decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    original = _find_run_by_tokens(rows, ("A0_original", "A0_clean_full_teacher"))
    clone = _find_run_by_tokens(rows, ("A0_clone_generated",))
    pilot_disabled = _find_run_by_tokens(rows, ("pilot_disabled",))
    c1 = _find_run_by_tokens(rows, ("C1_view_gate_warmup_only", "C1_view_gate_warmup"))
    c2 = _find_run_by_tokens(rows, ("C2_no_internal_gru_only", "C2_no_internal_gru"))
    clone_parity = _row_bool(clone, "a0_parity_passed") if clone is not None else None
    original_acc = _to_float(original.get("final_acc")) if original is not None else None
    clone_acc = _to_float(clone.get("final_acc")) if clone is not None else None
    c1_acc = _to_float(c1.get("final_acc")) if c1 is not None else None
    c2_acc = _to_float(c2.get("final_acc")) if c2 is not None else None
    missing_required_diagnostics = _missing_required_debug_diagnostics(rows, original, clone, pilot_disabled, c1, c2)
    pilot_invalid = any(_row_bool(row, "pilot_noise_scale_valid") is False for row in rows)
    c1_unhealthy = clone_acc is not None and c1_acc is not None and _is_high(clone_acc) and _meaningfully_lower(c1_acc, clone_acc)
    c2_unhealthy = clone_acc is not None and c2_acc is not None and _is_high(clone_acc) and _meaningfully_lower(c2_acc, clone_acc)
    if missing_required_diagnostics:
        full_status = "invalid_due_to_missing_debug_diagnostics"
    elif clone is None or clone_parity is None:
        full_status = "pending_a0_clone_parity"
    elif clone_parity is False:
        full_status = "invalid_due_to_a0_parity"
    elif pilot_invalid:
        full_status = "invalid_due_to_pilot_noise_scale"
    elif c1_unhealthy:
        full_status = "pending_debug_c1_health"
    elif c2_unhealthy:
        full_status = "pending_debug_c2_health"
    else:
        full_status = "valid"

    global_invalid_reason = None
    if full_status.startswith("invalid_due_to"):
        global_invalid_reason = full_status
    elif full_status.startswith("pending"):
        global_invalid_reason = full_status
    result = []
    for row in rows:
        item = dict(row)
        run_name = str(item.get("run_name") or item.get("experiment_name") or "")
        item["a0_parity_status"] = item.get("a0_parity_status") or (clone or {}).get("a0_parity_status")
        decision = global_invalid_reason if full_status != "valid" else "no_debug_issue_detected"
        if original_acc is not None and clone_acc is not None and _is_high(original_acc) and _is_low(clone_acc):
            decision = "config_generation_or_inheritance_failure"
        elif clone_acc is not None and _is_high(clone_acc):
            current_acc = _to_float(item.get("final_acc"))
            if run_name == str((pilot_disabled or {}).get("run_name")) and current_acc is not None and _meaningfully_lower(current_acc, clone_acc):
                decision = "pilot_disabled_parse_or_dataflow_failure"
            elif run_name == str((c1 or {}).get("run_name")) and current_acc is not None and _meaningfully_lower(current_acc, clone_acc):
                decision = "view_gate_warmup_failure"
            elif run_name == str((c2 or {}).get("run_name")) and current_acc is not None and _meaningfully_lower(current_acc, clone_acc):
                decision = "no_internal_gru_path_failure"
            elif ("B3" in run_name or "B4" in run_name) and current_acc is not None and _meaningfully_lower(current_acc, clone_acc):
                decision = "hardening_transform_failure"
            elif "A1" in run_name and current_acc is not None and _meaningfully_lower(current_acc, clone_acc):
                decision = "pilot_estimator_noise_calculation_failure"
        invalid_reason = _row_invalid_reason(item, global_invalid_reason)
        item["full_sweep_status"] = full_status
        item["debug_decision"] = decision
        item["invalid_reason"] = invalid_reason
        item["candidate_eligible"] = _candidate_eligible(item)
        item["hardening_design_failed"] = False if full_status != "valid" else bool(item.get("is_destructive"))
        result.append(item)
    return result


def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        if not _candidate_eligible(row):
            continue
        item = dict(row)
        item["candidate_score"] = _candidate_score_mapping(item)
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (
            bool(item.get("is_slow_high_ceiling")),
            _to_float(item.get("candidate_score")) or float("-inf"),
            _to_float(item.get("E90_ratio")) or float("-inf"),
            -(_to_float(item.get("ceiling_gap_acc")) or float("inf")),
        ),
        reverse=True,
    )


def write_plots_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    plt = _load_pyplot()
    curves = []
    for row in rows:
        epochs = _json_list(row.get("curve_epochs"))
        values = _json_list(row.get("curve_val_acc"))
        if epochs and values:
            curves.append((str(row.get("run_name") or row.get("experiment_name")), epochs, values))
    if len(curves) >= 2:
        if plt is None:
            _write_blank_png(out_dir / "learning_curves.png")
        else:
            plt.figure(figsize=(9, 5))
            for name, epochs, values in curves:
                plt.plot(epochs, values, label=name)
            plt.xlabel("Epoch")
            plt.ylabel("Validation accuracy")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "learning_curves.png", dpi=160)
            plt.close()

    scatter = [
        row
        for row in rows
        if _to_float(row.get("ceiling_gap_acc")) is not None and _to_float(row.get("E90_ratio")) is not None
    ]
    if len(rows) >= 2 and scatter:
        if plt is None:
            _write_blank_png(out_dir / "ceiling_gap_vs_E90_ratio.png")
        else:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                [_to_float(row.get("E90_ratio")) for row in scatter],
                [_to_float(row.get("ceiling_gap_acc")) for row in scatter],
            )
            for row in scatter:
                plt.annotate(
                    str(row.get("run_name")),
                    (_to_float(row.get("E90_ratio")), _to_float(row.get("ceiling_gap_acc"))),
                    fontsize=7,
                )
            plt.axhline(0.03, color="tab:green", linestyle="--", linewidth=1)
            plt.axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
            plt.axvline(1.5, color="tab:green", linestyle="--", linewidth=1)
            plt.xlabel("E90 ratio vs clean")
            plt.ylabel("Ceiling gap accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / "ceiling_gap_vs_E90_ratio.png", dpi=160)
            plt.close()


def _series_from_sources(train_log: dict[str, Any], metrics: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    for key in keys:
        series = _numeric_list(train_log.get(key))
        if series:
            return series
    epoch_logs = train_log.get("epoch_logs")
    if isinstance(epoch_logs, list):
        for key in keys:
            series = _numeric_list([item.get(key) for item in epoch_logs if isinstance(item, dict)])
            if series:
                return series
        for key in keys:
            series = _numeric_list(
                [
                    _nested_metric(item.get("validation_metrics"), key)
                    for item in epoch_logs
                    if isinstance(item, dict)
                ]
            )
            if series:
                return series
    for key in keys:
        series = _numeric_list(metrics.get(key))
        if series:
            return series
    return []


def _nested_metric(metrics: Any, key: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    if key in metrics:
        return metrics[key]
    if key in {"val_acc", "beam/accuracy_val", "accuracy_val", "val_top1"}:
        topk = metrics.get("topk")
        if isinstance(topk, dict):
            values = topk.get("1") or topk.get(1)
            if isinstance(values, list) and values:
                return values[0]
    if key in {"val_adba", "beam/adba_val", "adba_val"}:
        values = metrics.get("dba")
        if isinstance(values, list) and values:
            return sum(float(value) for value in values) / len(values)
    return None


def _epoch_series(train_log: dict[str, Any], length: int) -> list[int]:
    epoch_logs = train_log.get("epoch_logs")
    if isinstance(epoch_logs, list):
        epochs = [int(item.get("epoch", idx + 1)) for idx, item in enumerate(epoch_logs) if isinstance(item, dict)]
        if len(epochs) >= length:
            return epochs[:length]
    return list(range(1, int(length) + 1))


def _last_n_mean(values: list[float], n: int) -> float | None:
    if not values:
        return None
    window = values[-min(int(n), len(values)) :]
    return float(sum(window) / len(window))


def _first_epoch_at_fraction(values: list[float], epochs: list[int], final_acc: float | None, fraction: float) -> int | None:
    if final_acc is None:
        return None
    threshold = float(fraction) * float(final_acc)
    for epoch, value in zip(epochs, values):
        if value >= threshold:
            return int(epoch)
    return None


def _run_metadata(run_dir: Path, final_config: dict[str, Any], train_log: dict[str, Any]) -> dict[str, Any]:
    experiment = final_config.get("experiment") if isinstance(final_config.get("experiment"), dict) else {}
    model = final_config.get("model") if isinstance(final_config.get("model"), dict) else {}
    output = final_config.get("output") if isinstance(final_config.get("output"), dict) else {}
    runtime = train_log.get("runtime") if isinstance(train_log.get("runtime"), dict) else {}
    modalities = (
        model.get("modalities")
        or (model.get("student") or {}).get("modalities")
        or (model.get("teacher") or {}).get("modalities")
        or []
    )
    return {
        "run_name": output.get("run_name") or experiment.get("name") or run_dir.name,
        "experiment_name": experiment.get("name") or run_dir.name,
        "seed": experiment.get("seed"),
        "modalities": [str(item) for item in modalities],
        "runtime_run_dir": runtime.get("run_dir"),
    }


def _find_csi_hardening(config: dict[str, Any]) -> dict[str, Any]:
    dataset = (config.get("data") or {}).get("dataset") if isinstance(config.get("data"), dict) else {}
    if isinstance(dataset, dict) and isinstance(dataset.get("csi_hardening"), dict):
        return dataset["csi_hardening"]
    for role in ("student", "teacher"):
        role_cfg = (config.get("model") or {}).get(role) if isinstance(config.get("model"), dict) else {}
        if not isinstance(role_cfg, dict):
            continue
        encoders = role_cfg.get("encoders")
        csi_cfg = encoders.get("csi") if isinstance(encoders, dict) else {}
        if isinstance(csi_cfg, dict) and isinstance(csi_cfg.get("csi_hardening"), dict):
            return csi_cfg["csi_hardening"]
    return {}


def _find_csi_degradation(config: dict[str, Any]) -> dict[str, Any]:
    dataset = (config.get("data") or {}).get("dataset") if isinstance(config.get("data"), dict) else {}
    value = dataset.get("csi_degradation") if isinstance(dataset, dict) else None
    return value if isinstance(value, dict) else {}


def _find_csi_estimation(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    for role in ("student", "teacher"):
        role_cfg = model.get(role) if isinstance(model, dict) else {}
        encoders = role_cfg.get("encoders") if isinstance(role_cfg, dict) else {}
        csi_cfg = encoders.get("csi") if isinstance(encoders, dict) else {}
        if isinstance(csi_cfg, dict):
            pilot = csi_cfg.get("pilot_estimator") or csi_cfg.get("csi_estimation") or {}
            if isinstance(pilot, dict):
                return pilot
    return {}


def _read_csi_debug_records(run_dir: Path, train_log: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = _read_json_value(run_dir / "csi_first_batch_diagnostics.json")
    if isinstance(artifact, list):
        return [item for item in artifact if isinstance(item, dict)]
    records = train_log.get("csi_first_batch_diagnostics")
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    runtime = train_log.get("runtime") if isinstance(train_log.get("runtime"), dict) else {}
    records = runtime.get("csi_first_batch_diagnostics")
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    return []


def _read_pilot_noise_validity(
    run_dir: Path,
    final_config: dict[str, Any],
    train_log: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact = _read_json(run_dir / "pilot_noise_validity.json")
    if artifact:
        return artifact
    logged = train_log.get("pilot_noise_validity")
    if isinstance(logged, dict):
        return logged
    runtime = train_log.get("runtime") if isinstance(train_log.get("runtime"), dict) else {}
    logged = runtime.get("pilot_noise_validity")
    if isinstance(logged, dict):
        return logged
    return evaluate_pilot_noise_validity(final_config, records)


def _pilot_mode(pilot_cfg: dict[str, Any]) -> str:
    if not isinstance(pilot_cfg, dict):
        return "none"
    enabled = bool(pilot_cfg.get("enabled", pilot_cfg.get("enable", True)))
    if not enabled:
        return "none"
    return str(pilot_cfg.get("mode", "none") or "none").lower()


def _snr_db_from_estimation(pilot_cfg: dict[str, Any]) -> Any:
    if not isinstance(pilot_cfg, dict):
        return None
    return pilot_cfg.get("snr_db") if pilot_cfg.get("snr_db") is not None else pilot_cfg.get("est_snr_db")


def _uses_fixed_pilot_scaling_config(config: dict[str, Any], pilot_validity: dict[str, Any]) -> bool:
    debug_cfg = config.get("debug") if isinstance(config.get("debug"), dict) else {}
    if debug_cfg.get("pilot_scaling_config_version") == FIXED_PILOT_SCALING_VERSION:
        return True
    mode = str(pilot_validity.get("mode", "")).lower()
    if pilot_validity.get("is_mild_pilot_estimation") and mode in {"est_snr", "estimation_snr"}:
        return True
    if mode in {"none", "clean"} and not pilot_validity.get("is_destructive_control"):
        return True
    return False


def _select_clean_reference_row(rows: list[dict[str, Any]], clean_teacher_run: str) -> dict[str, Any]:
    value = str(clean_teacher_run)
    name = Path(value).name
    for row in rows:
        if value in {str(row.get("run_name")), str(row.get("experiment_name")), str(row.get("run_dir"))}:
            return row
    for row in rows:
        if name in {str(row.get("run_name")), str(row.get("experiment_name")), Path(str(row.get("run_dir"))).name}:
            return row
    raise ValueError(f"Clean teacher run '{clean_teacher_run}' was not found in analyzed runs.")


def _candidate_score_mapping(row: dict[str, Any]) -> float:
    ratio = _to_float(row.get("E90_ratio")) or 0.0
    gap = _to_float(row.get("ceiling_gap_acc")) or 0.0
    final_acc = _to_float(row.get("final_acc")) or 0.0
    destructive_penalty = 1.0 if bool(row.get("is_destructive")) else 0.0
    return float(ratio + final_acc - 10.0 * max(gap, 0.0) - destructive_penalty)


def _missing_required_debug_diagnostics(
    rows: list[dict[str, Any]],
    original: dict[str, Any] | None,
    clone: dict[str, Any] | None,
    pilot_disabled: dict[str, Any] | None,
    c1: dict[str, Any] | None,
    c2: dict[str, Any] | None,
) -> bool:
    if any(_legacy_invalid_sweep_marker(row) for row in rows) and not any(_row_bool(row, "has_debug_diagnostics") for row in rows):
        return True
    required = [row for row in (original, clone, pilot_disabled, c1, c2) if row is not None]
    required.extend(row for row in rows if bool(row.get("is_mild_pilot_estimation")))
    if not required:
        return not any(_row_bool(row, "has_debug_diagnostics") for row in rows)
    return any(_row_bool(row, "has_debug_diagnostics") is not True for row in required)


def _row_invalid_reason(row: dict[str, Any], global_invalid_reason: str | None) -> str | None:
    pilot_reason = row.get("pilot_noise_invalid_reason")
    if _has_value(pilot_reason):
        return str(pilot_reason)
    if _legacy_invalid_sweep_marker(row) and _row_bool(row, "uses_fixed_pilot_scaling_config") is not True:
        return "invalid_due_to_legacy_pilot_scaling_config"
    if global_invalid_reason:
        return global_invalid_reason
    return None


def _candidate_eligible(row: dict[str, Any]) -> bool:
    if str(row.get("full_sweep_status") or "") != "valid":
        return False
    if _has_value(row.get("invalid_reason")):
        return False
    if _row_bool(row, "pilot_noise_scale_valid") is False:
        return False
    return True


def _legacy_invalid_sweep_marker(row: dict[str, Any]) -> bool:
    text = str(row.get("run_dir") or "")
    return any(marker in text for marker in LEGACY_INVALID_SWEEP_MARKERS)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def build_analysis_metadata(rows: list[dict[str, Any]], runs_root: Path) -> dict[str, Any]:
    statuses = sorted({str(row.get("full_sweep_status")) for row in rows if row.get("full_sweep_status") is not None})
    non_destructive_rows = [row for row in rows if _row_bool(row, "is_destructive_control") is not True]
    fixed_rows = [row for row in non_destructive_rows if _row_bool(row, "uses_fixed_pilot_scaling_config") is True]
    invalid_rows = [row for row in rows if _has_value(row.get("invalid_reason"))]
    return {
        "runs_root": str(runs_root),
        "change": "fix-csi-pilot-estimation-noise-scaling",
        "pilot_scaling_config_version": FIXED_PILOT_SCALING_VERSION,
        "uses_fixed_pilot_scaling_config": bool(non_destructive_rows) and len(fixed_rows) == len(non_destructive_rows),
        "legacy_invalid_sweep_isolated": any(
            marker in str(runs_root) or any(marker in str(row.get("run_dir") or "") for row in rows)
            for marker in LEGACY_INVALID_SWEEP_MARKERS
        ),
        "full_sweep_statuses": statuses,
        "invalid_run_count": len(invalid_rows),
        "ranked_candidate_input_count": sum(1 for row in rows if _candidate_eligible(row)),
    }


def _find_run_by_tokens(rows: list[dict[str, Any]], tokens: tuple[str, ...]) -> dict[str, Any] | None:
    for row in rows:
        name = str(row.get("run_name") or row.get("experiment_name") or "")
        if any(token in name for token in tokens):
            return row
    return None


def _row_bool(row: dict[str, Any] | None, key: str) -> bool | None:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "none", "nan"}


def _is_high(value: float) -> bool:
    return float(value) >= 0.30


def _is_low(value: float) -> bool:
    return float(value) <= 0.20


def _meaningfully_lower(value: float, reference: float) -> bool:
    return float(reference) - float(value) > 0.05


def _mapping_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _read_json_value(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _read_metrics_artifact(run_dir: Path) -> dict[str, Any]:
    for name in ("metrics.json", "test_report.json"):
        metrics = _read_json(run_dir / name)
        if metrics:
            if isinstance(metrics.get("metrics"), dict):
                return metrics["metrics"]
            return metrics
    return {}


def _numeric_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        numeric = _to_float(item)
        if numeric is not None:
            result.append(numeric)
    return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_list(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _numeric_list(decoded)
    return _numeric_list(value)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _write_blank_png(path: Path) -> None:
    width, height = 1, 1
    raw = b"\x00\xff\xff\xff"
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
