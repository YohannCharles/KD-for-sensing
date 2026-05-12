from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kd_sensing.config.io import deep_merge, safe_load_yaml
from kd_sensing.diagnostics.conditional_utility import (
    WEAK_MODALITIES,
    diagnose_modalities,
    read_table,
    write_json,
)
from kd_sensing.modalities import normalize_modalities
from kd_sensing.utils.paths import project_root, resolve_path


DEFAULT_BOOTSTRAP = {
    "num_bootstrap": 1000,
    "confidence": 0.95,
    "random_seed": 0,
    "cluster_key_preference": ["seq_id", "sample_id", "dataset_index"],
}

DEFAULT_THRESHOLDS = {
    "global_delta_dba": 0.001,
    "global_delta_ce": 0.0,
    "conditional_delta_dba": 0.02,
    "conditional_delta_ce": 0.02,
    "teacher_rescue_rate": 0.10,
    "oracle_gain_dba": 0.02,
    "min_bucket_samples": 20,
}

DEFAULT_BASELINE_SUBSETS = {
    "strong_only": {
        "slug": "gps_mmwave",
        "modalities": ["gps", "mmwave"],
        "config": "configs/fusion/gps_mmwave_teacher_no_kd.yaml",
    },
    "strong_plus_image": {
        "slug": "image_gps_mmwave",
        "modalities": ["image", "gps", "mmwave"],
        "config": "configs/fusion/image_gps_mmwave_teacher_no_kd.yaml",
    },
    "strong_plus_radar": {
        "slug": "radar_gps_mmwave",
        "modalities": ["radar", "gps", "mmwave"],
        "config": "configs/fusion/radar_gps_mmwave_teacher_no_kd.yaml",
    },
    "strong_plus_lidar": {
        "slug": "gps_lidar_mmwave",
        "modalities": ["gps", "lidar", "mmwave"],
        "config": "configs/fusion/gps_lidar_mmwave_teacher_no_kd.yaml",
    },
    "all": {
        "slug": "image_radar_gps_lidar_mmwave",
        "modalities": ["image", "radar", "gps", "lidar", "mmwave"],
        "config": "configs/fusion/image_radar_gps_lidar_mmwave_teacher_no_kd.yaml",
    },
}

DEFAULT_MANIFEST = {
    "output_dir": "outputs/scene32/phase_1_5_utility_validation",
    "conditional_utility_input": "outputs/scene32/marf/conditional_utility",
    "bootstrap": DEFAULT_BOOTSTRAP,
    "thresholds": DEFAULT_THRESHOLDS,
    "checkpoint_matrix": {
        "config": "configs/analysis/marf_conditional_utility_audit.yaml",
        "run_name": "marf",
        "checkpoints_dir": "outputs/scene32/marf/checkpoints",
        "roles": {
            "best_top1": {
                "checkpoint": "best_top1.pth",
                "audit_dir": "outputs/scene32/marf/conditional_utility",
            },
            "best": {
                "checkpoint": "best.pth",
                "audit_dir": "outputs/scene32/phase_1_5_utility_validation/checkpoint_audits/marf_best",
            },
            "last": {
                "checkpoint": "last.pth",
                "audit_dir": "outputs/scene32/phase_1_5_utility_validation/checkpoint_audits/marf_last",
            },
        },
    },
    "baseline_matrix": {
        "seeds": [0, 1, 2],
        "primary_baseline": "strong_only",
        "subsets": DEFAULT_BASELINE_SUBSETS,
        "run_name_template": "phase_1_5_{slug}_seed{seed}",
        "training_overrides": {},
    },
}


def load_phase_1_5_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = deepcopy(DEFAULT_MANIFEST)
    if path is None:
        return _normalize_manifest(manifest)
    target = resolve_path(path)
    if target is None:
        return _normalize_manifest(manifest)
    with target.open("r", encoding="utf-8") as f:
        loaded = safe_load_yaml(f.read()) or {}
    manifest = deep_merge(manifest, loaded)
    manifest["_manifest_path"] = str(target)
    return _normalize_manifest(manifest)


def run_phase_1_5_utility_validation(
    manifest_or_path: str | Path | dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    num_bootstrap: int | None = None,
) -> dict[str, Any]:
    manifest = (
        _normalize_manifest(deepcopy(manifest_or_path))
        if isinstance(manifest_or_path, dict)
        else load_phase_1_5_manifest(manifest_or_path)
    )
    if output_dir is not None:
        manifest["output_dir"] = str(resolve_path(output_dir))
    if num_bootstrap is not None:
        manifest.setdefault("bootstrap", {})["num_bootstrap"] = int(num_bootstrap)
    output_path = Path(manifest["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    audit_dir = Path(manifest["conditional_utility_input"])
    subset = _read_optional_table(audit_dir, "subset_predictions")
    delta = _read_optional_table(audit_dir, "conditional_utility_per_sample_delta")
    bootstrap_ci = compute_bootstrap_confidence(
        subset,
        delta,
        bootstrap_cfg=manifest.get("bootstrap", {}),
    )
    bootstrap_path = output_path / "conditional_utility_bootstrap_ci.csv"
    bootstrap_ci.to_csv(bootstrap_path, index=False)

    checkpoint_rows, audit_commands = build_checkpoint_matrix(manifest)
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    checkpoint_path = output_path / "checkpoint_comparison.csv"
    checkpoint_frame.to_csv(checkpoint_path, index=False)
    commands_path = output_path / "checkpoint_audit_commands.sh"
    _write_commands(commands_path, audit_commands)

    baseline_rows, baseline_commands = build_baseline_matrix(manifest)
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_manifest_path = output_path / "baseline_manifest.csv"
    baseline_frame.to_csv(baseline_manifest_path, index=False)
    baseline_commands_path = output_path / "baseline_training_commands.sh"
    _write_commands(baseline_commands_path, baseline_commands)
    baseline_summary = summarize_baseline_metrics(baseline_frame)
    baseline_summary_path = output_path / "fixed_subset_baseline_summary.csv"
    baseline_summary.to_csv(baseline_summary_path, index=False)

    bucket = _read_csv_if_exists(audit_dir / "conditional_utility_by_bucket.csv")
    teacher_summary = _read_json_if_exists(audit_dir / "teacher_complementarity_summary.json")
    phase_summary = build_phase_1_5_summary(
        manifest=manifest,
        bootstrap_ci=bootstrap_ci,
        checkpoint_frame=checkpoint_frame,
        baseline_frame=baseline_frame,
        baseline_summary=baseline_summary,
        bucket_summary=bucket,
        teacher_summary=teacher_summary,
        outputs={
            "bootstrap_ci": str(bootstrap_path),
            "checkpoint_comparison": str(checkpoint_path),
            "checkpoint_audit_commands": str(commands_path),
            "baseline_manifest": str(baseline_manifest_path),
            "baseline_training_commands": str(baseline_commands_path),
            "fixed_subset_baseline_summary": str(baseline_summary_path),
        },
    )
    summary_path = write_json(phase_summary, output_path / "phase_1_5_summary.json")
    report_path = write_phase_1_5_report(phase_summary, output_path / "phase_1_5_report.md")

    metadata = {
        "manifest": manifest,
        "input_status": _input_status(manifest, subset, delta, bucket),
        "outputs": {**phase_summary["outputs"], "summary": str(summary_path), "report": str(report_path)},
    }
    metadata_path = write_json(metadata, output_path / "phase_1_5_metadata.json")
    return {
        "output_dir": str(output_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "metadata": str(metadata_path),
        "bootstrap_ci": str(bootstrap_path),
        "checkpoint_comparison": str(checkpoint_path),
        "baseline_summary": str(baseline_summary_path),
        "decision": phase_summary.get("decision", {}),
    }


def compute_paired_delta_frame(
    subset_predictions: pd.DataFrame,
    marginal_deltas: pd.DataFrame,
    *,
    baseline_subset: str = "strong_only",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    delta = marginal_deltas.copy()
    if not delta.empty:
        for weak, group in delta.groupby("weak_modality", sort=True):
            item = group.copy()
            item["comparison"] = f"strong_plus_{weak}_vs_{baseline_subset}"
            item["comparison_subset"] = item.get("strong_plus_subset", f"strong_plus_{weak}")
            item["delta_loss"] = -item["delta_ce"]
            rows.append(item)
    subset = subset_predictions.copy()
    if not subset.empty and {"subset_name", "sample_id", "dataset_index", "horizon_idx", "horizon_name"}.issubset(
        subset.columns
    ):
        keys = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
        extra_keys = [name for name in ("seq_id",) if name in subset.columns]
        base = subset[(subset["subset_name"] == baseline_subset) & _valid_mask(subset)].copy()
        comp = subset[(subset["subset_name"] == "all") & _valid_mask(subset)].copy()
        if not base.empty and not comp.empty:
            cols = keys + extra_keys + ["gt_beam", "ce", "top1_hit", "top3_hit", "dba_score"]
            base_cols = [col for col in cols if col in base.columns]
            comp_cols = [col for col in cols if col in comp.columns]
            base = base[base_cols].rename(
                columns={
                    "ce": "ce_strong_only",
                    "top1_hit": "strong_only_top1",
                    "top3_hit": "strong_only_top3",
                    "dba_score": "strong_only_dba",
                }
            )
            comp = comp[comp_cols].rename(
                columns={
                    "ce": "ce_strong_plus",
                    "top1_hit": "strong_plus_top1",
                    "top3_hit": "strong_plus_top3",
                    "dba_score": "strong_plus_dba",
                }
            )
            merge_keys = [key for key in keys + extra_keys + ["gt_beam"] if key in base.columns and key in comp.columns]
            merged = base.merge(comp, on=merge_keys, how="inner")
            if not merged.empty:
                merged["weak_modality"] = "all"
                merged["strong_plus_subset"] = "all"
                merged["comparison_subset"] = "all"
                merged["comparison"] = f"all_vs_{baseline_subset}"
                merged["delta_ce"] = merged["ce_strong_only"] - merged["ce_strong_plus"]
                merged["delta_top1"] = merged["strong_plus_top1"] - merged["strong_only_top1"]
                merged["delta_top3"] = merged["strong_plus_top3"] - merged["strong_only_top3"]
                merged["delta_dba"] = merged["strong_plus_dba"] - merged["strong_only_dba"]
                merged["delta_loss"] = -merged["delta_ce"]
                rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def compute_bootstrap_confidence(
    subset_predictions: pd.DataFrame,
    marginal_deltas: pd.DataFrame,
    *,
    bootstrap_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = {**DEFAULT_BOOTSTRAP, **(bootstrap_cfg or {})}
    paired = compute_paired_delta_frame(subset_predictions, marginal_deltas)
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "comparison",
                "weak_modality",
                "metric",
                "horizon_name",
                "mean_delta",
                "ci_lower",
                "ci_upper",
                "positive_rate",
                "num_bootstrap",
                "num_samples",
                "num_clusters",
                "cluster_key",
                "cluster_key_status",
            ]
        )
    cluster_key, cluster_status = choose_cluster_key(
        paired,
        cfg.get("cluster_key_preference", DEFAULT_BOOTSTRAP["cluster_key_preference"]),
    )
    rows: list[dict[str, Any]] = []
    metrics = ["delta_top1", "delta_top3", "delta_dba", "delta_ce"]
    for (comparison, weak), group in paired.groupby(["comparison", "weak_modality"], sort=True):
        slices: list[tuple[str, pd.DataFrame]] = [("overall", group)]
        for horizon_name, horizon_group in group.groupby("horizon_name", sort=True):
            slices.append((str(horizon_name), horizon_group))
        for horizon_name, horizon_group in slices:
            for metric in metrics:
                if metric not in horizon_group.columns:
                    continue
                stat = cluster_bootstrap_mean(
                    horizon_group,
                    metric,
                    cluster_key=cluster_key,
                    num_bootstrap=int(cfg.get("num_bootstrap", 1000)),
                    confidence=float(cfg.get("confidence", 0.95)),
                    random_seed=int(cfg.get("random_seed", 0)),
                )
                rows.append(
                    {
                        "comparison": str(comparison),
                        "weak_modality": str(weak),
                        "metric": metric,
                        "horizon_name": horizon_name,
                        "mean_delta": stat["mean_delta"],
                        "ci_lower": stat["ci_lower"],
                        "ci_upper": stat["ci_upper"],
                        "positive_rate": stat["positive_rate"],
                        "num_bootstrap": stat["num_bootstrap"],
                        "num_samples": stat["num_samples"],
                        "num_clusters": stat["num_clusters"],
                        "cluster_key": cluster_key,
                        "cluster_key_status": cluster_status,
                    }
                )
    return pd.DataFrame(rows)


def choose_cluster_key(frame: pd.DataFrame, preferences: Iterable[str]) -> tuple[str, str]:
    for index, key in enumerate(preferences):
        if key in frame.columns and frame[key].notna().any():
            return str(key), "preferred" if index == 0 else "fallback"
    frame["__row_cluster"] = np.arange(len(frame))
    return "__row_cluster", "row_fallback"


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    metric: str,
    *,
    cluster_key: str,
    num_bootstrap: int,
    confidence: float,
    random_seed: int,
) -> dict[str, Any]:
    work = frame[[cluster_key, metric]].dropna().copy()
    if work.empty:
        return {
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "positive_rate": 0.0,
            "num_bootstrap": int(num_bootstrap),
            "num_samples": 0,
            "num_clusters": 0,
        }
    clusters = work[cluster_key].dropna().unique()
    cluster_values = [work.loc[work[cluster_key] == cluster, metric].astype(float).to_numpy() for cluster in clusters]
    values = work[metric].astype(float).to_numpy()
    mean_delta = float(np.mean(values))
    rng = np.random.default_rng(random_seed)
    boot_means = []
    for _ in range(int(num_bootstrap)):
        indices = rng.integers(0, len(cluster_values), size=len(cluster_values))
        sample = np.concatenate([cluster_values[int(idx)] for idx in indices])
        boot_means.append(float(np.mean(sample)) if len(sample) else 0.0)
    alpha = max(0.0, min(1.0, 1.0 - float(confidence)))
    lower, upper = np.quantile(np.asarray(boot_means), [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_delta": mean_delta,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "positive_rate": float(np.mean(values > 0)),
        "num_bootstrap": int(num_bootstrap),
        "num_samples": int(len(work)),
        "num_clusters": int(len(cluster_values)),
    }


def build_checkpoint_matrix(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    cfg = manifest.get("checkpoint_matrix", {})
    checkpoints_dir = Path(cfg.get("checkpoints_dir", ""))
    config_path = str(cfg.get("config", "configs/analysis/marf_conditional_utility_audit.yaml"))
    rows = []
    commands = []
    for role, role_cfg in (cfg.get("roles") or {}).items():
        checkpoint = _resolve_checkpoint_path(checkpoints_dir, role, role_cfg)
        audit_dir = Path(str(role_cfg.get("audit_dir", "")))
        summary_path = audit_dir / "conditional_utility_summary.json"
        status = "complete" if summary_path.exists() else "pending"
        if not checkpoint.exists():
            status = "missing"
        command = _checkpoint_audit_command(config_path, checkpoint, audit_dir)
        if status == "pending":
            commands.append(command)
        summary = _read_json_if_exists(summary_path)
        rows.append(
            {
                "role": str(role),
                "checkpoint": str(checkpoint),
                "checkpoint_exists": bool(checkpoint.exists()),
                "audit_dir": str(audit_dir),
                "summary_path": str(summary_path),
                "status": status,
                "command": command,
                "strong_only_top1_avg": _summary_subset_metric(summary, "strong_only", "top1"),
                "strong_only_dba_avg": _summary_subset_metric(summary, "strong_only", "dba"),
                "all_top1_avg": _summary_subset_metric(summary, "all", "top1"),
                "all_dba_avg": _summary_subset_metric(summary, "all", "dba"),
                "oracle_delta_dba": _summary_oracle_delta(summary, "delta_dba"),
                "diagnosis_labels": _diagnosis_label_text(summary),
            }
        )
    return rows, commands


def build_baseline_matrix(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    cfg = manifest.get("baseline_matrix", {})
    seeds = [int(seed) for seed in cfg.get("seeds", [0, 1, 2])]
    subsets = cfg.get("subsets") or DEFAULT_BASELINE_SUBSETS
    run_template = str(cfg.get("run_name_template", "phase_1_5_{slug}_seed{seed}"))
    training_overrides = cfg.get("training_overrides") or {}
    rows = []
    commands = []
    for subset_name, subset_cfg in subsets.items():
        modalities = normalize_modalities(tuple(subset_cfg["modalities"]), context=f"phase 1.5 subset {subset_name}")
        slug = str(subset_cfg.get("slug") or "_".join(modalities))
        config_path = str(subset_cfg.get("config") or f"configs/fusion/{slug}_teacher_no_kd.yaml")
        for seed in seeds:
            run_name = run_template.format(subset=subset_name, slug=slug, seed=seed)
            metrics_path = resolve_path(Path("outputs") / "scene32" / run_name / "metrics.json") or (
                Path("outputs") / "scene32" / run_name / "metrics.json"
            )
            command = _baseline_train_command(
                config_path,
                seed=seed,
                run_name=run_name,
                extra_overrides=training_overrides,
            )
            status = "complete" if metrics_path.exists() else "pending"
            if status == "pending":
                commands.append(command)
            metrics = _read_json_if_exists(metrics_path)
            row = {
                "subset": str(subset_name),
                "slug": slug,
                "modalities": ",".join(modalities),
                "seed": seed,
                "config": config_path,
                "run_name": run_name,
                "metrics_path": str(metrics_path),
                "status": status,
                "command": command,
            }
            row.update(_flatten_metrics(metrics))
            rows.append(row)
    return rows, commands


def summarize_baseline_metrics(baseline_frame: pd.DataFrame) -> pd.DataFrame:
    if baseline_frame.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in baseline_frame.columns
        if col.startswith(("top1_", "top3_", "dba_", "loss"))
        and pd.api.types.is_numeric_dtype(baseline_frame[col])
    ]
    rows = []
    for subset, group in baseline_frame.groupby("subset", sort=True):
        complete = group[group["status"] == "complete"]
        row = {
            "subset": str(subset),
            "status": "complete" if len(complete) == len(group) and len(group) > 0 else "pending",
            "num_seeds": int(len(group)),
            "complete_seeds": int(len(complete)),
        }
        for col in metric_cols:
            values = complete[col].dropna().astype(float)
            row[f"{col}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_phase_1_5_summary(
    *,
    manifest: dict[str, Any],
    bootstrap_ci: pd.DataFrame,
    checkpoint_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    teacher_summary: dict[str, Any],
    outputs: dict[str, str],
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(manifest.get("thresholds") or {})}
    baseline_decision = _baseline_decision(baseline_summary, thresholds, manifest)
    bootstrap_decision = _bootstrap_decision(bootstrap_ci, thresholds)
    checkpoint_status = (
        "complete"
        if not checkpoint_frame.empty and (checkpoint_frame["status"] == "complete").all()
        else "pending"
    )
    final_status = "pending"
    label = "pending"
    recommendation = "等待 dedicated fixed-subset baseline 和 checkpoint matrix 补齐后再给最终路线结论。"
    if (
        bootstrap_decision["status"] == "complete"
        and checkpoint_status == "complete"
        and baseline_decision["status"] == "complete"
    ):
        final_status = "complete"
        if not baseline_decision["has_stable_gain"] and not bootstrap_decision["has_significant_global_gain"]:
            label = "low_weak_utility"
            recommendation = "Scene32 clean setting 优先转向 strong-path 精度、safe fusion 和 degraded robustness。"
        else:
            label = "conditionally_useful"
            recommendation = "弱模态存在可复核收益，下一步可以进入 MARF-Comm 条件效用 router 设计。"
    diagnosis = diagnose_modalities(
        _marginal_from_bootstrap(bootstrap_ci),
        bucket_summary,
        teacher_summary,
        thresholds=thresholds,
        bootstrap_confidence=bootstrap_ci,
    )
    consistency = _checkpoint_consistency(checkpoint_frame)
    return {
        "manifest": manifest,
        "outputs": outputs,
        "thresholds": thresholds,
        "bootstrap": {
            "status": bootstrap_decision["status"],
            "cluster_key": _first_nonempty(bootstrap_ci, "cluster_key"),
            "cluster_key_status": _first_nonempty(bootstrap_ci, "cluster_key_status"),
            "num_rows": int(len(bootstrap_ci)),
            "decision": bootstrap_decision,
            "key_ci": _key_ci_records(bootstrap_ci),
        },
        "checkpoint_matrix": {
            "status": checkpoint_status,
            "num_roles": int(len(checkpoint_frame)),
            "complete_roles": int((checkpoint_frame.get("status") == "complete").sum()) if not checkpoint_frame.empty else 0,
            "weak_utility_consistency": consistency,
        },
        "baseline_matrix": baseline_decision,
        "diagnosis": diagnosis,
        "decision": {
            "status": final_status,
            "label": label,
            "recommendation": recommendation,
            "evidence_level": "final" if final_status == "complete" else "exploratory",
        },
    }


def write_phase_1_5_report(summary: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    decision = summary.get("decision", {})
    lines = [
        "# Phase 1.5 Utility Validation Report",
        "",
        f"- Status: {decision.get('status', 'pending')}",
        f"- Label: {decision.get('label', 'pending')}",
        f"- Evidence level: {decision.get('evidence_level', 'exploratory')}",
        f"- Recommendation: {decision.get('recommendation', '')}",
        "",
        "## Bootstrap",
        "",
        f"- Status: {summary.get('bootstrap', {}).get('status', 'pending')}",
        f"- Cluster key: {summary.get('bootstrap', {}).get('cluster_key')} ({summary.get('bootstrap', {}).get('cluster_key_status')})",
        f"- Significant global gain: {summary.get('bootstrap', {}).get('decision', {}).get('has_significant_global_gain')}",
        "",
        "## Baselines",
        "",
        f"- Status: {summary.get('baseline_matrix', {}).get('status', 'pending')}",
        f"- Primary baseline: {summary.get('baseline_matrix', {}).get('primary_baseline')}",
        f"- Stable gain: {summary.get('baseline_matrix', {}).get('has_stable_gain')}",
        "",
        "## Checkpoints",
        "",
        f"- Status: {summary.get('checkpoint_matrix', {}).get('status', 'pending')}",
        f"- Consistency: {summary.get('checkpoint_matrix', {}).get('weak_utility_consistency')}",
        "",
        "## Outputs",
        "",
    ]
    for name, value in (summary.get("outputs") or {}).items():
        lines.append(f"- {name}: `{value}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _normalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    result = deepcopy(manifest)
    result["output_dir"] = str(resolve_path(result.get("output_dir"), root))
    result["conditional_utility_input"] = str(resolve_path(result.get("conditional_utility_input"), root))
    ckpt = result.setdefault("checkpoint_matrix", {})
    if "config" in ckpt:
        ckpt["config"] = str(resolve_path(ckpt["config"], root))
    if "checkpoints_dir" in ckpt:
        ckpt["checkpoints_dir"] = str(resolve_path(ckpt["checkpoints_dir"], root))
    for role_cfg in (ckpt.get("roles") or {}).values():
        if "audit_dir" in role_cfg:
            role_cfg["audit_dir"] = str(resolve_path(role_cfg["audit_dir"], root))
        if "checkpoint" in role_cfg and ("/" in str(role_cfg["checkpoint"]) or str(role_cfg["checkpoint"]).startswith(".")):
            role_cfg["checkpoint"] = str(resolve_path(role_cfg["checkpoint"], root))
    baseline = result.setdefault("baseline_matrix", {})
    for subset_cfg in (baseline.get("subsets") or {}).values():
        if "config" in subset_cfg:
            subset_cfg["config"] = _project_relative_or_absolute(subset_cfg["config"], root)
    return result


def _project_relative_or_absolute(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(candidate) if not (root / candidate).exists() else str((root / candidate).resolve())


def _read_optional_table(audit_dir: Path, stem: str) -> pd.DataFrame:
    try:
        return read_table(audit_dir, stem)
    except FileNotFoundError:
        return pd.DataFrame()


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "valid" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["valid"].astype(bool)


def _resolve_checkpoint_path(checkpoints_dir: Path, role: str, role_cfg: dict[str, Any]) -> Path:
    raw = role_cfg.get("checkpoint")
    if raw:
        candidate = Path(str(raw))
        if candidate.is_absolute():
            return candidate
        if candidate.parent != Path("."):
            return resolve_path(candidate) or candidate
        return checkpoints_dir / candidate
    filename = "best_dba.pth" if role == "best_dba" else f"{role}.pth"
    return checkpoints_dir / filename


def _checkpoint_audit_command(config_path: str, checkpoint: Path, audit_dir: Path) -> str:
    return (
        "conda run -n kd_mm_beam python tools/analysis/run_conditional_utility_audit.py "
        f"--config {config_path} --weights {checkpoint} --output-dir {audit_dir}"
    )


def _baseline_train_command(
    config_path: str,
    *,
    seed: int,
    run_name: str,
    extra_overrides: dict[str, Any],
) -> str:
    overrides = {
        "experiment.seed": seed,
        "experiment.name": run_name,
        "output.run_name": run_name,
        "output.overwrite": False,
        **_flatten_override_dict(extra_overrides),
    }
    override_text = " ".join(f"-o {key}={json.dumps(value) if isinstance(value, (list, dict)) else value}" for key, value in overrides.items())
    return f"conda run -n kd_mm_beam python scripts/train.py --config {config_path} {override_text}".strip()


def _flatten_override_dict(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_override_dict(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def _write_commands(path: Path, commands: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "#!/usr/bin/env bash\nset -euo pipefail\n"
    if commands:
        text += "\n".join(commands) + "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    if not metrics:
        return {}
    row = {"loss": _float_or_nan(metrics.get("loss"))}
    topk = metrics.get("topk") or {}
    for metric_name, raw_values in (("top1", topk.get("1")), ("top3", topk.get("3")), ("dba", metrics.get("dba"))):
        values = list(raw_values or [])
        for idx, value in enumerate(values[:3], start=1):
            row[f"{metric_name}_t{idx}"] = _float_or_nan(value)
        row[f"{metric_name}_avg"] = float(np.nanmean(values)) if values else np.nan
    return row


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _summary_subset_metric(summary: dict[str, Any], subset: str, metric: str) -> float:
    aggregate = summary.get("aggregate_metrics", {}).get(subset, {}) if isinstance(summary, dict) else {}
    if metric == "top1":
        values = aggregate.get("topk", {}).get("1", [])
    elif metric == "top3":
        values = aggregate.get("topk", {}).get("3", [])
    else:
        values = aggregate.get(metric, [])
    return float(np.nanmean(values)) if values else np.nan


def _summary_oracle_delta(summary: dict[str, Any], metric: str) -> float:
    value = summary.get("oracle_subset", {}).get("oracle_gain_vs_strong_only", {}).get(metric)
    return _float_or_nan(value)


def _diagnosis_label_text(summary: dict[str, Any]) -> str:
    diagnosis = summary.get("diagnosis") or {}
    if not isinstance(diagnosis, dict):
        return ""
    return ",".join(f"{name}:{item.get('label')}" for name, item in sorted(diagnosis.items()))


def _baseline_decision(
    baseline_summary: pd.DataFrame,
    thresholds: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    primary = manifest.get("baseline_matrix", {}).get("primary_baseline", "strong_only")
    if baseline_summary.empty:
        return {"status": "pending", "primary_baseline": primary, "has_stable_gain": False, "comparisons": []}
    if not (baseline_summary["status"] == "complete").all():
        return {
            "status": "pending",
            "primary_baseline": primary,
            "has_stable_gain": False,
            "comparisons": _baseline_comparisons(baseline_summary, primary, thresholds),
        }
    comparisons = _baseline_comparisons(baseline_summary, primary, thresholds)
    return {
        "status": "complete",
        "primary_baseline": primary,
        "has_stable_gain": any(item.get("stable_gain", False) for item in comparisons),
        "comparisons": comparisons,
    }


def _baseline_comparisons(
    baseline_summary: pd.DataFrame,
    primary: str,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if baseline_summary.empty or primary not in set(baseline_summary["subset"]):
        return []
    base = baseline_summary[baseline_summary["subset"] == primary].iloc[0]
    base_dba = _float_or_nan(base.get("dba_avg_mean"))
    threshold = float(thresholds.get("global_delta_dba", 0.001))
    rows = []
    for _, row in baseline_summary.iterrows():
        subset = str(row["subset"])
        if subset == primary:
            continue
        delta = _float_or_nan(row.get("dba_avg_mean")) - base_dba
        rows.append(
            {
                "subset": subset,
                "metric": "dba_avg",
                "mean_delta": delta,
                "threshold": threshold,
                "stable_gain": bool(np.isfinite(delta) and delta >= threshold and row.get("status") == "complete"),
            }
        )
    return rows


def _bootstrap_decision(bootstrap_ci: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    if bootstrap_ci.empty:
        return {"status": "pending", "has_significant_global_gain": False, "comparisons": []}
    rows = []
    threshold = float(thresholds.get("global_delta_dba", 0.001))
    overall = bootstrap_ci[(bootstrap_ci["metric"] == "delta_dba") & (bootstrap_ci["horizon_name"] == "overall")]
    for _, row in overall.iterrows():
        significant = float(row["mean_delta"]) >= threshold and float(row["ci_lower"]) > 0.0
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "mean_delta": float(row["mean_delta"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
                "threshold": threshold,
                "significant": bool(significant),
            }
        )
    return {
        "status": "complete",
        "has_significant_global_gain": any(item["significant"] for item in rows),
        "comparisons": rows,
    }


def _marginal_from_bootstrap(bootstrap_ci: pd.DataFrame) -> dict[str, Any]:
    if bootstrap_ci.empty:
        return {}
    overall = bootstrap_ci[bootstrap_ci["horizon_name"] == "overall"]
    result: dict[str, Any] = {}
    for weak, group in overall.groupby("weak_modality", sort=True):
        if weak == "all":
            continue
        result[str(weak)] = {}
        for metric, metric_group in group.groupby("metric", sort=True):
            row = metric_group.iloc[0]
            result[str(weak)][str(metric)] = float(row["mean_delta"])
            if str(metric).startswith("delta_"):
                result[str(weak)][str(metric)] = float(row["mean_delta"])
    return result


def _key_ci_records(bootstrap_ci: pd.DataFrame) -> list[dict[str, Any]]:
    if bootstrap_ci.empty:
        return []
    key = bootstrap_ci[
        (bootstrap_ci["horizon_name"] == "overall") & (bootstrap_ci["metric"].isin(["delta_dba", "delta_ce"]))
    ].copy()
    return key.to_dict(orient="records")


def _checkpoint_consistency(frame: pd.DataFrame) -> str:
    if frame.empty or not (frame["status"] == "complete").any():
        return "pending"
    labels = []
    for text in frame.loc[frame["status"] == "complete", "diagnosis_labels"].dropna().tolist():
        labels.append(str(text))
    return "consistent" if len(set(labels)) <= 1 and labels else "mixed"


def _first_nonempty(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna().tolist()
    return values[0] if values else None


def _input_status(manifest: dict[str, Any], subset: pd.DataFrame, delta: pd.DataFrame, bucket: pd.DataFrame) -> dict[str, Any]:
    return {
        "conditional_utility_input": manifest.get("conditional_utility_input"),
        "subset_predictions": {"status": "complete" if not subset.empty else "missing", "num_rows": int(len(subset))},
        "conditional_utility_per_sample_delta": {
            "status": "complete" if not delta.empty else "missing",
            "num_rows": int(len(delta)),
        },
        "conditional_utility_by_bucket": {"status": "complete" if not bucket.empty else "missing", "num_rows": int(len(bucket))},
    }


__all__ = [
    "DEFAULT_BASELINE_SUBSETS",
    "DEFAULT_MANIFEST",
    "build_baseline_matrix",
    "build_checkpoint_matrix",
    "build_phase_1_5_summary",
    "choose_cluster_key",
    "cluster_bootstrap_mean",
    "compute_bootstrap_confidence",
    "compute_paired_delta_frame",
    "load_phase_1_5_manifest",
    "run_phase_1_5_utility_validation",
    "summarize_baseline_metrics",
    "write_phase_1_5_report",
]
