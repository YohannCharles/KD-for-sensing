import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

import yaml

from kd_sensing.models.architecture_summary import CANDIDATE_PARAMETER_SOURCE, summarize_sweep_candidate


SWEEP_NAME = "cnn_hybrid_jepa_visual_prior_sweep"
DEFAULT_MANIFEST = Path("configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep")
OUTPUT_ROOT_MARKER = ".cnn_hybrid_jepa_visual_prior_sweep_output_root"
CONDA_PREFIX = "conda run -n kd_mm_beam "

BASE_PRETRAINING_CONFIG = Path("configs/pretraining/deepsense6g_gps_conditioned_jepa_smoke.yaml")
BASE_JEPA_DOWNSTREAM_CONFIG = Path(
    "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml"
)
BASE_SUPERVISED_IMAGE_GPS_CONFIG = Path(
    "configs/fusion/experiments/jepa_image_gps/image_gps_supervised_beambench_fair_lowmem.yaml"
)
BASE_GPS_ONLY_CONFIG = Path("configs/gps/strong.yaml")

REQUIRED_FAMILIES = {
    "existing_controls",
    "patch_resolution_stage1",
    "overlap_stage1",
    "cnn_supervised_tokens",
    "cnn_jepa_tokens",
    "tinyvit_jepa_encoders",
    "hybrid_tokenizers",
    "pooler_core_ablation",
    "teacher_guided_stabilization",
    "compute_controls",
    "seed_confirm",
}

SCREENING_VARIANT_IDS = (
    "gps_only_control",
    "patch16_mean_baseline",
    "patch16_gps_query_pool",
    "patch14_stage1_gps_query",
    "patch16_resolution_stage1_gps_query",
    "patch14_resolution_stage1_gps_query",
    "patch12_resolution_stage1_gps_query",
    "overlap_k16_s8_resolution_stage1",
    "overlap_k12_s6_resolution_stage1",
    "resnet18_layer4_scratch_full_ft_supervised",
    "resnet18_layer4_imagenet_full_ft_supervised",
    "resnet18_layer3_layer4_scratch_full_ft_supervised",
    "resnet34_layer4_imagenet_full_ft_supervised",
    "resnet18_layer4_scratch_jepa_stage1",
    "resnet18_layer4_imagenet_jepa_stage1",
    "resnet18_layer3_layer4_scratch_jepa_stage1",
    "resnet34_layer4_imagenet_jepa_stage1",
    "tinyvit_5m_scratch_jepa_stage1",
    "tinyvit_5m_22k_jepa_stage1",
    "tinyvit_11m_scratch_jepa_stage1",
    "tinyvit_11m_22k_jepa_stage1",
    "conv_stem_s16_stage1",
    "conv_stem_s8_stage1",
    "local_patch14_stage1",
    "cvt_patch14_stage1",
    "pooler_mean",
    "pooler_gps_query_k2_frame",
    "pooler_gps_query_k2_tokens",
    "pooler_token_aware_core",
    "matched_trainable_params",
    "matched_token_count",
)

STRICT_COMPARABILITY_FIELDS = (
    "split",
    "scene_set",
    "seed",
    "history_window",
    "gps_input_source_window",
    "prediction_horizon",
    "beam_label_space",
    "metric_profile",
    "distance_metric",
    "normalization_artifact",
    "difficulty_digest",
    "output_root",
)

CANDIDATE_REQUIRED_FIELDS = (
    "variant_id",
    "family",
    "stage_plan",
    "checkpoint_policy",
    "checkpoint_selection",
    "availability",
    "pooler",
    "token_metadata",
    "params_metadata",
    "strict_comparable",
    "strict_comparability",
    "metrics_path_template",
)

JOB_FIELDS = (
    "job_id",
    "variant_id",
    "run_id",
    "family",
    "checkpoint_selection",
    "stage",
    "depends_on",
    "command",
    "output_dir",
    "metrics_path",
    "log_path",
    "config_path",
    "availability",
    "success_marker",
    "skip_reason",
    "gpu_policy",
    "max_parallel_group",
)

SUMMARY_FIELDS = (
    "variant_id",
    "run_id",
    "family",
    "seed",
    "checkpoint_selection",
    "status",
    "availability",
    "stage_plan",
    "checkpoint_policy",
    "pretrained_source",
    "freeze_policy",
    "teacher_variant",
    "token_count",
    "token_grid",
    "trainable_params",
    "total_params",
    "image_encoder_params",
    "visual_context_encoder_params",
    "parameter_count_source",
    "compute_proxy",
    "top1",
    "top3",
    "top5",
    "dba",
    "beam_distance",
    "strict_comparable",
    "claim_eligible",
    "claim_gate_reason",
)

ALLOWED_STAGE_PLANS = {
    "supervised_only",
    "stage1_then_downstream",
    "teacher_then_student",
    "reeval_only",
}
ALLOWED_AVAILABILITY = {"available", "requires_component", "requires_artifact", "skipped_unavailable"}


class FullSweepManifestError(ValueError):
    """Raised when the CNN/hybrid JEPA visual-prior sweep manifest is invalid."""


class FullSweepRunnerError(RuntimeError):
    """Raised when the generated job graph or runner request is unsafe."""


def load_full_sweep_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise FullSweepManifestError(f"Full sweep manifest must be a mapping, got {type(payload).__name__}.")
    return validate_full_sweep_manifest(payload, manifest_path=manifest_path)


def validate_full_sweep_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(manifest)
    families = payload.get("families")
    if not isinstance(families, Mapping):
        raise FullSweepManifestError("Full sweep manifest requires a families mapping.")
    missing_families = sorted(REQUIRED_FAMILIES - {str(key) for key in families})
    if missing_families:
        raise FullSweepManifestError(f"Full sweep manifest is missing required families: {missing_families}.")
    output_root = Path(str(payload.get("output_root", DEFAULT_OUTPUT_ROOT)))
    _assert_output_root_boundary(output_root)
    payload["output_root"] = output_root.as_posix()
    payload.setdefault("strict_comparability_fields", list(STRICT_COMPARABILITY_FIELDS))
    defaults = _mode_defaults(payload, "full")
    base_candidates = _expand_base_candidates(payload)
    _validate_base_candidates(base_candidates)
    expanded = expand_candidate_runs(
        base_candidates,
        seeds=defaults["seeds"],
        checkpoint_selections=defaults["checkpoint_selections"],
        output_root=output_root,
    )
    payload["manifest_path"] = "" if manifest_path is None else Path(manifest_path).as_posix()
    payload["base_candidates"] = base_candidates
    payload["candidates"] = base_candidates
    payload["expanded_candidates"] = expanded
    payload["expanded_candidate_count"] = len(expanded)
    return payload


def generate_runtime_bundle(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_root: str | Path | None = None,
    mode: str = "full",
    families: Iterable[str] | None = None,
    force: bool = False,
    skip_unavailable: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    manifest = load_full_sweep_manifest(manifest_path)
    defaults = _mode_defaults(manifest, mode)
    selected_families = {str(item) for item in families or [] if str(item)}
    base_candidates = _select_base_candidates(manifest["base_candidates"], mode=mode, families=selected_families)
    if not base_candidates:
        raise FullSweepManifestError("Full sweep candidate selection is empty.")
    out_root = Path(output_root or manifest["output_root"])
    _assert_output_root_boundary(out_root)
    _ensure_output_layout(out_root)
    expanded = expand_candidate_runs(
        base_candidates,
        seeds=defaults["seeds"],
        checkpoint_selections=defaults["checkpoint_selections"],
        output_root=out_root,
    )
    manifest_expanded_json, manifest_expanded_csv = _write_expanded_manifest(expanded, out_root)
    config_paths = _write_generated_configs(expanded, out_root, project_root=root, force=force)
    jobs = build_job_manifest(
        expanded,
        out_root,
        project_root=root,
        config_paths=config_paths,
        skip_unavailable=skip_unavailable,
        eval_checkpoint_selections=defaults.get("eval_checkpoint_selections"),
    )
    job_paths = _write_job_tables(jobs, out_root)
    run_script = _write_run_script(out_root, mode=mode)
    summary_script = _write_summary_script(out_root)
    run_note = _write_run_note(
        out_root,
        manifest_path=Path(manifest_path),
        expanded_count=len(expanded),
        job_count=len(jobs["all"]),
        mode=mode,
    )
    return {
        "name": SWEEP_NAME,
        "mode": mode,
        "output_root": out_root.as_posix(),
        "manifest": manifest,
        "expanded_candidates": expanded,
        "base_candidate_count": len(base_candidates),
        "manifest_expanded_json": manifest_expanded_json.as_posix(),
        "manifest_expanded_csv": manifest_expanded_csv.as_posix(),
        "job_paths": {key: value.as_posix() for key, value in job_paths.items()},
        "run_script": run_script.as_posix(),
        "summary_script": summary_script.as_posix(),
        "run_note": run_note.as_posix(),
        "job_count": len(jobs["all"]),
    }


def _select_base_candidates(
    base_candidates: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    families: set[str],
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in base_candidates]
    if mode == "screening":
        screening_ids = set(SCREENING_VARIANT_IDS)
        rows = [row for row in rows if str(row["variant_id"]) in screening_ids]
        found = {str(row["variant_id"]) for row in rows}
        missing = sorted(screening_ids - found)
        if missing:
            raise FullSweepManifestError(f"Screening mode is missing required variants: {missing}.")
    elif mode != "full":
        raise FullSweepManifestError(f"Unsupported mode for {SWEEP_NAME}: {mode!r}.")
    if families:
        rows = [row for row in rows if str(row["family"]) in families]
    return rows


def expand_candidate_runs(
    base_candidates: Iterable[Mapping[str, Any]],
    *,
    seeds: Iterable[int],
    checkpoint_selections: Iterable[str],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    out_root = Path(output_root)
    records: list[dict[str, Any]] = []
    for candidate in base_candidates:
        for seed in seeds:
            for checkpoint_selection in checkpoint_selections:
                variant_id = str(candidate["variant_id"])
                run_id = f"{variant_id}__seed{int(seed)}__{checkpoint_selection}"
                strict = dict(candidate["strict_comparability"])
                strict["seed"] = int(seed)
                strict["output_root"] = out_root.as_posix()
                record = dict(candidate)
                record["base_variant_id"] = variant_id
                record["run_id"] = run_id
                record["seed"] = int(seed)
                record["checkpoint_selection"] = str(checkpoint_selection)
                record["strict_comparability"] = strict
                for field, value in strict.items():
                    record[field] = value
                record["metrics_path"] = _row_metrics_path(record)
                record["checkpoint_path"] = _checkpoint_for_selection(record)
                records.append(record)
    return records


def build_job_manifest(
    expanded: Iterable[Mapping[str, Any]],
    output_root: str | Path,
    *,
    project_root: str | Path,
    config_paths: Mapping[str, Mapping[str, str]],
    skip_unavailable: bool = False,
    eval_checkpoint_selections: Iterable[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    out_root = Path(output_root)
    eval_selection_filter = {str(item) for item in eval_checkpoint_selections or [] if str(item)}
    jobs: dict[str, list[dict[str, str]]] = {
        "stage1": [],
        "downstream": [],
        "teacher_guided": [],
        "reeval": [],
        "summary": [],
        "all": [],
    }
    primary_terminal_jobs: dict[str, str] = {}
    terminal_jobs: list[str] = []
    for record in expanded:
        if str(record["checkpoint_selection"]) != "primary":
            continue
        run_id = str(record["run_id"])
        availability = str(record.get("availability", "available"))
        if availability != "available":
            job = _job_row(
                record,
                stage="unavailable",
                output_root=out_root,
                command=_unavailable_command(record),
                metrics_path=str(record["metrics_path"]),
                skip_reason=_availability_skip_reason(record, skip_unavailable=skip_unavailable),
            )
            jobs["downstream"].append(job)
            primary_terminal_jobs[run_id] = job["job_id"]
            terminal_jobs.append(job["job_id"])
            continue
        stage_plan = str(record["stage_plan"])
        deps: list[str] = []
        if stage_plan == "stage1_then_downstream":
            stage1 = _stage1_job(record, out_root, config_paths)
            jobs["stage1"].append(stage1)
            deps.append(stage1["job_id"])
            downstream = _downstream_job(record, out_root, config_paths, depends_on=deps)
            jobs["downstream"].append(downstream)
            primary_terminal_jobs[run_id] = downstream["job_id"]
            terminal_jobs.append(downstream["job_id"])
        elif stage_plan == "teacher_then_student":
            teacher = _teacher_job(record, out_root, config_paths)
            student = _teacher_guided_student_job(record, out_root, config_paths, depends_on=[teacher["job_id"]])
            jobs["teacher_guided"].extend([teacher, student])
            primary_terminal_jobs[run_id] = student["job_id"]
            terminal_jobs.append(student["job_id"])
        else:
            downstream = _downstream_job(record, out_root, config_paths, depends_on=[])
            jobs["downstream"].append(downstream)
            primary_terminal_jobs[run_id] = downstream["job_id"]
            terminal_jobs.append(downstream["job_id"])

    for record in expanded:
        run_id = str(record["run_id"])
        checkpoint_selection = str(record["checkpoint_selection"])
        if eval_selection_filter and checkpoint_selection not in eval_selection_filter:
            continue
        primary_run_id = run_id.replace(f"__{checkpoint_selection}", "__primary")
        dependency = primary_terminal_jobs.get(primary_run_id)
        depends_on = [dependency] if dependency else []
        if str(record.get("availability", "available")) != "available":
            eval_job = _job_row(
                record,
                stage="reeval",
                output_root=out_root,
                command=_unavailable_command(record),
                metrics_path=str(record["metrics_path"]),
                depends_on=depends_on,
                skip_reason=str(record.get("availability", "unavailable")),
            )
        else:
            eval_job = _eval_job(record, out_root, config_paths, depends_on=depends_on)
        jobs["reeval"].append(eval_job)
        terminal_jobs.append(eval_job["job_id"])

    summary_job = _summary_job(out_root, terminal_jobs)
    jobs["summary"].append(summary_job)
    jobs["all"] = [*jobs["stage1"], *jobs["downstream"], *jobs["teacher_guided"], *jobs["reeval"], *jobs["summary"]]
    _topological_job_order(jobs["all"])
    return jobs


def run_job_manifest(
    *,
    output_root: str | Path,
    jobs_path: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    force_rerun: bool = False,
    clean_output_root: bool = False,
    gpu_list: str | Iterable[str] = ("0", "1", "2", "3"),
    max_parallel: int = 8,
) -> dict[str, Any]:
    out_root = Path(output_root)
    if clean_output_root:
        _clean_output_root(out_root)
    _ensure_output_layout(out_root)
    job_table = Path(jobs_path or out_root / "jobs" / "all.tsv")
    jobs = _read_job_table(job_table)
    ordered = _topological_job_order(jobs)
    gpus = _normalize_gpu_list(gpu_list)
    status_dir = out_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = status_dir / "run_status.jsonl"
    status_csv = status_dir / "run_status.csv"
    if status_jsonl.exists() and not (retry_failed or force_rerun):
        status_jsonl.unlink()
    entries: list[dict[str, Any]] = []
    if dry_run:
        for index, job in enumerate(ordered):
            entry = _status_entry(
                job,
                status="dry_run",
                reason=str(job.get("skip_reason") or "dry_run_no_process_started"),
                gpu=gpus[index % len(gpus)],
            )
            entries.append(entry)
        _write_status(status_jsonl, status_csv, entries)
        command_list = out_root / "status" / "dry_run_commands.sh"
        command_list.write_text(
            "\n".join(_dry_run_command_line(entry, job) for entry, job in zip(entries, ordered)) + "\n",
            encoding="utf-8",
        )
        _write_concurrency_snapshot(out_root, jobs=ordered, entries=entries, gpus=gpus, max_parallel=max_parallel)
        return {
            "status": "dry_run",
            "job_count": len(ordered),
            "status_jsonl": status_jsonl.as_posix(),
            "status_csv": status_csv.as_posix(),
            "dry_run_commands": command_list.as_posix(),
        }
    result = _execute_jobs(
        ordered,
        out_root,
        status_jsonl=status_jsonl,
        status_csv=status_csv,
        gpus=gpus,
        max_parallel=max_parallel,
        resume=resume,
        retry_failed=retry_failed,
        force_rerun=force_rerun,
    )
    _write_concurrency_snapshot(out_root, jobs=ordered, entries=result["entries"], gpus=gpus, max_parallel=max_parallel)
    return result


def generate_summary(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_expanded_path: str | Path | None = None,
    status_path: str | Path | None = None,
) -> dict[str, str]:
    out_root = Path(output_root)
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_expanded_path or out_root / "manifest_expanded.json")
    if not manifest_path.exists():
        raise FullSweepManifestError(f"Expanded manifest not found: {manifest_path}")
    expanded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(expanded, list):
        raise FullSweepManifestError("Expanded manifest must contain a list of records.")
    statuses = _load_status_map(Path(status_path or out_root / "status" / "run_status.csv"))
    full_rows = [_summary_row(record, statuses) for record in expanded]
    strict_rows = [row for row in full_rows if row["claim_eligible"]]
    family_best = _family_best_rows(strict_rows)
    seed_rows = _seed_aggregation_rows(strict_rows)
    pareto_rows = _pareto_rows(strict_rows)
    checkpoint_rows = _checkpoint_selection_rows(strict_rows)
    outputs = {
        "full_results_json": summary_dir / "full_results.json",
        "full_results_csv": summary_dir / "full_results.csv",
        "strict_ranking_csv": summary_dir / "strict_ranking.csv",
        "family_best_csv": summary_dir / "family_best.csv",
        "pareto_csv": summary_dir / "pareto.csv",
        "checkpoint_selection_csv": summary_dir / "checkpoint_selection.csv",
        "seed_aggregation_csv": summary_dir / "seed_aggregation.csv",
        "eval_summary_md": summary_dir / "eval_summary.md",
    }
    outputs["full_results_json"].write_text(json.dumps(full_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(outputs["full_results_csv"], full_rows, SUMMARY_FIELDS)
    _write_csv(outputs["strict_ranking_csv"], _rank_rows(strict_rows), SUMMARY_FIELDS)
    _write_csv(outputs["family_best_csv"], family_best, None)
    _write_csv(outputs["pareto_csv"], pareto_rows, SUMMARY_FIELDS)
    _write_csv(outputs["checkpoint_selection_csv"], checkpoint_rows, None)
    _write_csv(outputs["seed_aggregation_csv"], seed_rows, None)
    outputs["eval_summary_md"].write_text(
        _markdown_summary(full_rows, strict_rows, family_best, pareto_rows, seed_rows),
        encoding="utf-8",
    )
    return {key: value.as_posix() for key, value in outputs.items()}


def _expand_base_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    families = manifest["families"]
    comparison = _comparison_protocol(manifest)
    output_root = Path(str(manifest.get("output_root", DEFAULT_OUTPUT_ROOT)))
    candidates: list[dict[str, Any]] = []
    for raw in _mapping(families["existing_controls"]).get("candidates", []):
        candidates.append(_candidate_from_raw(raw, "existing_controls", comparison, output_root=output_root))
    candidates.extend(_patch_resolution_candidates(_mapping(families["patch_resolution_stage1"]), comparison, output_root))
    candidates.extend(_overlap_candidates(_mapping(families["overlap_stage1"]), comparison, output_root))
    candidates.extend(_cnn_supervised_candidates(_mapping(families["cnn_supervised_tokens"]), comparison, output_root))
    candidates.extend(_cnn_jepa_candidates(_mapping(families["cnn_jepa_tokens"]), comparison, output_root))
    candidates.extend(_tinyvit_jepa_candidates(_mapping(families["tinyvit_jepa_encoders"]), comparison, output_root))
    candidates.extend(_hybrid_candidates(_mapping(families["hybrid_tokenizers"]), comparison, output_root))
    candidates.extend(_pooler_candidates(_mapping(families["pooler_core_ablation"]), comparison, output_root))
    candidates.extend(_teacher_guided_candidates(_mapping(families["teacher_guided_stabilization"]), comparison, output_root))
    candidates.extend(_compute_control_candidates(_mapping(families["compute_controls"]), comparison, output_root))
    return candidates


def _candidate_from_raw(
    raw: Mapping[str, Any],
    family: str,
    comparison: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    visual = _mapping(raw.get("visual_encoder", {"type": "none"}))
    pooler = _mapping(raw.get("pooler", {"type": "mean", "output_mode": "frame"}))
    stage_plan = str(raw.get("stage_plan", "supervised_only"))
    checkpoint_policy = str(raw.get("checkpoint_policy", "supervised_only_anchor"))
    availability = str(raw.get("availability", "available"))
    metadata = _token_metadata(visual)
    return _base_candidate(
        variant_id=str(raw["variant_id"]),
        family=family,
        stage_plan=stage_plan,
        checkpoint_policy=checkpoint_policy,
        availability=availability,
        visual_encoder=visual,
        pooler=pooler,
        token_metadata=metadata,
        params_metadata=_params_metadata(visual, pooler, freeze_policy=str(raw.get("freeze_policy", ""))),
        comparison=comparison,
        output_root=output_root,
        base_config=str(raw.get("base_config", "")),
    )


def _patch_resolution_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    image_size = int(spec.get("image_size", 224))
    pooler = _mapping(spec.get("pooler", _gps_query_pooler(2)))
    rows = []
    for patch_size in _int_list(spec.get("patch_sizes", [16, 14, 12, 10, 8])):
        visual = {
            "type": "patch_vit",
            "patch_size": patch_size,
            "image_size": [image_size, image_size],
            "max_tokens": max(_patch_token_count(image_size, patch_size), 256),
            "checkpoint_policy": "fresh_stage1_required" if patch_size != 16 else "exact_reuse",
            "positional_encoding": "native" if patch_size == 16 else "interpolate_from_patch16",
        }
        rows.append(
            _base_candidate(
                variant_id=f"patch{patch_size}_resolution_stage1_gps_query",
                family="patch_resolution_stage1",
                stage_plan="stage1_then_downstream",
                checkpoint_policy="fresh_stage1_required" if patch_size != 16 else "exact_reuse",
                availability="available",
                visual_encoder=visual,
                pooler=pooler,
                token_metadata=_token_metadata(visual),
                params_metadata=_params_metadata(visual, pooler),
                comparison=comparison,
                output_root=output_root,
            )
        )
    return rows


def _overlap_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    image_size = int(spec.get("image_size", 224))
    pooler = _mapping(spec.get("pooler", _gps_query_pooler(2)))
    rows = []
    for pair in spec.get("kernel_stride_pairs", []):
        kernel, stride = [int(item) for item in pair]
        visual = {
            "type": "overlap_patch",
            "kernel_size": kernel,
            "stride": stride,
            "image_size": [image_size, image_size],
            "max_tokens": max(_overlap_token_count(image_size, kernel, stride), 256),
            "checkpoint_policy": "fresh_stage1_required",
            "positional_encoding": "interpolate_from_patch16",
        }
        rows.append(
            _base_candidate(
                variant_id=f"overlap_k{kernel}_s{stride}_resolution_stage1",
                family="overlap_stage1",
                stage_plan="stage1_then_downstream",
                checkpoint_policy="fresh_stage1_required",
                availability="available",
                visual_encoder=visual,
                pooler=pooler,
                token_metadata=_token_metadata(visual),
                params_metadata=_params_metadata(visual, pooler),
                comparison=comparison,
                output_root=output_root,
            )
        )
    return rows


def _cnn_supervised_candidates(
    spec: Mapping[str, Any],
    comparison: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for backbone in _string_list(spec.get("backbones", [])):
        for token_source in _string_list(spec.get("token_sources", [])):
            for pretrained_source in _string_list(spec.get("pretrained", [])):
                for freeze_policy in _string_list(spec.get("freeze_policies", [])):
                    visual = _cnn_visual(backbone, token_source, pretrained_source, freeze_policy)
                    variant = "_".join(
                        [
                            backbone,
                            token_source.replace("+", "_"),
                            pretrained_source,
                            freeze_policy,
                            "supervised",
                        ]
                    )
                    rows.append(
                        _base_candidate(
                            variant_id=variant,
                            family="cnn_supervised_tokens",
                            stage_plan="supervised_only",
                            checkpoint_policy="supervised_only_anchor",
                            availability="available",
                            visual_encoder=visual,
                            pooler=_gps_query_pooler(2),
                            token_metadata=_token_metadata(visual),
                            params_metadata=_params_metadata(visual, _gps_query_pooler(2), freeze_policy=freeze_policy),
                            comparison=comparison,
                            output_root=output_root,
                            pretrained_source=pretrained_source,
                            freeze_policy=freeze_policy,
                        )
                    )
    return rows


def _cnn_jepa_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for backbone in _string_list(spec.get("backbones", [])):
        for token_source in _string_list(spec.get("token_sources", [])):
            for pretrained_source in _string_list(spec.get("pretrained", [])):
                visual = _cnn_visual(backbone, token_source, pretrained_source, "full_ft")
                variant = f"{backbone}_{token_source.replace('+', '_')}_{pretrained_source}_jepa_stage1"
                rows.append(
                    _base_candidate(
                        variant_id=variant,
                        family="cnn_jepa_tokens",
                        stage_plan="stage1_then_downstream",
                        checkpoint_policy="fresh_stage1_required",
                        availability="available",
                        visual_encoder=visual,
                        pooler=_gps_query_pooler(2),
                        token_metadata=_token_metadata(visual),
                        params_metadata=_params_metadata(visual, _gps_query_pooler(2), freeze_policy="full_ft"),
                        comparison=comparison,
                        output_root=output_root,
                        pretrained_source=pretrained_source,
                        freeze_policy="full_ft",
                    )
                )
    return rows


def _tinyvit_jepa_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    variants = _string_list(spec.get("variants", ["5m", "11m"]))
    pretrained_options = _string_list(spec.get("pretrained", ["scratch", "22k"]))
    for variant in variants:
        for pretrained_source in pretrained_options:
            visual = _tinyvit_visual(variant, pretrained_source)
            variant_id = f"tinyvit_{variant}_{pretrained_source}_jepa_stage1"
            rows.append(
                _base_candidate(
                    variant_id=variant_id,
                    family="tinyvit_jepa_encoders",
                    stage_plan="supervised_only",
                    checkpoint_policy="supervised_only_anchor",
                    availability="available",
                    visual_encoder=visual,
                    pooler=_gps_query_pooler(2),
                    token_metadata=_token_metadata(visual),
                    params_metadata=_params_metadata(visual, _gps_query_pooler(2), freeze_policy="full_ft"),
                    comparison=comparison,
                    output_root=output_root,
                    pretrained_source=pretrained_source,
                    freeze_policy="full_ft",
                    local_prior_mechanism="tinyvit_global_frame",
                )
            )
    return rows


def _hybrid_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for tokenizer in _string_list(spec.get("tokenizers", [])):
        visual = _hybrid_visual(tokenizer)
        rows.append(
            _base_candidate(
                variant_id=f"{tokenizer}_stage1",
                family="hybrid_tokenizers",
                stage_plan="stage1_then_downstream",
                checkpoint_policy="fresh_stage1_required",
                availability="available",
                visual_encoder=visual,
                pooler=_gps_query_pooler(2),
                token_metadata=_token_metadata(visual),
                params_metadata=_params_metadata(visual, _gps_query_pooler(2)),
                comparison=comparison,
                output_root=output_root,
                local_prior_mechanism=str(visual.get("local_prior_mechanism", "")),
            )
        )
    return rows


def _pooler_candidates(spec: Mapping[str, Any], comparison: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for variant in _string_list(spec.get("variants", [])):
        pooler, core = _pooler_variant(variant)
        visual = {"type": "patch_vit", "patch_size": 16, "image_size": [224, 224], "max_tokens": 256}
        rows.append(
            _base_candidate(
                variant_id=f"pooler_{variant}",
                family="pooler_core_ablation",
                stage_plan="supervised_only",
                checkpoint_policy="exact_reuse",
                availability="available",
                visual_encoder=visual,
                pooler=pooler,
                token_metadata=_token_metadata(visual),
                params_metadata=_params_metadata(visual, pooler),
                comparison=comparison,
                output_root=output_root,
                representation_core=core,
            )
        )
    return rows


def _teacher_guided_candidates(
    spec: Mapping[str, Any],
    comparison: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for teacher in _string_list(spec.get("teachers", [])):
        for student in _string_list(spec.get("students", [])):
            for temperature in _number_list(spec.get("temperatures", [])):
                for weight in _number_list(spec.get("weights", [])):
                    visual = _student_visual(student)
                    pooler = _gps_query_pooler(2)
                    variant = f"tg_{teacher}_to_{student}_t{_num_tag(temperature)}_w{_num_tag(weight)}"
                    rows.append(
                        _base_candidate(
                            variant_id=variant,
                            family="teacher_guided_stabilization",
                            stage_plan="teacher_then_student",
                            checkpoint_policy="teacher_guided_student",
                            availability="available",
                            visual_encoder=visual,
                            pooler=pooler,
                            token_metadata=_token_metadata(visual),
                            params_metadata=_params_metadata(visual, pooler),
                            comparison=comparison,
                            output_root=output_root,
                            teacher_variant=teacher,
                            teacher_checkpoint_source=f"{output_root.as_posix()}/teachers/{teacher}/checkpoints/best.pth",
                            teacher_logits_source=f"{output_root.as_posix()}/teachers/{teacher}/logits.pt",
                            teacher_guidance={
                                "enabled": True,
                                "mode": "opt_in_stabilization",
                                "temperature": float(temperature),
                                "weight": float(weight),
                                "detach_policy": "detach_teacher",
                                "detach_teacher": True,
                                "enabled_splits": ["train"],
                                "checkpoint_provenance": f"{teacher}_same_protocol",
                            },
                            local_prior_mechanism=str(visual.get("local_prior_mechanism", "")),
                        )
                    )
    return rows


def _compute_control_candidates(
    spec: Mapping[str, Any],
    comparison: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for control in _string_list(spec.get("controls", [])):
        visual = _compute_control_visual(control)
        pooler = _gps_query_pooler(2)
        rows.append(
            _base_candidate(
                variant_id=control,
                family="compute_controls",
                stage_plan="supervised_only",
                checkpoint_policy="supervised_only_anchor",
                availability="requires_component" if control == "random_feature_controls" else "available",
                visual_encoder=visual,
                pooler=pooler,
                token_metadata=_token_metadata(visual),
                params_metadata=_params_metadata(visual, pooler, control=control),
                comparison=comparison,
                output_root=output_root,
                compute_control=control,
            )
        )
    return rows


def _base_candidate(
    *,
    variant_id: str,
    family: str,
    stage_plan: str,
    checkpoint_policy: str,
    availability: str,
    visual_encoder: Mapping[str, Any],
    pooler: Mapping[str, Any],
    token_metadata: Mapping[str, Any],
    params_metadata: Mapping[str, Any],
    comparison: Mapping[str, Any],
    output_root: Path,
    **extra: Any,
) -> dict[str, Any]:
    strict = dict(comparison)
    strict["output_root"] = output_root.as_posix()
    candidate: dict[str, Any] = {
        "variant_id": variant_id,
        "family": family,
        "stage_plan": stage_plan,
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_selection": "primary",
        "availability": availability,
        "run_tier": "strict",
        "evidence_scope": "strict",
        "visual_encoder": dict(visual_encoder),
        "token_source": dict(visual_encoder),
        "pooler": dict(pooler),
        "token_metadata": dict(token_metadata),
        "params_metadata": dict(params_metadata),
        "strict_comparable": availability == "available",
        "strict_comparability": strict,
        "metrics_path_template": f"{output_root.as_posix()}/metrics/{{run_id}}/metrics.json",
        "smoke_only": False,
        "pretrained_source": extra.pop("pretrained_source", _pretrained_source(visual_encoder)),
        "freeze_policy": extra.pop("freeze_policy", _freeze_policy(visual_encoder)),
    }
    candidate.update(extra)
    for field, value in strict.items():
        candidate[field] = value
    return candidate


def _validate_base_candidates(candidates: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        variant_id = str(candidate.get("variant_id") or "")
        if not variant_id:
            raise FullSweepManifestError(f"Candidate {index} is missing variant_id.")
        if variant_id in seen:
            raise FullSweepManifestError(f"Full sweep manifest has duplicate variant_id {variant_id!r}.")
        seen.add(variant_id)
        missing = [field for field in CANDIDATE_REQUIRED_FIELDS if field not in candidate]
        if missing:
            raise FullSweepManifestError(f"Candidate {variant_id!r} is missing fields: {missing}.")
        if str(candidate["stage_plan"]) not in ALLOWED_STAGE_PLANS:
            raise FullSweepManifestError(f"Candidate {variant_id!r} has invalid stage_plan {candidate['stage_plan']!r}.")
        if str(candidate["availability"]) not in ALLOWED_AVAILABILITY:
            raise FullSweepManifestError(
                f"Candidate {variant_id!r} has invalid availability {candidate['availability']!r}."
            )
        visual = candidate.get("visual_encoder")
        token_source = candidate.get("token_source")
        pooler = candidate.get("pooler")
        if not isinstance(visual, Mapping) or "type" not in visual:
            raise FullSweepManifestError(f"Candidate {variant_id!r} requires visual_encoder.type.")
        if not isinstance(token_source, Mapping) or "type" not in token_source:
            raise FullSweepManifestError(f"Candidate {variant_id!r} requires token_source.type.")
        if not isinstance(pooler, Mapping) or "type" not in pooler:
            raise FullSweepManifestError(f"Candidate {variant_id!r} requires pooler.type.")
        for metadata_key in ("token_metadata", "params_metadata", "strict_comparability"):
            if not isinstance(candidate.get(metadata_key), Mapping):
                raise FullSweepManifestError(f"Candidate {variant_id!r} requires {metadata_key} metadata.")


def _write_generated_configs(
    expanded: Iterable[Mapping[str, Any]],
    output_root: Path,
    *,
    project_root: Path,
    force: bool,
) -> dict[str, dict[str, str]]:
    config_paths: dict[str, dict[str, str]] = {}
    for record in expanded:
        run_id = str(record["run_id"])
        paths: dict[str, str] = {}
        if str(record["checkpoint_selection"]) == "primary":
            if str(record["stage_plan"]) == "stage1_then_downstream":
                pretraining_path = output_root / "generated_configs" / "pretraining" / f"{run_id}.yaml"
                _write_yaml(pretraining_path, _pretraining_config(record, pretraining_path, project_root), force=force)
                paths["pretraining"] = pretraining_path.as_posix()
            downstream_path = output_root / "generated_configs" / "downstream" / f"{run_id}.yaml"
            _write_yaml(downstream_path, _downstream_config(record, downstream_path, project_root), force=force)
            paths["downstream"] = downstream_path.as_posix()
            if str(record["stage_plan"]) == "teacher_then_student":
                teacher_path = output_root / "generated_configs" / "downstream" / f"{run_id}__teacher.yaml"
                _write_yaml(teacher_path, _teacher_config(record, teacher_path, project_root), force=force)
                paths["teacher"] = teacher_path.as_posix()
        eval_path = output_root / "generated_configs" / "eval" / f"{run_id}.yaml"
        _write_yaml(eval_path, _eval_config(record, eval_path, project_root), force=force)
        paths["eval"] = eval_path.as_posix()
        config_paths[run_id] = paths
    return config_paths


def _pretraining_config(record: Mapping[str, Any], path: Path, project_root: Path) -> dict[str, Any]:
    visual = _config_visual_encoder(record)
    return {
        "_base_": _rel_config_path(path, project_root / BASE_PRETRAINING_CONFIG),
        "experiment": {"name": str(record["run_id"]), "seed": int(record["seed"])},
        "model": {
            "primary": {
                "visual_encoder": visual,
                "predictor": {"max_tokens": int(record["token_metadata"]["token_count"])},
            }
        },
        "output": {
            "dir": str(_run_stage_parent(record)),
            "run_name": "stage1",
            "group_by_scene": False,
            "overwrite": True,
            "tensorboard": {"enabled": False},
            "progress": {"enabled": False},
        },
        "sweep": _config_sweep_metadata(record),
    }


def _downstream_config(record: Mapping[str, Any], path: Path, project_root: Path) -> dict[str, Any]:
    base = _base_downstream_config(record)
    config: dict[str, Any] = {
        "_base_": _rel_config_path(path, project_root / base),
        "experiment": {"name": str(record["run_id"]), "seed": int(record["seed"])},
        "output": {
            "dir": str(_run_stage_parent(record)),
            "run_name": "downstream",
            "group_by_scene": False,
            "overwrite": True,
            "tensorboard": {"enabled": False},
            "progress": {"enabled": False},
        },
        "strict_comparability": dict(record["strict_comparability"]),
        "sweep": _config_sweep_metadata(record),
    }
    if _visual_type(record) != "none":
        config.setdefault("model", {}).setdefault("primary", {}).setdefault("encoders", {})["image"] = {
            "type": "jepa_context_image",
            "checkpoint_path": _stage1_checkpoint_path(record),
            "strict": False,
            "output_dim": 64,
            "latent_dim": 64,
            "image_channels": 3,
            "visual_encoder": _config_visual_encoder(record),
            "pooler": _config_pooler(record),
        }
    representation_core = record.get("representation_core")
    if isinstance(representation_core, Mapping):
        config.setdefault("model", {}).setdefault("primary", {})["representation_core"] = dict(representation_core)
    if str(record["stage_plan"]) == "teacher_then_student":
        guidance = dict(record.get("teacher_guidance", {}))
        guidance["checkpoint_path"] = str(record.get("teacher_checkpoint_source", ""))
        guidance["logits_path"] = str(record.get("teacher_logits_source", ""))
        config["loss"] = {"teacher_guidance": guidance}
    return config


def _teacher_config(record: Mapping[str, Any], path: Path, project_root: Path) -> dict[str, Any]:
    teacher_variant = str(record.get("teacher_variant", "teacher"))
    return {
        "_base_": _rel_config_path(path, project_root / BASE_SUPERVISED_IMAGE_GPS_CONFIG),
        "experiment": {"name": f"{teacher_variant}__teacher_seed{int(record['seed'])}", "seed": int(record["seed"])},
        "output": {
            "dir": str(Path(record["output_root"]) / "teachers"),
            "run_name": f"{teacher_variant}__seed{int(record['seed'])}",
            "tensorboard": {"enabled": False},
            "progress": {"enabled": False},
        },
        "sweep": {
            "name": SWEEP_NAME,
            "role": "teacher",
            "teacher_variant": teacher_variant,
            "student_variant": str(record["variant_id"]),
        },
    }


def _eval_config(record: Mapping[str, Any], path: Path, project_root: Path) -> dict[str, Any]:
    config = _downstream_config(record, path, project_root)
    config["experiment"] = {"name": f"{record['run_id']}__eval", "seed": int(record["seed"])}
    config["evaluation"] = {
        "weights": _checkpoint_for_selection(record),
        "checkpoint_selection": str(record["checkpoint_selection"]),
    }
    config["output"] = {
        "dir": str(Path(record["output_root"]) / "eval"),
        "run_name": str(record["run_id"]),
        "group_by_scene": False,
        "tensorboard": {"enabled": False},
        "progress": {"enabled": False},
    }
    config["sweep"] = _config_sweep_metadata(record)
    return config


def _stage1_job(record: Mapping[str, Any], output_root: Path, config_paths: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    run_id = str(record["run_id"])
    config = config_paths[run_id]["pretraining"]
    return _job_row(
        record,
        stage="stage1",
        output_root=output_root,
        config_path=config,
        command=f"{CONDA_PREFIX}kd-sensing-train --config {config}",
        output_dir=str(_training_run_dir(record, "stage1")),
        metrics_path=str(_training_run_dir(record, "stage1") / "metrics.json"),
    )


def _downstream_job(
    record: Mapping[str, Any],
    output_root: Path,
    config_paths: Mapping[str, Mapping[str, str]],
    *,
    depends_on: list[str],
) -> dict[str, str]:
    run_id = str(record["run_id"])
    config = config_paths[run_id]["downstream"]
    return _job_row(
        record,
        stage="downstream",
        output_root=output_root,
        config_path=config,
        command=f"{CONDA_PREFIX}kd-sensing-train --config {config}",
        output_dir=str(_training_run_dir(record, "downstream")),
        metrics_path=str(_training_run_dir(record, "downstream") / "metrics.json"),
        depends_on=depends_on,
    )


def _teacher_job(record: Mapping[str, Any], output_root: Path, config_paths: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    run_id = str(record["run_id"])
    config = config_paths[run_id]["teacher"]
    teacher_variant = str(record.get("teacher_variant", "teacher"))
    return _job_row(
        record,
        stage="teacher",
        output_root=output_root,
        config_path=config,
        command=f"{CONDA_PREFIX}kd-sensing-train --config {config}",
        output_dir=str(output_root / "teachers" / teacher_variant / f"seed{int(record['seed'])}"),
        metrics_path=str(output_root / "teachers" / teacher_variant / f"seed{int(record['seed'])}" / "metrics.json"),
    )


def _teacher_guided_student_job(
    record: Mapping[str, Any],
    output_root: Path,
    config_paths: Mapping[str, Mapping[str, str]],
    *,
    depends_on: list[str],
) -> dict[str, str]:
    run_id = str(record["run_id"])
    config = config_paths[run_id]["downstream"]
    return _job_row(
        record,
        stage="teacher_guided",
        output_root=output_root,
        config_path=config,
        command=f"{CONDA_PREFIX}kd-sensing-train --config {config}",
        output_dir=str(_training_run_dir(record, "downstream")),
        metrics_path=str(_training_run_dir(record, "downstream") / "metrics.json"),
        depends_on=depends_on,
    )


def _eval_job(
    record: Mapping[str, Any],
    output_root: Path,
    config_paths: Mapping[str, Mapping[str, str]],
    *,
    depends_on: list[str],
) -> dict[str, str]:
    run_id = str(record["run_id"])
    config = config_paths[run_id]["eval"]
    return _job_row(
        record,
        stage="reeval",
        output_root=output_root,
        config_path=config,
        command=(
            f"{CONDA_PREFIX}kd-sensing-evaluate --config {config} "
            f"--weights {_checkpoint_for_selection(record)} --output-dir {_eval_output_dir(record)}"
        ),
        output_dir=str(_eval_output_dir(record)),
        metrics_path=str(_eval_metrics_path(record)),
        depends_on=depends_on,
    )


def _summary_job(output_root: Path, depends_on: list[str]) -> dict[str, str]:
    command = f"{CONDA_PREFIX}python -m kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep --summarize --output-root {output_root.as_posix()}"
    return {
        "job_id": "summary__full_sweep",
        "variant_id": "summary",
        "run_id": "summary__full_sweep",
        "family": "summary",
        "stage": "summary",
        "depends_on": ";".join(dict.fromkeys(depends_on)),
        "command": command,
        "output_dir": str(output_root / "summary"),
        "metrics_path": str(output_root / "summary" / "full_results.json"),
        "log_path": str(output_root / "logs" / "summary__full_sweep.log"),
        "config_path": "",
        "availability": "available",
        "success_marker": str(output_root / "summary" / "_SUCCESS"),
        "skip_reason": "",
        "gpu_policy": "none",
        "max_parallel_group": "summary",
    }


def _job_row(
    record: Mapping[str, Any],
    *,
    stage: str,
    output_root: Path,
    command: str,
    metrics_path: str,
    config_path: str = "",
    output_dir: str | None = None,
    depends_on: list[str] | None = None,
    skip_reason: str = "",
) -> dict[str, str]:
    run_id = str(record["run_id"])
    safe_stage = stage.replace("/", "_")
    job_id = f"{safe_stage}__{run_id}"
    out_dir = output_dir or str(output_root / "runs" / run_id / safe_stage)
    return {
        "job_id": job_id,
        "variant_id": str(record["variant_id"]),
        "run_id": run_id,
        "family": str(record["family"]),
        "checkpoint_selection": str(record["checkpoint_selection"]),
        "stage": stage,
        "depends_on": ";".join(depends_on or []),
        "command": command,
        "output_dir": out_dir,
        "metrics_path": metrics_path,
        "log_path": str(output_root / "logs" / f"{job_id}.log"),
        "config_path": config_path,
        "availability": str(record.get("availability", "available")),
        "success_marker": str(Path(out_dir) / "_SUCCESS"),
        "skip_reason": skip_reason,
        "gpu_policy": "runner_injects_cuda_visible_devices" if stage not in {"summary", "unavailable"} else "none",
        "max_parallel_group": "project",
    }


def _execute_jobs(
    jobs: list[dict[str, str]],
    output_root: Path,
    *,
    status_jsonl: Path,
    status_csv: Path,
    gpus: list[str],
    max_parallel: int,
    resume: bool,
    retry_failed: bool,
    force_rerun: bool,
) -> dict[str, Any]:
    pending = {job["job_id"]: job for job in jobs}
    succeeded: set[str] = set()
    failed: set[str] = set()
    running: dict[str, tuple[subprocess.Popen[Any], Any, dict[str, str], str]] = {}
    entries: list[dict[str, Any]] = []
    gpu_index = 0
    max_parallel = max(1, int(max_parallel))
    while pending or running:
        for job_id, (proc, log_handle, job, gpu) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log_handle.close()
            pending.pop(job_id, None)
            if code == 0:
                Path(job["success_marker"]).parent.mkdir(parents=True, exist_ok=True)
                Path(job["success_marker"]).write_text("success\n", encoding="utf-8")
                succeeded.add(job_id)
                entries.append(_status_entry(job, status="success", gpu=gpu, returncode=code))
            else:
                failed.add(job_id)
                entries.append(_status_entry(job, status="failed", gpu=gpu, returncode=code))
            running.pop(job_id, None)
            _write_status(status_jsonl, status_csv, entries)
        launched = False
        for job_id, job in list(pending.items()):
            if job_id in running:
                continue
            deps = _job_deps(job)
            if any(dep in failed for dep in deps):
                pending.pop(job_id, None)
                failed.add(job_id)
                entries.append(_status_entry(job, status="blocked_dependency_failed", reason=";".join(deps)))
                _write_status(status_jsonl, status_csv, entries)
                launched = True
                continue
            if not all(dep in succeeded for dep in deps):
                continue
            if str(job.get("availability")) != "available":
                pending.pop(job_id, None)
                succeeded.add(job_id)
                entries.append(_status_entry(job, status="skipped_unavailable", reason=str(job.get("availability"))))
                _write_status(status_jsonl, status_csv, entries)
                launched = True
                continue
            if resume and not force_rerun and _job_complete(job):
                pending.pop(job_id, None)
                succeeded.add(job_id)
                entries.append(_status_entry(job, status="skipped_resume", reason="metrics_and_success_marker_exist"))
                _write_status(status_jsonl, status_csv, entries)
                launched = True
                continue
            if len(running) >= max_parallel:
                break
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            if job.get("gpu_policy") != "none":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(job["command"], shell=True, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            running[job_id] = (proc, log_handle, job, gpu)
            entries.append(_status_entry(job, status="running", gpu=gpu, reason="retry_failed" if retry_failed else ""))
            _write_status(status_jsonl, status_csv, entries)
            launched = True
        if not launched:
            if not running and pending:
                blocked = sorted(pending)
                for job_id in blocked:
                    job = pending.pop(job_id)
                    entries.append(_status_entry(job, status="blocked_dependency", reason=job.get("depends_on", "")))
                _write_status(status_jsonl, status_csv, entries)
                raise FullSweepRunnerError(f"Job graph stalled with blocked jobs: {blocked[:5]}")
            time.sleep(0.5)
    status = "success" if not failed else "failed"
    return {"status": status, "entries": entries, "status_jsonl": status_jsonl.as_posix(), "status_csv": status_csv.as_posix()}


def _summary_row(record: Mapping[str, Any], statuses: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    row = dict(record)
    architecture_summary = summarize_sweep_candidate(record)
    row["architecture_summary"] = architecture_summary
    metrics_path = Path(str(record.get("metrics_path", "")))
    metrics = _read_metrics(metrics_path)
    status = _record_status(record, metrics, statuses)
    row["status"] = status
    row.update(metrics)
    row["top1"] = _metric_value(metrics, "top1")
    row["top3"] = _metric_value(metrics, "top3")
    row["top5"] = _metric_value(metrics, "top5")
    row["dba"] = _metric_value(metrics, "dba", "DBA")
    row["beam_distance"] = _metric_value(metrics, "beam_distance", "adjacent_beam_error", "mean_circular_error")
    token_metadata = _mapping(record.get("token_metadata", {}))
    parameters = _mapping(architecture_summary.get("parameters", {}))
    comparability = _mapping(architecture_summary.get("comparability", {}))
    row["token_count"] = comparability.get("token_count", token_metadata.get("token_count"))
    row["token_grid"] = token_metadata.get("token_grid")
    row["trainable_params"] = parameters.get("trainable_params")
    row["total_params"] = parameters.get("total_params")
    row["image_encoder_params"] = parameters.get("image_encoder_params")
    row["visual_context_encoder_params"] = parameters.get("visual_context_encoder_params")
    row["parameter_count_source"] = parameters.get("parameter_count_source")
    row["compute_proxy"] = comparability.get("compute_proxy")
    eligible, reason = _claim_gate_reason(row)
    row["claim_eligible"] = eligible
    row["claim_gate_reason"] = reason
    return row


def _claim_gate_reason(row: Mapping[str, Any]) -> tuple[bool, str]:
    reasons: list[str] = []
    if str(row.get("availability")) != "available":
        reasons.append("unavailable")
    if str(row.get("status")) != "success":
        reasons.append(f"status_{row.get('status')}")
    if not bool(row.get("strict_comparable", False)):
        reasons.append("strict_comparable_false")
    if bool(row.get("smoke_only", False)):
        reasons.append("smoke_only")
    if str(row.get("checkpoint_selection", "")) not in {"primary", "best_top1", "last"}:
        reasons.append("checkpoint_selection_mismatch")
    strict = row.get("strict_comparability")
    if not isinstance(strict, Mapping):
        reasons.append("missing_strict_comparability")
    else:
        missing = [field for field in STRICT_COMPARABILITY_FIELDS if strict.get(field) in (None, "")]
        if missing:
            reasons.append("missing_strict_fields:" + ",".join(missing))
        for field in ("split", "metric_profile", "checkpoint_selection"):
            if field in row and field in strict and row.get(field) not in (strict.get(field), None, ""):
                reasons.append(f"{field}_mismatch")
        if row.get("seed") not in (17, 23, 42):
            reasons.append("seed_mismatch")
    if row.get("top1") is None or row.get("dba") is None:
        reasons.append("missing_primary_metrics")
    if not row.get("checkpoint_policy"):
        reasons.append("missing_checkpoint_policy")
    if str(row.get("stage_plan")) in {"stage1_then_downstream", "teacher_then_student"} and not row.get(
        "checkpoint_path"
    ):
        reasons.append("missing_checkpoint_provenance")
    return (not reasons, "eligible" if not reasons else ";".join(reasons))


def _family_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family"))].append(row)
    result = []
    for family, items in sorted(grouped.items()):
        best = max(items, key=lambda item: (_numeric(item.get("dba")), _numeric(item.get("top1"))))
        aggregate = _aggregate_seed_metrics(items)
        result.append(
            {
                "family": family,
                "best_variant_id": best.get("variant_id"),
                "best_run_id": best.get("run_id"),
                "checkpoint_selection": best.get("checkpoint_selection"),
                "top1": best.get("top1"),
                "top3": best.get("top3"),
                "top5": best.get("top5"),
                "dba": best.get("dba"),
                "beam_distance": best.get("beam_distance"),
                "total_params": best.get("total_params"),
                "trainable_params": best.get("trainable_params"),
                "image_encoder_params": best.get("image_encoder_params"),
                "visual_context_encoder_params": best.get("visual_context_encoder_params"),
                "token_count": best.get("token_count"),
                "compute_proxy": best.get("compute_proxy"),
                **aggregate,
            }
        )
    return result


def _seed_aggregation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("variant_id")), str(row.get("checkpoint_selection")))].append(row)
    result = []
    for (variant_id, checkpoint_selection), items in sorted(grouped.items()):
        aggregate = _aggregate_seed_metrics(items)
        result.append(
            {
                "variant_id": variant_id,
                "checkpoint_selection": checkpoint_selection,
                "seeds": ",".join(str(item.get("seed")) for item in sorted(items, key=lambda item: int(item.get("seed", 0)))),
                **aggregate,
            }
        )
    return result


def _checkpoint_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row.get("variant_id")), int(row.get("seed", 0)))][str(row.get("checkpoint_selection"))] = row
    result = []
    for (variant_id, seed), selections in sorted(grouped.items()):
        primary = selections.get("primary", {})
        best = selections.get("best_top1", {})
        result.append(
            {
                "variant_id": variant_id,
                "seed": seed,
                "primary_top1": primary.get("top1"),
                "best_top1": best.get("top1"),
                "delta_top1": _numeric(best.get("top1")) - _numeric(primary.get("top1")),
                "primary_dba": primary.get("dba"),
                "best_dba": best.get("dba"),
                "delta_dba": _numeric(best.get("dba")) - _numeric(primary.get("dba")),
            }
        )
    return result


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("top1") is not None and row.get("dba") is not None]
    pareto = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            enriched = dict(row)
            enriched["cnn_candidate"] = "cnn" in str(row.get("family")) or str(row.get("visual_encoder", "")).find("cnn") >= 0
            enriched["imagenet_candidate"] = str(row.get("pretrained_source")) == "imagenet"
            enriched["frozen_candidate"] = str(row.get("freeze_policy")).startswith("freeze")
            enriched["teacher_guided_candidate"] = str(row.get("family")) == "teacher_guided_stabilization"
            enriched["hybrid_candidate"] = str(row.get("family")) == "hybrid_tokenizers"
            pareto.append(enriched)
    return _rank_rows(pareto)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    better_or_equal = (
        _numeric(left.get("dba")) >= _numeric(right.get("dba"))
        and _numeric(left.get("top1")) >= _numeric(right.get("top1"))
        and _numeric(left.get("trainable_params")) <= _numeric(right.get("trainable_params"))
        and _numeric(left.get("total_params")) <= _numeric(right.get("total_params"))
        and _numeric(left.get("image_encoder_params")) <= _numeric(right.get("image_encoder_params"))
        and _numeric(left.get("visual_context_encoder_params")) <= _numeric(right.get("visual_context_encoder_params"))
        and _numeric(left.get("token_count")) <= _numeric(right.get("token_count"))
        and _numeric(left.get("compute_proxy")) <= _numeric(right.get("compute_proxy"))
    )
    strictly_better = (
        _numeric(left.get("dba")) > _numeric(right.get("dba"))
        or _numeric(left.get("top1")) > _numeric(right.get("top1"))
        or _numeric(left.get("trainable_params")) < _numeric(right.get("trainable_params"))
        or _numeric(left.get("total_params")) < _numeric(right.get("total_params"))
        or _numeric(left.get("image_encoder_params")) < _numeric(right.get("image_encoder_params"))
        or _numeric(left.get("visual_context_encoder_params")) < _numeric(right.get("visual_context_encoder_params"))
        or _numeric(left.get("token_count")) < _numeric(right.get("token_count"))
        or _numeric(left.get("compute_proxy")) < _numeric(right.get("compute_proxy"))
    )
    return better_or_equal and strictly_better


def _markdown_summary(
    full_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    family_best: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> str:
    status_counts: dict[str, int] = defaultdict(int)
    for row in full_rows:
        status_counts[str(row.get("status"))] += 1
    family_lookup = {str(row.get("family")): row for row in family_best}
    comparisons = [
        ("resolution", "patch_resolution_stage1"),
        ("overlap", "overlap_stage1"),
        ("CNN local prior", "cnn_supervised_tokens"),
        ("pretraining/freeze", "cnn_jepa_tokens"),
        ("teacher-guided", "teacher_guided_stabilization"),
    ]
    lines = [
        "# CNN/Hybrid JEPA Visual Prior Sweep Summary",
        "",
        f"- Full rows: {len(full_rows)}",
        f"- Strict rows: {len(strict_rows)}",
        f"- Pareto candidates: {len(pareto_rows)}",
        f"- Seed aggregate rows: {len(seed_rows)}",
        f"- Status counts: {dict(sorted(status_counts.items()))}",
        "",
        "## Interpretable Comparisons",
        "",
        "| comparison | best variant | top1 | DBA | checkpoint | seed mean/std |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for label, family in comparisons:
        row = family_lookup.get(family, {})
        lines.append(
            "| {label} | `{variant}` | {top1} | {dba} | {checkpoint} | {mean_std} |".format(
                label=label,
                variant=row.get("best_variant_id", "missing"),
                top1=_fmt(row.get("top1")),
                dba=_fmt(row.get("dba")),
                checkpoint=row.get("checkpoint_selection", ""),
                mean_std=f"{_fmt(row.get('seed_top1_mean'))}/{_fmt(row.get('seed_top1_std'))}",
            )
        )
    lines.extend(
        [
            "",
            "## Parameter Scale",
            "",
            "| variant | total params | image encoder params | visual/context encoder params | tokens | compute proxy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    scale_rows = [
        row
        for row in full_rows
        if str(row.get("checkpoint_selection")) == "primary"
        and str(row.get("variant_id"))
        in {"patch14_stage1_gps_query", "resnet18_layer4_tokens", "resnet18_layer3_layer4_tokens"}
    ]
    for row in sorted(scale_rows, key=lambda item: _numeric(item.get("total_params"))):
        lines.append(
            "| `{variant}` | {total} | {image} | {visual} | {tokens} | {compute} |".format(
                variant=row.get("variant_id", ""),
                total=_fmt(row.get("total_params")),
                image=_fmt(row.get("image_encoder_params")),
                visual=_fmt(row.get("visual_context_encoder_params")),
                tokens=_fmt(row.get("token_count")),
                compute=_fmt(row.get("compute_proxy")),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            "Strict ranking includes only rows with strict comparability metadata, non-smoke scope, complete checkpoint provenance, matching split/metric fields, and metrics files present.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_expanded_manifest(expanded: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    json_path = output_root / "manifest_expanded.json"
    csv_path = output_root / "manifest_expanded.csv"
    json_path.write_text(json.dumps(expanded, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, expanded, None)
    return json_path, csv_path


def _write_job_tables(jobs: Mapping[str, list[dict[str, str]]], output_root: Path) -> dict[str, Path]:
    job_dir = output_root / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rows in jobs.items():
        path = job_dir / f"{name}.tsv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOB_FIELDS, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in JOB_FIELDS})
        paths[name] = path
    return paths


def _write_run_script(output_root: Path, *, mode: str) -> Path:
    path = output_root / "run_full_sweep.sh"
    text = f"""#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root.as_posix()}}}"
conda run -n kd_mm_beam python -m kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep --mode {mode} --output-root "$OUTPUT_ROOT" --run-jobs "$@"
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_summary_script(output_root: Path) -> Path:
    path = output_root / "summarize_full_sweep.sh"
    text = f"""#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root.as_posix()}}}"
conda run -n kd_mm_beam python -m kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep --summarize --output-root "$OUTPUT_ROOT" "$@"
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_run_note(output_root: Path, *, manifest_path: Path, expanded_count: int, job_count: int, mode: str) -> Path:
    path = output_root / "RUN_NOTE.md"
    path.write_text(
        "\n".join(
            [
                "# CNN/Hybrid JEPA Visual Prior Sweep Runtime Bundle",
                "",
                f"- Source manifest: `{manifest_path.as_posix()}`",
                f"- Mode: `{mode}`",
                f"- Expanded rows: `{expanded_count}`",
                f"- Jobs: `{job_count}`",
                "- Default GPUs: `0,1,2,3`",
                "- Default max parallel project jobs: `8`",
                "- All generated configs, logs, metrics, checkpoints, summaries, and status files stay under this output root.",
                "",
                "Dry-run command:",
                "",
                "```bash",
                f"conda run -n kd_mm_beam python -m kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep --mode {mode} --dry-run --output-root {output_root.as_posix()}",
                "```",
                "",
                "Actual training is intentionally an explicit action through `run_full_sweep.sh` or `--run-jobs`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _topological_job_order(jobs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    job_list = [dict(job) for job in jobs]
    by_id = {job["job_id"]: job for job in job_list}
    indegree = {job_id: 0 for job_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for job in job_list:
        for dep in _job_deps(job):
            if dep not in by_id:
                continue
            indegree[job["job_id"]] += 1
            children[dep].append(job["job_id"])
    queue = deque([job_id for job_id, count in indegree.items() if count == 0])
    ordered: list[dict[str, str]] = []
    while queue:
        job_id = queue.popleft()
        ordered.append(by_id[job_id])
        for child in children[job_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(ordered) != len(job_list):
        blocked = sorted(job_id for job_id, count in indegree.items() if count > 0)
        raise FullSweepRunnerError(f"Job dependency graph has a cycle or unsatisfied cycle: {blocked[:10]}")
    return ordered


def _job_deps(job: Mapping[str, str]) -> list[str]:
    return [item for item in str(job.get("depends_on", "")).split(";") if item]


def _read_job_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FullSweepRunnerError(f"Job manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def _write_status(jsonl_path: Path, csv_path: Path, entries: list[Mapping[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(json.dumps(dict(entry), sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    _write_csv(csv_path, [dict(entry) for entry in entries], None)


def _status_entry(
    job: Mapping[str, str],
    *,
    status: str,
    reason: str = "",
    gpu: str = "",
    returncode: int | None = None,
) -> dict[str, Any]:
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job.get("job_id", ""),
        "variant_id": job.get("variant_id", ""),
        "run_id": job.get("run_id", ""),
        "family": job.get("family", ""),
        "checkpoint_selection": job.get("checkpoint_selection", ""),
        "stage": job.get("stage", ""),
        "status": status,
        "reason": reason or job.get("skip_reason", ""),
        "gpu": gpu,
        "cuda_visible_devices": gpu,
        "returncode": "" if returncode is None else int(returncode),
        "metrics_path": job.get("metrics_path", ""),
        "success_marker": job.get("success_marker", ""),
        "log_path": job.get("log_path", ""),
        "command": job.get("command", ""),
    }


def _write_concurrency_snapshot(
    output_root: Path,
    *,
    jobs: list[dict[str, str]],
    entries: list[Mapping[str, Any]],
    gpus: list[str],
    max_parallel: int,
) -> None:
    snapshot = {
        "gpu_list": gpus,
        "max_parallel": int(max_parallel),
        "job_count": len(jobs),
        "status_counts": dict(_count_by(entries, "status")),
        "why_less_than_max_parallel": [
            "dependency_blocking",
            "unavailable_or_skipped_rows",
            "fewer_ready_jobs_than_slots",
        ],
    }
    path = output_root / "status" / "concurrency_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def _load_status_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        stage = str(row.get("stage", ""))
        if run_id and stage:
            latest[f"{run_id}:{stage}"] = row
    return latest


def _record_status(
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, str]],
) -> str:
    if str(record.get("availability")) != "available":
        return "unavailable"
    status_row = statuses.get(f"{record.get('run_id')}:reeval")
    if status_row and str(status_row.get("status")) in {"failed", "blocked_dependency", "blocked_dependency_failed"}:
        return str(status_row.get("status"))
    if metrics:
        return "success"
    if status_row and str(status_row.get("status")).startswith("skipped"):
        return "skipped"
    return "missing"


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics"), dict):
            return dict(payload["metrics"])
        return dict(payload)
    return {}


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields or [])
    for row in rows:
        for key in row:
            if key not in field_list:
                field_list.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in field_list})


def _write_yaml(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _ensure_output_layout(output_root: Path) -> None:
    _assert_output_root_boundary(output_root)
    for rel in (
        "generated_configs/pretraining",
        "generated_configs/downstream",
        "generated_configs/eval",
        "jobs",
        "logs",
        "metrics",
        "checkpoints",
        "summary",
        "status",
        "runs",
    ):
        (output_root / rel).mkdir(parents=True, exist_ok=True)
    (output_root / OUTPUT_ROOT_MARKER).write_text(SWEEP_NAME + "\n", encoding="utf-8")


def _clean_output_root(output_root: Path) -> None:
    _assert_cleanable_output_root(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)


def _assert_output_root_boundary(path: str | Path) -> None:
    raw = Path(path)
    text = raw.as_posix()
    if raw.is_absolute():
        if raw.name != DEFAULT_OUTPUT_ROOT.name and OUTPUT_ROOT_MARKER not in {item.name for item in raw.parents}:
            # Absolute tmp paths are allowed by tests and explicit users after the generic forbidden-path checks below.
            pass
    elif not text.startswith("outputs/"):
        raise FullSweepManifestError("Full sweep output_root must stay under ignored outputs/ unless explicitly absolute.")
    forbidden_parts = {"/root/.container_env", "/etc/profile", "/etc/environment", "/root/.ssh", "/etc/ssh", "/etc/systemd"}
    resolved = raw.expanduser().resolve()
    for forbidden in forbidden_parts:
        forbidden_path = Path(forbidden).resolve()
        if resolved == forbidden_path or forbidden_path in resolved.parents:
            raise FullSweepManifestError(f"Full sweep output_root must not target system path {forbidden}.")


def _assert_cleanable_output_root(path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    project_root = Path.cwd().resolve()
    forbidden_exact = [
        project_root,
        project_root / "outputs",
        project_root / "outputs" / "analysis",
        Path("/root/.container_env"),
        Path("/etc/profile"),
        Path("/etc/environment"),
    ]
    forbidden_subtrees = [
        project_root / "dataset",
        project_root / "All_models",
        Path("/root/.ssh"),
        Path("/etc/ssh"),
        Path("/etc/systemd"),
    ]
    for item in forbidden_exact:
        resolved = item.resolve()
        if target == resolved:
            raise FullSweepRunnerError(f"Refusing to clean unsafe path: {target}")
    for item in forbidden_subtrees:
        resolved = item.resolve()
        if target == resolved or resolved in target.parents:
            raise FullSweepRunnerError(f"Refusing to clean unsafe path: {target}")
    marker = target / OUTPUT_ROOT_MARKER
    if target.name != DEFAULT_OUTPUT_ROOT.name and not marker.exists():
        raise FullSweepRunnerError(
            f"Refusing to clean {target}; it is not the default sweep root and has no {OUTPUT_ROOT_MARKER} marker."
        )


def _mode_defaults(manifest: Mapping[str, Any], mode: str) -> dict[str, Any]:
    mode_defaults = _mapping(manifest.get("mode_defaults", {}))
    if mode not in mode_defaults and mode != "full":
        raise FullSweepManifestError(f"Unsupported mode for {SWEEP_NAME}: {mode!r}.")
    defaults = _mapping(mode_defaults.get(mode, {}))
    return {
        "seeds": [int(seed) for seed in defaults.get("seeds", [17, 23, 42])],
        "checkpoint_selections": [str(item) for item in defaults.get("checkpoint_selections", ["primary", "best_top1"])],
        "eval_checkpoint_selections": [
            str(item) for item in defaults.get("eval_checkpoint_selections", defaults.get("checkpoint_selections", []))
        ],
        "gpu_list": [str(item) for item in defaults.get("gpu_list", [0, 1, 2, 3])],
        "max_parallel": int(defaults.get("max_parallel", 8)),
    }


def _comparison_protocol(manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(_mapping(manifest.get("comparison_protocol", {})))
    for field in STRICT_COMPARABILITY_FIELDS:
        if field == "seed":
            continue
        if field == "output_root":
            protocol.setdefault("output_root", str(manifest.get("output_root", DEFAULT_OUTPUT_ROOT)))
            continue
        protocol.setdefault(field, "declared")
    return protocol


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_list(value: Any) -> list[int]:
    return [int(item) for item in value or []]


def _number_list(value: Any) -> list[float]:
    return [float(item) for item in value or []]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value or []]


def _token_metadata(visual: Mapping[str, Any]) -> dict[str, Any]:
    visual_type = str(visual.get("type", "patch_vit"))
    image_size = _image_size(visual.get("image_size", [224, 224]))
    if visual_type == "none":
        grid = [0, 0]
        count = 0
        stride = [0, 0]
    elif visual_type == "overlap_patch":
        kernel = int(visual.get("kernel_size", visual.get("patch_size", 16)))
        stride_value = int(visual.get("stride", max(kernel // 2, 1)))
        grid = _conv_grid(image_size[0], image_size[1], kernel, stride_value)
        count = grid[0] * grid[1]
        stride = [stride_value, stride_value]
    elif visual_type == "cnn_feature_map":
        stage = str(visual.get("stage", "layer4"))
        grid = [14, 14] if stage == "layer3" else [7, 7]
        count = grid[0] * grid[1]
        stride = [16, 16] if stage == "layer3" else [32, 32]
    elif visual_type == "multi_scale_cnn":
        grid = [14, 14]
        count = 14 * 14 + 7 * 7
        stride = [16, 16]
    elif visual_type == "tinyvit_frame":
        grid = [1, 1]
        count = 1
        stride = [224, 224]
    elif visual_type == "conv_stem":
        effective = int(visual.get("effective_stride", _stem_stride(visual)))
        grid = _conv_grid(image_size[0], image_size[1], 1, effective)
        count = grid[0] * grid[1]
        stride = [effective, effective]
    else:
        patch = int(visual.get("patch_size", 16))
        grid = _conv_grid(image_size[0], image_size[1], patch, patch)
        count = grid[0] * grid[1]
        stride = [patch, patch]
    return {
        "image_size": image_size,
        "token_grid": grid,
        "token_count": int(visual.get("token_count", count)),
        "effective_stride": stride,
        "token_budget": int(visual.get("max_tokens", max(count, 256))),
        "position_interpolation": str(visual.get("positional_encoding", visual.get("position_interpolation", "learned_absolute"))),
        "token_source_type": visual_type,
    }


def _params_metadata(
    visual: Mapping[str, Any],
    pooler: Mapping[str, Any],
    *,
    freeze_policy: str = "",
    control: str = "",
) -> dict[str, Any]:
    visual_type = str(visual.get("type", "patch_vit"))
    token_count = _token_metadata(visual)["token_count"]
    backbone = str(visual.get("backbone", "none"))
    total = 250_000 + token_count * 512
    image_encoder_params = 0
    visual_context_encoder_params = 0
    if visual_type in {"cnn_feature_map", "multi_scale_cnn"}:
        total = 11_700_000 if backbone == "resnet18" else 21_800_000
        if visual_type == "multi_scale_cnn":
            total += 100_000
        if backbone == "resnet18" and visual_type == "cnn_feature_map" and str(visual.get("stage", "")) == "layer4":
            image_encoder_params = 11_240_000
            visual_context_encoder_params = 11_210_000
            total = 11_300_000
        elif backbone == "resnet18" and visual_type == "multi_scale_cnn":
            image_encoder_params = 14_050_000
            visual_context_encoder_params = 14_020_000
            total = 14_110_000
    elif visual_type == "tinyvit_frame":
        encoder_type = str(visual.get("encoder_type", "tinyvit_5m_scratch_rgb"))
        total = 5_500_000 if "5m" in encoder_type else 11_000_000
        image_encoder_params = total
        visual_context_encoder_params = total
    elif visual_type == "conv_stem":
        total = 450_000 + token_count * 256
    elif visual_type in {"local_token_mixing", "cvt"}:
        total = 320_000 + token_count * 512
    if control == "matched_trainable_params":
        total = 1_000_000
    trainable = total
    if freeze_policy in {"freeze_backbone_projection", "frozen_backbone_controls"} or str(visual.get("freeze_backbone")) == "True":
        trainable = max(50_000, total // 8)
    if freeze_policy == "unfreeze_layer4":
        trainable = max(100_000, total // 3)
    if visual_type == "patch_vit" and int(visual.get("patch_size", 16)) == 14 and str(pooler.get("type")) == "gps_query_attention":
        total = 177_000
        image_encoder_params = 117_000
        visual_context_encoder_params = 88_000
    pooler_params = 20_000 if str(pooler.get("type")) != "mean" else 0
    trainable = min(trainable, total)
    if not image_encoder_params:
        image_encoder_params = int(total)
    if not visual_context_encoder_params:
        visual_context_encoder_params = int(total)
    return {
        "total_params": int(total + pooler_params),
        "trainable_params": int(trainable + pooler_params),
        "visual_params": int(total),
        "image_encoder_params": int(image_encoder_params),
        "visual_context_encoder_params": int(visual_context_encoder_params),
        "pooler_params": int(pooler_params),
        "token_count": int(token_count),
        "attention_token_proxy": int(token_count * max(int(pooler.get("k_queries", 1) or 1), 1)),
        "compute_proxy": int((total // 1000) + token_count * max(int(pooler.get("k_queries", 1) or 1), 1)),
        "backbone_family": backbone,
        "parameter_count_source": CANDIDATE_PARAMETER_SOURCE,
    }


def _cnn_visual(backbone: str, token_source: str, pretrained_source: str, freeze_policy: str) -> dict[str, Any]:
    pretrained = pretrained_source == "imagenet"
    freeze_backbone = freeze_policy in {"freeze_backbone_projection", "unfreeze_layer4"}
    if token_source == "layer3+layer4":
        return {
            "type": "multi_scale_cnn",
            "backbone": backbone,
            "stages": ["layer3", "layer4"],
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "max_tokens": 256,
            "checkpoint_policy": "supervised_only_anchor",
        }
    return {
        "type": "cnn_feature_map",
        "backbone": backbone,
        "stage": token_source,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "max_tokens": 256,
        "checkpoint_policy": "supervised_only_anchor",
    }


def _tinyvit_visual(variant: str, pretrained_source: str) -> dict[str, Any]:
    normalized_variant = str(variant).strip().lower()
    normalized_pretrained = str(pretrained_source).strip().lower()
    if normalized_variant not in {"5m", "11m"}:
        raise FullSweepManifestError(f"Unknown TinyViT variant {variant!r}.")
    if normalized_pretrained not in {"scratch", "22k"}:
        raise FullSweepManifestError(f"Unknown TinyViT pretrained source {pretrained_source!r}.")
    encoder_type = f"tinyvit_{normalized_variant}_{'22k' if normalized_pretrained == '22k' else 'scratch'}_rgb"
    return {
        "type": "tinyvit_frame",
        "encoder_type": encoder_type,
        "variant": normalized_variant,
        "pretrained": normalized_pretrained == "22k",
        "pretrained_source": "imagenet22k_distill" if normalized_pretrained == "22k" else "scratch",
        "freeze_backbone": False,
        "allow_download": True,
        "max_tokens": 1,
        "token_count": 1,
        "checkpoint_policy": "supervised_only_anchor",
    }


def _hybrid_visual(tokenizer: str) -> dict[str, Any]:
    if tokenizer == "conv_stem_s16":
        return {"type": "conv_stem", "stem_strides": [2, 2, 4], "max_tokens": 256, "local_prior_mechanism": "conv_stem"}
    if tokenizer == "conv_stem_s8":
        return {"type": "conv_stem", "stem_strides": [2, 2, 2], "max_tokens": 784, "local_prior_mechanism": "conv_stem"}
    if tokenizer == "local_patch16":
        return {"type": "local_token_mixing", "patch_size": 16, "max_tokens": 256, "local_prior_mechanism": "depthwise_local_mixing"}
    if tokenizer == "local_patch14":
        return {"type": "local_token_mixing", "patch_size": 14, "max_tokens": 256, "local_prior_mechanism": "depthwise_local_mixing"}
    if tokenizer == "cvt_patch16":
        return {"type": "cvt", "patch_size": 16, "max_tokens": 256, "local_prior_mechanism": "cvt_depthwise_projection"}
    if tokenizer == "cvt_patch14":
        return {"type": "cvt", "patch_size": 14, "max_tokens": 256, "local_prior_mechanism": "cvt_depthwise_projection"}
    if tokenizer == "conv_stem_patch14":
        return {"type": "conv_stem", "stem_strides": [2, 7], "max_tokens": 256, "local_prior_mechanism": "conv_stem_patch14"}
    if tokenizer == "conv_stem_patch12":
        return {"type": "conv_stem", "stem_strides": [3, 4], "max_tokens": 361, "local_prior_mechanism": "conv_stem_patch12"}
    raise FullSweepManifestError(f"Unknown hybrid tokenizer {tokenizer!r}.")


def _student_visual(student: str) -> dict[str, Any]:
    if student == "patch14":
        return {"type": "patch_vit", "patch_size": 14, "max_tokens": 256}
    if student == "patch12":
        return {"type": "patch_vit", "patch_size": 12, "max_tokens": 361}
    if student == "overlap_k16_s8":
        return {"type": "overlap_patch", "kernel_size": 16, "stride": 8, "max_tokens": 729}
    if student == "conv_stem_s16":
        return _hybrid_visual("conv_stem_s16")
    if student == "local_patch14":
        return _hybrid_visual("local_patch14")
    raise FullSweepManifestError(f"Unknown teacher-guided student {student!r}.")


def _compute_control_visual(control: str) -> dict[str, Any]:
    if control == "matched_token_count":
        return {"type": "patch_vit", "patch_size": 16, "max_tokens": 256, "token_count": 49}
    if control == "frozen_backbone_controls":
        return {"type": "cnn_feature_map", "backbone": "resnet18", "stage": "layer4", "pretrained": False, "freeze_backbone": True}
    if control == "random_feature_controls":
        return {"type": "cnn_feature_map", "backbone": "resnet18", "stage": "layer4", "pretrained": False, "freeze_backbone": True}
    return {"type": "patch_vit", "patch_size": 16, "max_tokens": 256}


def _pooler_variant(variant: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if variant == "mean":
        return {"type": "mean", "output_mode": "frame"}, None
    if variant.startswith("gps_query_k"):
        pieces = variant.split("_")
        k_value = int(pieces[2][1:])
        output_mode = "tokens" if variant.endswith("_tokens") else "frame"
        return _gps_query_pooler(k_value, output_mode=output_mode), (
            {"type": "token_aware_transformer", "d_model": 64, "num_heads": 4, "num_layers": 1}
            if output_mode == "tokens"
            else None
        )
    if variant == "hybrid_residual_query":
        return {"type": "hybrid_residual_query", "content_queries": 2, "gps_queries": 2, "output_mode": "frame"}, None
    if variant == "token_aware_core":
        return _gps_query_pooler(2, output_mode="tokens"), {
            "type": "token_aware_transformer",
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
        }
    raise FullSweepManifestError(f"Unknown pooler/core variant {variant!r}.")


def _gps_query_pooler(k_queries: int, *, output_mode: str = "frame") -> dict[str, Any]:
    return {
        "type": "gps_query_attention",
        "k_queries": int(k_queries),
        "num_heads": 4,
        "condition_dim": 64,
        "latent_dim": 64,
        "dropout": 0.0,
        "condition_source": "projected_gps",
        "return_attention": False,
        "output_mode": output_mode,
    }


def _config_visual_encoder(record: Mapping[str, Any]) -> dict[str, Any]:
    visual = dict(_mapping(record.get("visual_encoder")))
    visual.setdefault("image_channels", 3)
    visual.setdefault("latent_dim", 64)
    visual.setdefault("image_profile", "rgb_imagenet")
    visual.setdefault("max_tokens", int(record["token_metadata"].get("token_budget", 256)))
    visual.setdefault("variant_id", str(record["variant_id"]))
    visual.setdefault("checkpoint_policy", str(record.get("checkpoint_policy", "exact_reuse")))
    return visual


def _config_pooler(record: Mapping[str, Any]) -> dict[str, Any]:
    pooler = dict(_mapping(record.get("pooler")))
    if pooler.get("type") not in {"none", None}:
        pooler.setdefault("latent_dim", 64)
        pooler.setdefault("condition_dim", 64)
    return pooler


def _config_sweep_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"distillation", "logits_kd", "rkd"}
    payload = {
        "name": SWEEP_NAME,
        "variant_id": str(record["variant_id"]),
        "run_id": str(record["run_id"]),
        "family": str(record["family"]),
        "stage_plan": str(record["stage_plan"]),
        "checkpoint_policy": str(record["checkpoint_policy"]),
        "checkpoint_selection": str(record["checkpoint_selection"]),
        "availability": str(record["availability"]),
        "strict_comparability": dict(record["strict_comparability"]),
        "token_metadata": dict(record["token_metadata"]),
        "params_metadata": dict(record["params_metadata"]),
    }
    text = json.dumps(payload, sort_keys=True)
    if any(token in text for token in forbidden):
        raise FullSweepManifestError("Generated sweep metadata unexpectedly referenced a removed KD route.")
    return payload


def _primary_run_id(record: Mapping[str, Any]) -> str:
    run_id = str(record["run_id"])
    selection = str(record.get("checkpoint_selection", "primary"))
    suffix = f"__{selection}"
    if selection and run_id.endswith(suffix):
        return f"{run_id[: -len(suffix)]}__primary"
    return run_id


def _run_stage_parent(record: Mapping[str, Any], *, run_id: str | None = None) -> Path:
    return Path(record["output_root"]) / "runs" / str(run_id or record["run_id"])


def _training_run_dir(record: Mapping[str, Any], stage: str, *, run_id: str | None = None) -> Path:
    return _run_stage_parent(record, run_id=run_id) / stage


def _eval_output_dir(record: Mapping[str, Any]) -> Path:
    return Path(record["output_root"]) / "eval" / str(record["run_id"])


def _eval_metrics_path(record: Mapping[str, Any]) -> str:
    return (_eval_output_dir(record) / "metrics.json").as_posix()


def _row_metrics_path(record: Mapping[str, Any]) -> str:
    if str(record.get("checkpoint_selection", "primary")) == "primary":
        return (_training_run_dir(record, "downstream") / "metrics.json").as_posix()
    return _eval_metrics_path(record)


def _base_downstream_config(record: Mapping[str, Any]) -> Path:
    base_config = str(record.get("base_config", ""))
    if base_config:
        return Path(base_config)
    if _visual_type(record) == "none":
        return BASE_GPS_ONLY_CONFIG
    if str(record["stage_plan"]) == "supervised_only" and str(record["family"]) == "existing_controls":
        return BASE_SUPERVISED_IMAGE_GPS_CONFIG
    return BASE_JEPA_DOWNSTREAM_CONFIG


def _stage1_checkpoint_path(record: Mapping[str, Any]) -> str:
    if str(record.get("stage_plan")) == "stage1_then_downstream":
        return str(_training_run_dir(record, "stage1", run_id=_primary_run_id(record)) / "checkpoints" / "best.pth")
    return ""


def _checkpoint_for_selection(record: Mapping[str, Any]) -> str:
    selection = str(record.get("checkpoint_selection", "primary"))
    name = "best_top1.pth" if selection == "best_top1" else "best.pth" if selection == "primary" else "last.pth"
    return str(_training_run_dir(record, "downstream", run_id=_primary_run_id(record)) / "checkpoints" / name)


def _visual_type(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("visual_encoder")).get("type", ""))


def _rel_config_path(config_path: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=config_path.parent.resolve())


def _image_size(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value), int(value)]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return [224, 224]


def _conv_grid(height: int, width: int, kernel: int, stride: int) -> list[int]:
    rows = math.floor((int(height) - int(kernel)) / int(stride) + 1)
    cols = math.floor((int(width) - int(kernel)) / int(stride) + 1)
    return [max(rows, 1), max(cols, 1)]


def _patch_token_count(image_size: int, patch_size: int) -> int:
    grid = _conv_grid(image_size, image_size, patch_size, patch_size)
    return int(grid[0] * grid[1])


def _overlap_token_count(image_size: int, kernel: int, stride: int) -> int:
    grid = _conv_grid(image_size, image_size, kernel, stride)
    return int(grid[0] * grid[1])


def _stem_stride(visual: Mapping[str, Any]) -> int:
    stride = 1
    for item in visual.get("stem_strides", [2, 2, 4]):
        stride *= int(item)
    return stride


def _pretrained_source(visual: Mapping[str, Any]) -> str:
    return "imagenet" if bool(visual.get("pretrained", False)) else "scratch"


def _freeze_policy(visual: Mapping[str, Any]) -> str:
    return "freeze_backbone_projection" if bool(visual.get("freeze_backbone", False)) else "full_ft"


def _availability_skip_reason(record: Mapping[str, Any], *, skip_unavailable: bool) -> str:
    if skip_unavailable:
        return "skip_unavailable_requested"
    return str(record.get("availability", "unavailable"))


def _unavailable_command(record: Mapping[str, Any]) -> str:
    reason = str(record.get("availability", "unavailable"))
    return f"{CONDA_PREFIX}python -c \"print('skipped {record['variant_id']}: {reason}')\""


def _normalize_gpu_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return items or ["0", "1", "2", "3"]


def _job_complete(job: Mapping[str, str]) -> bool:
    metrics = Path(str(job.get("metrics_path", "")))
    marker = Path(str(job.get("success_marker", "")))
    return metrics.exists() and marker.exists()


def _dry_run_command_line(entry: Mapping[str, Any], job: Mapping[str, str]) -> str:
    gpu = str(entry.get("gpu", ""))
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu and job.get("gpu_policy") != "none" else ""
    return prefix + str(job.get("command", ""))


def _read_metrics_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _read_metrics_value(metrics.get(key))
        if value is not None:
            return value
    return None


def _numeric(value: Any) -> float:
    parsed = _read_metrics_value(value)
    return 0.0 if parsed is None else float(parsed)


def _aggregate_seed_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_numeric(row.get("top1")) for row in rows if row.get("top1") is not None]
    dbas = [_numeric(row.get("dba")) for row in rows if row.get("dba") is not None]
    return {
        "seed_count": len({int(row.get("seed", 0)) for row in rows}),
        "seed_top1_mean": mean(values) if values else None,
        "seed_top1_std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
        "seed_dba_mean": mean(dbas) if dbas else None,
        "seed_dba_std": pstdev(dbas) if len(dbas) > 1 else 0.0 if dbas else None,
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_numeric(row.get("dba")), _numeric(row.get("top1"))), reverse=True)


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return counts


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _fmt(value: Any) -> str:
    parsed = _read_metrics_value(value)
    return "" if parsed is None else f"{parsed:.4f}"


def _num_tag(value: float) -> str:
    return str(value).replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and run the CNN/hybrid JEPA visual-prior full sweep.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--mode", default="full", choices=["full", "screening"])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-jobs", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--clean-output-root", action="store_true")
    parser.add_argument("--gpu-list", default="0,1,2,3")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT)
    if args.summarize:
        paths = generate_summary(output_root=output_root)
        print(json.dumps(paths, indent=2, sort_keys=True))
        return 0
    cleaned_before_generate = False
    if args.clean_output_root:
        _clean_output_root(output_root)
        cleaned_before_generate = True
    bundle = generate_runtime_bundle(
        manifest_path=args.manifest,
        output_root=output_root,
        mode=args.mode,
        families=args.families,
        force=args.force,
        skip_unavailable=args.skip_unavailable,
    )
    if args.dry_run or args.run_jobs:
        status = run_job_manifest(
            output_root=output_root,
            dry_run=args.dry_run,
            resume=args.resume,
            retry_failed=args.retry_failed,
            force_rerun=args.force_rerun,
            clean_output_root=False if cleaned_before_generate else args.clean_output_root,
            gpu_list=args.gpu_list,
            max_parallel=args.max_parallel,
        )
        bundle["runner"] = status
    hidden = {"manifest", "expanded_candidates"}
    print(json.dumps({key: value for key, value in bundle.items() if key not in hidden}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
