import hashlib
import json
import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.runtime import prepare_task_batch, run_model_step
from kd_sensing.eval.metrics import expected_calibration_error, reliability_error_stats
from kd_sensing.evaluation.metrics import beam_classification_circular_summary
from kd_sensing.eval.missing_patterns import (
    canonical_missing_pattern_name,
    get_default_missing_patterns,
    make_fixed_missing_mask,
    sample_eval_random_missing_mask,
)

COMPARABILITY_FIELDS = (
    "run_name",
    "method",
    "seed",
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "target_source",
    "modalities",
    "pattern_name",
    "difficulty_digest",
)


def evaluate_missing_matrix(
    model,
    dataloader,
    device,
    modalities: list[str],
    patterns: dict[str, list[int]] | None = None,
    random_missing: list[float] | None = None,
    prediction_index: int | str = "last",
    max_batches: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[dict]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    fixed_patterns = patterns or get_default_missing_patterns(modalities)
    results = []
    with torch.no_grad():
        for name, pattern in fixed_patterns.items():
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=name,
                    pattern=pattern,
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                    modalities=modalities,
                )
            )
        for p_missing in random_missing or []:
            pattern_name = f"random_{float(p_missing):g}"
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=pattern_name,
                    pattern=None,
                    num_modalities=len(modalities),
                    random_p=float(p_missing),
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                    modalities=modalities,
                )
            )
    if "avg_missing" not in {row.get("pattern") for row in results}:
        avg = _average_missing_results(results)
        if avg is not None:
            results.append(avg)
    _attach_comparability_metadata(results, modalities, cfg)
    return results


def evaluate_oracle_gate_matrix(
    model,
    dataloader,
    device,
    modalities: list[str],
    patterns: dict[str, list[int]] | None = None,
    random_missing: list[float] | None = None,
    prediction_index: int | str = "last",
    max_batches: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[dict]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    fixed_patterns = patterns or get_default_missing_patterns(modalities)
    results = []
    with torch.no_grad():
        for name, pattern in fixed_patterns.items():
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=name,
                    pattern=pattern,
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                    modalities=modalities,
                    oracle_gate=True,
                )
            )
        for p_missing in random_missing or []:
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=f"random_{float(p_missing):g}",
                    pattern=None,
                    num_modalities=len(modalities),
                    random_p=float(p_missing),
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                    modalities=modalities,
                    oracle_gate=True,
                )
            )
    if "avg_missing" not in {row.get("pattern") for row in results}:
        avg = _average_missing_results(results)
        if avg is not None:
            avg["oracle_chosen_modality_distribution"] = ""
            results.append(avg)
    _attach_comparability_metadata(results, modalities, cfg)
    for row in results:
        row["oracle_gate"] = "true"
    return results


def pattern_group_metadata(modalities: list[str], results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = results or []
    groups: dict[str, list[str]] = {
        "full": [],
        "single_missing": [],
        "multi_missing": [],
        "only_modality": [],
        "non_gps_only": [],
        "random_missing": [],
        "aggregate": [],
    }
    for row in rows:
        pattern = str(row.get("pattern_name", row.get("pattern", "")))
        group = _pattern_group(pattern)
        groups.setdefault(group, []).append(pattern)
    return {
        "modalities": list(modalities),
        "definitions": {
            "full": "all modalities available",
            "single_missing": "exactly one modality missing",
            "multi_missing": "two or more modalities missing while at least one remains",
            "only_modality": "exactly one modality available",
            "non_gps_only": "GPS missing and all non-GPS modalities available",
            "random_missing": "per-sample random missing mask",
            "aggregate": "derived group metric",
        },
        "members": {key: sorted(set(value)) for key, value in groups.items()},
    }


def _evaluate_pattern(
    model,
    dataloader,
    device: torch.device,
    *,
    pattern_name: str,
    pattern: list[int] | None,
    prediction_index: int | str,
    max_batches: int | None,
    cfg: dict[str, Any] | None,
    num_modalities: int | None = None,
    random_p: float | None = None,
    modalities: list[str] | None = None,
    oracle_gate: bool = False,
) -> dict[str, Any]:
    accumulator = _Accumulator()
    oracle_counts: Counter[str] = Counter()
    mask_label = ",".join(str(int(value)) for value in pattern) if pattern is not None else f"random_{random_p:g}"
    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch_size = _batch_size(raw_batch)
        metric_missing_mask = (
            make_fixed_missing_mask(batch_size, pattern, device=device)
            if pattern is not None
            else sample_eval_random_missing_mask(batch_size, int(num_modalities), float(random_p), device=device)
        )
        forward_missing_mask = None if cfg is not None and _is_full_pattern(pattern) else metric_missing_mask
        logits, target, diagnostics = _forward_batch(
            model,
            raw_batch,
            forward_missing_mask,
            device,
            prediction_index=prediction_index,
            cfg=cfg,
        )
        if oracle_gate:
            logits, chosen = _oracle_logits_from_diagnostics(diagnostics, logits, target, metric_missing_mask, modalities)
            oracle_counts.update(chosen)
        metrics = {
            "loss": float(F.cross_entropy(logits, target).detach().cpu().item()),
            **_beam_classification_metrics(logits, target, cfg),
            **reliability_error_stats(
                logits,
                target,
                global_reliability=_diagnostic_tensor(diagnostics, "global_reliability"),
                modality_reliability=_diagnostic_tensor(diagnostics, "modality_reliability"),
                missing_mask=metric_missing_mask,
            ),
            "ece": expected_calibration_error(logits, target),
        }
        accumulator.update(metrics, int(target.numel()))
    return {
        "pattern": pattern_name,
        "mask": mask_label,
        "num_samples": accumulator.num_samples,
        "sample_count": accumulator.num_samples,
        "count": accumulator.num_samples,
        **accumulator.mean(),
        **({"oracle_chosen_modality_distribution": _counter_payload(oracle_counts)} if oracle_gate else {}),
    }


def _forward_batch(
    model,
    raw_batch,
    missing_mask: torch.Tensor | None,
    device: torch.device,
    *,
    prediction_index: int | str,
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if cfg is not None:
        model_cfg = cfg["model"]["primary"]
        num_pred = int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)))
        step = run_model_step(
            model,
            cfg.get("experiment", {}).get("task", "image"),
            raw_batch,
            model_cfg=model_cfg,
            seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 8))),
            num_pred=num_pred,
            downsample_ratio=int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1))),
            device=device,
            extra_model_kwargs={"missing_mask": missing_mask} if missing_mask is not None else {},
        )
        return (
            _select_prediction(step.logits, prediction_index),
            _select_target(step.labels, prediction_index),
            step.model_output.diagnostics,
        )

    batch = _move_batch(prepare_task_batch(raw_batch), device)
    output = _direct_model_call(model, batch, missing_mask)
    model_output = adapt_model_output(output)
    return (
        _select_prediction(model_output.logits, prediction_index),
        _select_target(_extract_target(batch), prediction_index),
        model_output.diagnostics,
    )


def _direct_model_call(model, batch: dict[str, Any], missing_mask: torch.Tensor | None):
    if missing_mask is None:
        try:
            return model(batch)
        except TypeError:
            return model(**batch)
    try:
        return model(batch, missing_mask=missing_mask)
    except TypeError:
        return model(missing_mask=missing_mask, **batch)


def _is_full_pattern(pattern: list[int] | None) -> bool:
    return pattern is not None and all(int(value) == 1 for value in pattern)


def _select_prediction(logits: torch.Tensor, prediction_index: int | str) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, K] or [B, T, K], got {tuple(logits.shape)}.")
    index = _prediction_index(prediction_index, logits.shape[1])
    return logits[:, index, :]


def _select_target(target: torch.Tensor | None, prediction_index: int | str) -> torch.Tensor:
    if target is None:
        raise ValueError("target labels are required for evaluation.")
    target = target.to(dtype=torch.long)
    if target.ndim <= 1:
        return target.reshape(-1)
    index = _prediction_index(prediction_index, target.shape[1])
    return target[:, index].reshape(-1)


def _prediction_index(value: int | str, length: int) -> int:
    if value == "last":
        return length - 1
    if value == "first":
        return 0
    index = int(value)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise ValueError(f"prediction_index {value!r} out of range for {length} predictions.")
    return index


def _oracle_logits_from_diagnostics(
    diagnostics: dict[str, Any],
    fallback_logits: torch.Tensor,
    target: torch.Tensor,
    available_mask: torch.Tensor,
    modalities: list[str] | None,
) -> tuple[torch.Tensor, list[str]]:
    logits = diagnostics.get("pcpg_unimodal_logits")
    if not torch.is_tensor(logits):
        logits = diagnostics.get("unimodal_logits")
    if not torch.is_tensor(logits) or logits.ndim != 3:
        return fallback_logits, ["fused"] * int(fallback_logits.shape[0])
    mask = available_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape[:2]:
        diag_mask = diagnostics.get("pcpg_available_mask", diagnostics.get("missing_mask"))
        mask = diag_mask.to(device=logits.device, dtype=torch.bool) if torch.is_tensor(diag_mask) else torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
    predictions = logits.argmax(dim=-1)
    target = target.to(device=logits.device, dtype=torch.long).view(-1, 1)
    distance = (predictions - target).abs()
    distance = torch.minimum(distance, logits.shape[-1] - distance)
    distance = distance.masked_fill(~mask, torch.iinfo(distance.dtype).max)
    chosen = distance.argmin(dim=1)
    oracle_logits = logits[torch.arange(logits.shape[0], device=logits.device), chosen, :]
    names = list(modalities or [f"modality_{index}" for index in range(logits.shape[1])])
    return oracle_logits, [names[int(index)] if int(index) < len(names) else f"modality_{int(index)}" for index in chosen.detach().cpu()]


def _counter_payload(counter: Counter[str]) -> str:
    total = sum(counter.values())
    if total <= 0:
        return ""
    return json.dumps({key: count / total for key, count in sorted(counter.items())}, sort_keys=True)


def _extract_target(batch: dict[str, Any]) -> torch.Tensor:
    for key in ("target", "target_beam", "beam_label", "label", "labels", "y"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError("Batch must contain one of target, target_beam, beam_label, label, labels, or y.")


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    raise ValueError("Could not infer batch size from dataloader batch.")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _diagnostic_tensor(diagnostics: dict[str, Any], key: str) -> torch.Tensor | None:
    value = diagnostics.get(key)
    return value if torch.is_tensor(value) else None


class _Accumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.sums: dict[str, float] = {}

    def update(self, metrics: dict[str, float], count: int) -> None:
        self.num_samples += int(count)
        for key, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                self.sums[key] = self.sums.get(key, 0.0) + value * int(count)

    def mean(self) -> dict[str, float]:
        keys = [
            "loss",
            "top1",
            "top3",
            "top5",
            "within_3",
            "adba",
            "mae",
            "mean_confidence",
            "mean_global_reliability",
            "mean_global_reliability_correct",
            "mean_global_reliability_wrong",
            "mean_modality_reliability",
            "mean_available_modality_reliability",
            "ece",
        ]
        return {
            key: (self.sums[key] / self.num_samples if self.num_samples and key in self.sums else math.nan)
            for key in keys
        }


def _beam_classification_metrics(logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any] | None) -> dict[str, float]:
    eval_cfg = (cfg or {}).get("evaluation", {}) if isinstance(cfg, dict) else {}
    legacy_eval_cfg = (cfg or {}).get("eval", {}) if isinstance(cfg, dict) else {}
    circular = bool(legacy_eval_cfg.get("beam_distance_circular", eval_cfg.get("beam_distance_circular", True)))
    summary = beam_classification_circular_summary(
        logits,
        target,
        num_beams=int(logits.shape[-1]),
        dba_delta=float(eval_cfg.get("dba_delta", 5)),
        distance_mode="circular" if circular else "linear",
    )
    return {
        "top1": float(summary.get("top1", math.nan)),
        "top3": float(summary.get("top3", math.nan)),
        "top5": float(summary.get("top5", math.nan)),
        "within_3": float(summary.get("within_3", math.nan)),
        "adba": float(summary.get("DBA", math.nan)),
        "mae": float(summary.get("mean_error", summary.get("mean_circular_error", math.nan))),
    }


def _average_missing_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in results
        if str(row.get("pattern", "")) != "full"
        and (
            str(row.get("pattern", "")).startswith("missing_")
            or str(row.get("pattern", "")).startswith("only_")
            or str(row.get("pattern", "")).endswith("_only")
            or str(row.get("pattern", "")) == "non_gps_only"
        )
    ]
    total = sum(int(row.get("num_samples", row.get("sample_count", 0)) or 0) for row in rows)
    if not rows or total <= 0:
        return None
    sample_count = max(int(row.get("num_samples", row.get("sample_count", 0)) or 0) for row in rows)
    averaged = {
        "pattern": "avg_missing",
        "mask": "aggregate",
        "num_samples": sample_count,
        "sample_count": sample_count,
        "count": sample_count,
    }
    for key in (
        "loss",
        "top1",
        "top3",
        "top5",
        "within_3",
        "adba",
        "mae",
        "mean_confidence",
        "mean_global_reliability",
        "mean_global_reliability_correct",
        "mean_global_reliability_wrong",
        "mean_modality_reliability",
        "mean_available_modality_reliability",
        "ece",
    ):
        value = _weighted_mean(rows, key)
        if value is not None:
            averaged[key] = value
    return averaged


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = row.get(key)
        count = int(row.get("num_samples", row.get("sample_count", 0)) or 0)
        if count <= 0 or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        numerator += float(value) * count
        denominator += count
    return numerator / denominator if denominator else None


def _attach_comparability_metadata(
    results: list[dict[str, Any]],
    modalities: list[str],
    cfg: dict[str, Any] | None,
) -> None:
    base = _base_comparability_metadata(cfg, modalities)
    for row in results:
        pattern = str(row.get("pattern", ""))
        row["pattern_name"] = pattern
        row["condition_id"] = pattern
        row["pattern_group"] = _pattern_group(pattern)
        row["is_aggregate"] = str(row["pattern_group"] == "aggregate").lower()
        row["modalities"] = "|".join(modalities)
        available, missing = _availability_from_mask(row.get("mask"), modalities)
        row["available_modalities"] = "|".join(available)
        row["missing_modalities"] = "|".join(missing)
        for key, value in base.items():
            row.setdefault(key, value)
        row["difficulty_digest"] = row.get("difficulty_digest") or _stable_digest(
            {
                "pattern": pattern,
                "mask": row.get("mask"),
                "modalities": modalities,
                "difficulty": base.get("difficulty_digest", ""),
            }
        )
        missing_fields = [field for field in COMPARABILITY_FIELDS if row.get(field) in (None, "", [], {})]
        row["comparability_status"] = "strict" if not missing_fields else "incomplete"
        row["comparability_missing_fields"] = ";".join(missing_fields)
        if missing_fields:
            warning = "missing_comparability_fields:" + ",".join(missing_fields)
            row["warnings"] = ";".join(item for item in (str(row.get("warnings") or ""), warning) if item)


def _base_comparability_metadata(cfg: dict[str, Any] | None, modalities: list[str]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {
            "run_name": "",
            "method": "",
            "seed": "",
            "split": "",
            "label_space": "",
            "metric_profile": "u_mask_beam_jepa_eval_matrix_topk_dba",
            "target_source": "",
            "difficulty_digest": "",
        }
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    output = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    evaluation = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), dict) else {}
    paper_metadata = primary.get("paper_metadata", {}) if isinstance(primary.get("paper_metadata"), dict) else {}
    comparability = cfg.get("comparability", {}) if isinstance(cfg.get("comparability"), dict) else {}
    run_name = str(
        comparability.get("run_name")
        or experiment.get("run_name")
        or experiment.get("name")
        or output.get("run_name")
        or ""
    )
    method = str(comparability.get("method") or paper_metadata.get("model_group") or paper_metadata.get("method") or run_name)
    num_classes = primary.get("num_classes", model.get("num_classes"))
    label_space = str(comparability.get("label_space") or (f"beam{int(num_classes)}" if num_classes not in (None, "") else ""))
    difficulty = cfg.get("difficulty", {}) if isinstance(cfg.get("difficulty"), dict) else {}
    difficulty_digest = "|".join(
        str(profile.get("digest"))
        for profile in difficulty.get("profiles", [])
        if isinstance(profile, dict) and profile.get("digest")
    )
    return {
        "run_name": run_name,
        "method": method,
        "seed": experiment.get("seed", comparability.get("seed", "")),
        "split": str(comparability.get("split") or evaluation.get("split") or dataset.get("split") or ""),
        "label_space": label_space,
        "metric_profile": str(
            comparability.get("metric_profile")
            or evaluation.get("metric_profile")
            or "u_mask_beam_jepa_eval_matrix_topk_dba"
        ),
        "target_source": str(
            comparability.get("target_source")
            or dataset.get("beam_target_source")
            or data.get("beam_target_source")
            or ""
        ),
        "difficulty_digest": str(comparability.get("difficulty_digest") or difficulty_digest),
    }


def _pattern_group(pattern: str) -> str:
    try:
        name = canonical_missing_pattern_name(pattern) if pattern else ""
    except ValueError:
        name = str(pattern)
    if name == "full":
        return "full"
    if name in {"avg_missing", "overall_mean", "balanced"}:
        return "aggregate"
    if name == "non_gps_only":
        return "non_gps_only"
    if name.startswith("random_"):
        return "random_missing"
    if name.endswith("_only"):
        return "only_modality"
    if name.startswith("missing_"):
        missing = [item for item in name.removeprefix("missing_").split("_") if item]
        return "single_missing" if len(missing) == 1 else "multi_missing"
    return "custom"


def _availability_from_mask(mask: Any, modalities: list[str]) -> tuple[list[str], list[str]]:
    if not isinstance(mask, str) or mask in {"", "aggregate"} or mask.startswith("random_"):
        return ([], [])
    values = [item.strip() for item in mask.split(",") if item.strip() != ""]
    if len(values) != len(modalities):
        return ([], [])
    available = [modality for modality, keep in zip(modalities, values) if keep == "1"]
    missing = [modality for modality, keep in zip(modalities, values) if keep == "0"]
    return available, missing


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
