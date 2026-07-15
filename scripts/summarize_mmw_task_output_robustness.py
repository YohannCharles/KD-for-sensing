#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("T2", "amber_full", "rmbp_mm")
T2_ABLATION_METHODS = ("T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
SUPPORTED_METHODS = (*METHODS, *T2_ABLATION_METHODS)
METHOD_LABELS = {
    "T2": "T2",
    "amber_full": "AMBER-Full",
    "rmbp_mm": "RMBP-MM",
    "T2-NoBPA": "T2 w/o BPA",
    "T2-BPA2CMA": "T2: BPA to CMA",
    "T2-Linear": "T2: linear topology",
    "T2-CLS": "T2: classifier",
    "T2-CLS-CMA": "T2: classifier + CMA",
}
BASELINE_SCOPES = {
    "amber_full": "local adaptation, not paper-equivalent reproduction",
    "rmbp_mm": "channel-attention local adaptation, out-of-paper-scope diagnostic",
}
EVIDENCE_SCOPES = {
    "T2": ("project_mainline", "mainline_local_validation"),
    "amber_full": ("amber_full_local_adaptation", "local_adaptation_diagnostic"),
    "rmbp_mm": ("rmbp_mm_channel_attention_local", "out_of_paper_scope_diagnostic"),
    **{
        method: ("project_mainline_t2_ablation", "paired_objective_topology_head_ablation")
        for method in T2_ABLATION_METHODS
    },
}
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
NUM_CLASSES = 64
METRICS = (
    "top1",
    "relative_clean_top1",
    "exact",
    "within1",
    "within3",
    "mae",
    "true_margin_delta",
    "normalized_js",
)
IDENTITY_FIELDS = (
    "condition",
    "scene",
    "sample_csv_sha256",
    "sample_ids",
    "labels",
    "rates",
    "mask_digests",
    "cache_checksums",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paired MMW task-output robustness evidence.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--expected-domains", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_task_outputs(
        Path(args.raw_root),
        Path(args.output_dir),
        methods=tuple(_csv(args.methods)),
        seeds=tuple(int(value) for value in _csv(args.seeds)),
        expected_domains=int(args.expected_domains),
    )
    return 0


def summarize_task_outputs(
    raw_root: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = (1, 2, 3),
    expected_domains: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    bundles = load_bundles(raw_root, methods=methods, seeds=seeds, expected_domains=expected_domains)
    domain_mask_rows, coverage_rows = build_domain_mask_metrics(bundles)
    domain_rate_rows = aggregate_domain_rates(domain_mask_rows)
    seed_rate_rows = aggregate_seed_rates(domain_rate_rows)
    multiseed_rows = aggregate_multiseed_rates(seed_rate_rows, requested_seed_count=len(seeds))
    delta_rows = build_t2_baseline_domain_deltas(domain_rate_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "domain_mask_metrics.csv": domain_mask_rows,
        "domain_rate_metrics.csv": domain_rate_rows,
        "seed_rate_metrics.csv": seed_rate_rows,
        "multiseed_rate_summary.csv": multiseed_rows,
        "common_clean_coverage.csv": coverage_rows,
        "t2_baseline_domain_deltas.csv": delta_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / name, rows)
    plot_all_sample_curves(multiseed_rows, output_dir / "all_sample_robustness_curves.png")
    plot_common_clean_curves(multiseed_rows, output_dir / "common_clean_robustness_curves.png")
    plot_t2_baseline_heatmap(delta_rows, output_dir / "t2_baseline_15domain_heatmap.png")
    (output_dir / "summary.md").write_text(
        render_markdown(multiseed_rows, coverage_rows, methods=methods, seeds=seeds),
        encoding="utf-8",
    )
    return {
        "domain_mask": domain_mask_rows,
        "domain_rate": domain_rate_rows,
        "seed_rate": seed_rate_rows,
        "multiseed": multiseed_rows,
        "coverage": coverage_rows,
        "deltas": delta_rows,
    }


def load_bundles(
    raw_root: Path,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    expected_domains: int,
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if not methods or not seeds or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("methods and seeds must be non-empty and unique")
    bundles: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed in seeds:
        bundles[seed] = {}
        for method in methods:
            paths = sorted((raw_root / f"seed{seed}" / method / "domains").glob("*.npz"))
            if len(paths) != expected_domains:
                raise ValueError(
                    f"seed={seed} method={method}: expected {expected_domains} domain artifacts, found {len(paths)}"
                )
            domains = {}
            for path in paths:
                domain = load_domain_output(path, expected_seed=seed)
                domain_id = domain["domain_id"]
                if domain_id in domains:
                    raise ValueError(f"seed={seed} method={method}: duplicate domain_id={domain_id!r}")
                domains[domain_id] = domain
            _validate_method_seed_provenance(domains, seed=seed, method=method)
            _validate_worker_checkpoint_provenance(
                raw_root / f"seed{seed}" / method,
                domains,
                seed=seed,
                method=method,
            )
            bundles[seed][method] = domains
    validate_strict_alignment(bundles, methods=methods, seeds=seeds)
    return bundles


def load_domain_output(path: Path, *, expected_seed: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(
            {
                "logits",
                "predictions",
                "labels",
                "sample_ids",
                "rates",
                "mask_digests",
                "cache_checksums",
                "domain_id",
                "condition",
                "scene",
                "sample_csv_sha256",
                "checkpoint_sha256",
                "seed",
            }
            - set(payload.files)
        )
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        domain = {key: payload[key].copy() for key in payload.files}
    for key in ("domain_id", "condition", "scene", "sample_csv_sha256", "checkpoint_sha256"):
        domain[key] = _scalar_text(domain[key], path=path, field=key)
    domain["seed"] = int(np.asarray(domain["seed"]).item())
    domain["sample_ids"] = np.asarray(domain["sample_ids"]).astype(str).reshape(-1)
    domain["mask_digests"] = np.asarray(domain["mask_digests"]).astype(str).reshape(-1)
    domain["cache_checksums"] = np.asarray(domain["cache_checksums"]).astype(str).reshape(-1)
    domain["labels"] = np.asarray(domain["labels"], dtype=np.int64).reshape(-1)
    domain["predictions"] = np.asarray(domain["predictions"], dtype=np.int64)
    domain["rates"] = np.asarray(domain["rates"], dtype=np.float64).reshape(-1)
    logits = np.asarray(domain["logits"])
    if logits.dtype != np.float32:
        raise ValueError(f"{path}: logits must be float32, got {logits.dtype}")
    domain["logits"] = logits
    _validate_domain_output(domain, path=path, expected_seed=expected_seed)
    return domain


def _validate_domain_output(domain: dict[str, Any], *, path: Path, expected_seed: int) -> None:
    logits = domain["logits"]
    predictions = domain["predictions"]
    labels = domain["labels"]
    sample_ids = domain["sample_ids"]
    condition_count = int(logits.shape[0]) if logits.ndim == 3 else -1
    if logits.ndim != 3 or logits.shape[2] != NUM_CLASSES or condition_count != 65:
        raise ValueError(f"{path}: logits must have shape [65,N,64], got {logits.shape}")
    if predictions.shape != logits.shape[:2]:
        raise ValueError(f"{path}: predictions shape {predictions.shape} does not match logits {logits.shape[:2]}")
    if labels.shape != (logits.shape[1],) or sample_ids.shape != labels.shape:
        raise ValueError(f"{path}: labels/sample_ids do not align with N={logits.shape[1]}")
    for field in ("rates", "mask_digests", "cache_checksums"):
        if np.asarray(domain[field]).shape != (condition_count,):
            raise ValueError(f"{path}: {field} must have shape [{condition_count}]")
    if domain["seed"] != int(expected_seed):
        raise ValueError(f"{path}: seed={domain['seed']} does not match path seed={expected_seed}")
    if not np.isfinite(logits).all():
        raise ValueError(f"{path}: logits contain non-finite values")
    if np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError(f"{path}: labels must be in [0,{NUM_CLASSES - 1}]")
    if not np.array_equal(predictions, logits.argmax(axis=-1)):
        raise ValueError(f"{path}: predictions do not equal logits argmax")
    if any(not value for value in sample_ids) or len(set(sample_ids.tolist())) != labels.size:
        raise ValueError(f"{path}: sample_ids must be non-empty and unique within domain")
    if any(not value for value in domain["mask_digests"]) or any(not value for value in domain["cache_checksums"]):
        raise ValueError(f"{path}: mask/cache identity values must be non-empty")
    expected_counts = {0.0: 1, 0.2: 16, 0.4: 16, 0.6: 16, 0.8: 16}
    observed_counts = {rate: int(np.isclose(domain["rates"], rate, atol=1e-6).sum()) for rate in RATES}
    if observed_counts != expected_counts or not np.all(
        np.logical_or.reduce([np.isclose(domain["rates"], rate, atol=1e-6) for rate in RATES])
    ):
        raise ValueError(f"{path}: expected clean + 16 masks at each 20/40/60/80 rate, got {observed_counts}")
    for rate, expected_count in expected_counts.items():
        selected = np.isclose(domain["rates"], rate, atol=1e-6)
        unique_masks = len(set(domain["mask_digests"][selected].tolist()))
        if unique_masks != expected_count:
            raise ValueError(
                f"{path}: rate={rate} mask_digests must contain {expected_count} unique values, got {unique_masks}"
            )
        if len(set(domain["cache_checksums"][selected].tolist())) != 1:
            raise ValueError(f"{path}: rate={rate} must use one cache checksum")


def _validate_method_seed_provenance(
    domains: dict[str, dict[str, Any]],
    *,
    seed: int,
    method: str,
) -> None:
    reference = next(iter(domains.values()))
    for domain_id, domain in domains.items():
        for field in ("rates", "mask_digests", "cache_checksums", "checkpoint_sha256"):
            if not _identity_equal(reference[field], domain[field]):
                raise ValueError(
                    f"seed={seed} method={method}: domain provenance mismatch field={field} domain={domain_id}"
                )


def _validate_worker_checkpoint_provenance(
    method_dir: Path,
    domains: dict[str, dict[str, Any]],
    *,
    seed: int,
    method: str,
) -> None:
    worker_paths = sorted(method_dir.glob("worker_*_of_*.json"))
    if not worker_paths:
        raise ValueError(f"seed={seed} method={method}: worker provenance missing")
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_paths]
    shard_counts = {int(worker.get("domain_shard_count", -1)) for worker in workers}
    shard_indices = {int(worker.get("domain_shard_index", -1)) for worker in workers}
    if shard_counts != {len(workers)} or shard_indices != set(range(len(workers))):
        raise ValueError(f"seed={seed} method={method}: worker shard provenance incomplete")
    if any(worker.get("method") != method or int(worker.get("seed", -1)) != seed for worker in workers):
        raise ValueError(f"seed={seed} method={method}: worker method/seed provenance mismatch")
    checkpoints = {str(worker.get("checkpoint", "")) for worker in workers}
    checksums = {str(worker.get("checkpoint_sha256", "")) for worker in workers}
    completed = {domain for worker in workers for domain in worker.get("completed_domains", [])}
    if len(checkpoints) != 1 or len(checksums) != 1 or completed != set(domains):
        raise ValueError(f"seed={seed} method={method}: worker checkpoint/domain provenance mismatch")
    checkpoint = Path(next(iter(checkpoints)))
    if (
        checkpoint.name != "last.pth"
        or checkpoint.parent.name != "checkpoints"
        or checkpoint.parent.parent.name != f"seed{seed}"
        or checkpoint.parent.parent.parent.name != method
        or not checkpoint.is_file()
    ):
        raise ValueError(f"seed={seed} method={method}: worker checkpoint is not method/seed last.pth")
    expected_checksum = next(iter(checksums))
    artifact_checksums = {str(domain["checkpoint_sha256"]) for domain in domains.values()}
    if artifact_checksums != {expected_checksum} or _sha256(checkpoint) != expected_checksum:
        raise ValueError(f"seed={seed} method={method}: checkpoint SHA256 mismatch")


def validate_strict_alignment(
    bundles: dict[int, dict[str, dict[str, dict[str, Any]]]],
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    reference: dict[str, dict[str, Any]] | None = None
    reference_name = ""
    for seed in seeds:
        for method in methods:
            domains = bundles[seed][method]
            if reference is None:
                reference = domains
                reference_name = f"seed={seed} method={method}"
                continue
            if set(domains) != set(reference):
                missing = sorted(set(reference) - set(domains))
                extra = sorted(set(domains) - set(reference))
                raise ValueError(
                    f"seed={seed} method={method}: domain identity mismatch vs {reference_name}; "
                    f"missing={missing}, extra={extra}"
                )
            for domain_id in sorted(reference):
                for field in IDENTITY_FIELDS:
                    left = reference[domain_id][field]
                    right = domains[domain_id][field]
                    if not _identity_equal(left, right):
                        detail = _first_mismatch(left, right)
                        raise ValueError(
                            f"seed={seed} method={method} domain={domain_id}: identity mismatch field={field}{detail}"
                        )


def build_domain_mask_metrics(
    bundles: dict[int, dict[str, dict[str, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for seed, method_bundles in sorted(bundles.items()):
        methods = tuple(method for method in METHODS if method in method_bundles)
        domain_ids = sorted(next(iter(method_bundles.values())))
        for domain_id in domain_ids:
            domains = {method: method_bundles[method][domain_id] for method in methods}
            labels = next(iter(domains.values()))["labels"]
            clean_indices = {
                method: int(np.flatnonzero(np.isclose(domain["rates"], 0.0, atol=1e-6))[0])
                for method, domain in domains.items()
            }
            clean_correct = {
                method: domain["predictions"][clean_indices[method]] == labels
                for method, domain in domains.items()
            }
            scopes = _common_scopes(clean_correct, labels.size)
            reference = next(iter(domains.values()))
            for scope, (scope_kind, scope_methods, subset) in scopes.items():
                coverage_rows.append(
                    {
                        "seed": seed,
                        "domain_id": domain_id,
                        "condition": reference["condition"],
                        "scene": reference["scene"],
                        "scope": scope,
                        "scope_kind": scope_kind,
                        "scope_methods": ",".join(scope_methods),
                        "scope_evidence": ";".join(
                            f"{member}={EVIDENCE_SCOPES[member][0]}" for member in scope_methods
                        ),
                        "sample_count": int(labels.size),
                        "common_count": int(subset.sum()),
                        "coverage": float(subset.mean()),
                        "status": "available" if subset.any() else "unavailable_empty_common_clean",
                    }
                )
                for method in scope_methods:
                    domain = domains[method]
                    clean_index = clean_indices[method]
                    clean_logits = domain["logits"][clean_index]
                    clean_prediction = domain["predictions"][clean_index]
                    clean_top1 = float(np.mean(clean_prediction[subset] == labels[subset])) if subset.any() else math.nan
                    for condition_index, rate in enumerate(domain["rates"]):
                        metrics = task_metrics(
                            domain["logits"][condition_index],
                            domain["predictions"][condition_index],
                            labels,
                            clean_logits,
                            clean_top1=clean_top1,
                            subset=subset,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "reproduction_scope": EVIDENCE_SCOPES[method][0],
                                "paper_equivalent": False,
                                "temporal_result_scope": EVIDENCE_SCOPES[method][1],
                                "domain_id": domain_id,
                                "condition": domain["condition"],
                                "scene": domain["scene"],
                                "sample_csv_sha256": domain["sample_csv_sha256"],
                                "checkpoint_sha256": domain["checkpoint_sha256"],
                                "rate": float(rate),
                                "mask_digest": str(domain["mask_digests"][condition_index]),
                                "cache_checksum": str(domain["cache_checksums"][condition_index]),
                                "scope": scope,
                                "scope_kind": scope_kind,
                                "scope_methods": ",".join(scope_methods),
                                "sample_count": int(labels.size),
                                "subset_count": int(subset.sum()),
                                "coverage": float(subset.mean()),
                                "status": "available" if subset.any() else "unavailable_empty_common_clean",
                                **metrics,
                            }
                        )
    return rows, coverage_rows


def _common_scopes(
    clean_correct: dict[str, np.ndarray],
    sample_count: int,
) -> dict[str, tuple[str, tuple[str, ...], np.ndarray]]:
    result: dict[str, tuple[str, tuple[str, ...], np.ndarray]] = {}
    all_samples = np.ones(sample_count, dtype=bool)
    for method in clean_correct:
        result[f"all:{method}"] = ("all", (method,), all_samples)
    if "T2" in clean_correct:
        for baseline in ("amber_full", "rmbp_mm"):
            if baseline in clean_correct:
                members = ("T2", baseline)
                result[f"pairwise:T2:{baseline}"] = (
                    "pairwise_common_clean",
                    members,
                    clean_correct["T2"] & clean_correct[baseline],
                )
    if all(method in clean_correct for method in METHODS):
        result["three_way:T2:amber_full:rmbp_mm"] = (
            "three_way_common_clean",
            METHODS,
            np.logical_and.reduce([clean_correct[method] for method in METHODS]),
        )
    return result


def task_metrics(
    logits: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    clean_logits: np.ndarray,
    *,
    clean_top1: float,
    subset: np.ndarray,
) -> dict[str, float]:
    if not subset.any():
        return {metric: math.nan for metric in METRICS}
    selected_predictions = np.asarray(predictions)[subset]
    selected_labels = np.asarray(labels)[subset]
    distances = circular_beam_distance(selected_predictions, selected_labels)
    top1 = float(np.mean(selected_predictions == selected_labels))
    margin_delta = true_class_margin(logits, labels) - true_class_margin(clean_logits, labels)
    js = normalized_js_divergence(clean_logits, logits)
    return {
        "top1": top1,
        "relative_clean_top1": float(top1 / clean_top1) if math.isfinite(clean_top1) and clean_top1 > 0 else math.nan,
        "exact": top1,
        "within1": float(np.mean(distances <= 1)),
        "within3": float(np.mean(distances <= 3)),
        "mae": float(np.mean(distances)),
        "true_margin_delta": float(np.mean(margin_delta[subset])),
        "normalized_js": float(np.mean(js[subset])),
    }


def circular_beam_distance(left: np.ndarray, right: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    delta = np.abs(np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64))
    return np.minimum(delta, int(num_classes) - delta)


def true_class_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.size:
        raise ValueError("logits must be [N,C] and align with labels")
    if np.any((target < 0) | (target >= values.shape[1])):
        raise ValueError("labels fall outside logits class dimension")
    rows = np.arange(target.size)
    true_values = values[rows, target]
    alternatives = values.copy()
    alternatives[rows, target] = -np.inf
    return true_values - alternatives.max(axis=1)


def normalized_js_divergence(clean_logits: np.ndarray, missing_logits: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean_logits, dtype=np.float64)
    missing = np.asarray(missing_logits, dtype=np.float64)
    if clean.shape != missing.shape or clean.ndim != 2:
        raise ValueError("clean and missing logits must have the same [N,C] shape")
    p = _softmax(clean)
    q = _softmax(missing)
    middle = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    kl_p = np.sum(p * (np.log(np.clip(p, tiny, None)) - np.log(np.clip(middle, tiny, None))), axis=1)
    kl_q = np.sum(q * (np.log(np.clip(q, tiny, None)) - np.log(np.clip(middle, tiny, None))), axis=1)
    return np.clip(0.5 * (kl_p + kl_q) / math.log(2.0), 0.0, 1.0)


def aggregate_domain_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("seed", "method", "domain_id", "scope", "rate")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for key, selected in sorted(groups.items(), key=lambda item: item[0]):
        subset_counts = {int(row["subset_count"]) for row in selected}
        sample_counts = {int(row["sample_count"]) for row in selected}
        if len(subset_counts) != 1 or len(sample_counts) != 1:
            raise ValueError(f"Common-clean subset changed across masks for group={key}")
        first = selected[0]
        item = {
            "seed": first["seed"],
            "method": first["method"],
            "method_label": first["method_label"],
            "reproduction_scope": first["reproduction_scope"],
            "paper_equivalent": first["paper_equivalent"],
            "temporal_result_scope": first["temporal_result_scope"],
            "domain_id": first["domain_id"],
            "condition": first["condition"],
            "scene": first["scene"],
            "rate": first["rate"],
            "scope": first["scope"],
            "scope_kind": first["scope_kind"],
            "scope_methods": first["scope_methods"],
            "mask_count": len(selected),
            "sample_count": first["sample_count"],
            "subset_count": first["subset_count"],
            "coverage": first["coverage"],
            "status": first["status"],
        }
        item.update({metric: _finite_mean(row[metric] for row in selected) for metric in METRICS})
        result.append(item)
    return result


def aggregate_seed_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("seed", "method", "scope", "rate")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for _, selected in sorted(groups.items(), key=lambda item: item[0]):
        first = selected[0]
        eligible = [row for row in selected if int(row["subset_count"]) > 0]
        empty_count = len(selected) - len(eligible)
        nonfinite_metrics = sorted(
            metric
            for metric in METRICS
            if any(not math.isfinite(float(row[metric])) for row in selected)
        )
        available = empty_count == 0 and not nonfinite_metrics
        metric_rows = selected if available else []
        item = {
            "seed": first["seed"],
            "method": first["method"],
            "method_label": first["method_label"],
            "reproduction_scope": first["reproduction_scope"],
            "paper_equivalent": first["paper_equivalent"],
            "temporal_result_scope": first["temporal_result_scope"],
            "rate": first["rate"],
            "scope": first["scope"],
            "scope_kind": first["scope_kind"],
            "scope_methods": first["scope_methods"],
            "domain_count": len(selected),
            "eligible_domain_count": len(eligible),
            "empty_domain_count": empty_count,
            "sample_count": sum(int(row["sample_count"]) for row in selected),
            "subset_count": sum(int(row["subset_count"]) for row in selected),
            "coverage_domain_macro": float(np.mean([float(row["coverage"]) for row in selected])),
            "coverage_micro": float(
                sum(int(row["subset_count"]) for row in selected)
                / sum(int(row["sample_count"]) for row in selected)
            ),
            "status": (
                "available"
                if available
                else "unavailable_empty_domains"
                if empty_count
                else "unavailable_nonfinite_metrics"
            ),
            "nonfinite_metrics": ",".join(nonfinite_metrics),
        }
        item.update({metric: _finite_mean(row[metric] for row in metric_rows) for metric in METRICS})
        result.append(item)
    return result


def aggregate_multiseed_rates(
    rows: list[dict[str, Any]],
    *,
    requested_seed_count: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("method", "scope", "rate")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for _, selected in sorted(groups.items(), key=lambda item: item[0]):
        first = selected[0]
        complete = len(selected) == requested_seed_count and all(row["status"] == "available" for row in selected)
        item: dict[str, Any] = {
            "method": first["method"],
            "method_label": first["method_label"],
            "reproduction_scope": first["reproduction_scope"],
            "paper_equivalent": first["paper_equivalent"],
            "temporal_result_scope": first["temporal_result_scope"],
            "rate": first["rate"],
            "scope": first["scope"],
            "scope_kind": first["scope_kind"],
            "scope_methods": first["scope_methods"],
            "seed_count": len(selected),
            "requested_seed_count": requested_seed_count,
            "status": "complete" if complete else "partial",
            "coverage_domain_macro_mean": _finite_mean(row["coverage_domain_macro"] for row in selected),
            "coverage_micro_mean": _finite_mean(row["coverage_micro"] for row in selected),
            "sample_count_mean": _finite_mean(row["sample_count"] for row in selected),
            "subset_count_mean": _finite_mean(row["subset_count"] for row in selected),
            "empty_domain_count_max": max(int(row["empty_domain_count"]) for row in selected),
        }
        for metric in METRICS:
            values = np.asarray(
                [float(row[metric]) for row in selected if complete and math.isfinite(float(row[metric]))],
                dtype=np.float64,
            )
            item[f"{metric}_mean"] = float(values.mean()) if values.size else math.nan
            item[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else math.nan
        result.append(item)
    return result


def build_t2_baseline_domain_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["seed"], row["method"], row["domain_id"], row["scope"], float(row["rate"])): row
        for row in rows
    }
    result = []
    for baseline in ("amber_full", "rmbp_mm"):
        scope = f"pairwise:T2:{baseline}"
        keys = sorted(
            (seed, domain_id, rate)
            for seed, method, domain_id, row_scope, rate in lookup
            if method == "T2" and row_scope == scope
        )
        for seed, domain_id, rate in keys:
            t2 = lookup[(seed, "T2", domain_id, scope, rate)]
            other = lookup.get((seed, baseline, domain_id, scope, rate))
            t2_all = lookup.get((seed, "T2", domain_id, "all:T2", rate))
            baseline_all = lookup.get((seed, baseline, domain_id, f"all:{baseline}", rate))
            if other is None or t2_all is None or baseline_all is None:
                raise ValueError(f"Missing paired task-output row for seed={seed} domain={domain_id} baseline={baseline}")
            result.append(
                {
                    "seed": seed,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "baseline_reproduction_scope": EVIDENCE_SCOPES[baseline][0],
                    "baseline_paper_equivalent": False,
                    "baseline_temporal_result_scope": EVIDENCE_SCOPES[baseline][1],
                    "domain_id": domain_id,
                    "condition": t2["condition"],
                    "scene": t2["scene"],
                    "rate": rate,
                    "common_count": t2["subset_count"],
                    "coverage": t2["coverage"],
                    "status": t2["status"],
                    "t2_minus_baseline_all_top1": _difference(t2_all["top1"], baseline_all["top1"]),
                    "t2_minus_baseline_common_exact": _difference(t2["exact"], other["exact"]),
                    "t2_minus_baseline_common_within1": _difference(t2["within1"], other["within1"]),
                    "t2_minus_baseline_common_within3": _difference(t2["within3"], other["within3"]),
                    "t2_minus_baseline_common_mae": _difference(t2["mae"], other["mae"]),
                }
            )
    return result


def plot_t2_baseline_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    baselines = [baseline for baseline in ("amber_full", "rmbp_mm") if any(row["baseline"] == baseline for row in rows)]
    fig, axes = plt.subplots(1, max(1, len(baselines)), figsize=(11.5, 7.2), squeeze=False, constrained_layout=True)
    axes_list = axes.ravel()
    if not baselines:
        axes_list[0].text(0.5, 0.5, "No paired baseline rows", ha="center", va="center")
        axes_list[0].axis("off")
    else:
        domains = sorted({row["domain_id"] for row in rows})
        rates = sorted({float(row["rate"]) for row in rows if float(row["rate"]) > 0})
        expected_seeds = {int(row["seed"]) for row in rows}
        matrices = []
        for baseline in baselines:
            matrix = np.full((len(domains), len(rates)), np.nan, dtype=np.float64)
            for domain_index, domain_id in enumerate(domains):
                for rate_index, rate in enumerate(rates):
                    selected = [
                        row
                        for row in rows
                        if row["baseline"] == baseline
                        and row["domain_id"] == domain_id
                        and math.isclose(float(row["rate"]), rate, abs_tol=1e-6)
                    ]
                    values = [float(row["t2_minus_baseline_common_exact"]) for row in selected]
                    if (
                        {int(row["seed"]) for row in selected} == expected_seeds
                        and all(row["status"] == "available" for row in selected)
                        and all(math.isfinite(value) for value in values)
                    ):
                        matrix[domain_index, rate_index] = 100.0 * float(np.mean(values))
            matrices.append(matrix)
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        limit = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0
        cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d9d9d9")
        image = None
        for axis, baseline, matrix in zip(axes_list, baselines, matrices):
            image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            axis.set_title(f"T2 minus {METHOD_LABELS[baseline]}\npairwise common-clean exact")
            axis.set_xticks(range(len(rates)), [f"{int(rate * 100)}%" for rate in rates])
            axis.set_yticks(range(len(domains)), [_short_domain_label(domain) for domain in domains], fontsize=8)
            axis.set_xticks(np.arange(-0.5, len(rates), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.6, alpha=0.7)
            axis.tick_params(which="minor", bottom=False, left=False)
            for domain_index, rate_index in np.argwhere(np.isfinite(matrix)):
                value = matrix[domain_index, rate_index]
                axis.text(
                    rate_index,
                    domain_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.55 * limit else "black",
                )
            axis.set_xlabel("Missing modality-time cells")
            axis.set_ylabel("MMW domain")
            for boundary in (4.5, 9.5):
                axis.axhline(boundary, color="black", linewidth=1.0)
        if image is not None:
            fig.colorbar(image, ax=axes_list[: len(baselines)].tolist(), label="Exact retention delta (pp)", shrink=0.82)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_all_sample_curves(rows: list[dict[str, Any]], path: Path) -> None:
    panels = (
        ("top1", "Top1", 100.0, "%"),
        ("relative_clean_top1", "Top1 relative to clean", 100.0, "%"),
        ("true_margin_delta", "True-class margin delta", 1.0, "Delta true-class margin (logit units)"),
        ("normalized_js", "Clean-to-missing JS", 1.0, "normalized JS"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    for axis, (metric, title, scale, ylabel) in zip(axes.ravel(), panels):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and row["scope"] == f"all:{method}" and row["status"] == "complete"
                ),
                key=lambda row: float(row["rate"]),
            )
            if not selected:
                continue
            x = [100.0 * float(row["rate"]) for row in selected]
            y = [scale * float(row[f"{metric}_mean"]) for row in selected]
            std = [scale * float(row[f"{metric}_std"]) for row in selected]
            yerr = [value if math.isfinite(value) else 0.0 for value in std]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=2.5, label=METHOD_LABELS[method])
        axis.set_title(title)
        axis.set_xlabel("Missing modality-time cells (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 20, 40, 60, 80])
        axis.grid(alpha=0.22)
    axes[0, 0].legend()
    fig.suptitle("Task-output robustness under fixed modality-frame masks")
    _save_figure(fig, path)


def plot_common_clean_curves(rows: list[dict[str, Any]], path: Path) -> None:
    panels = (
        ("exact", "Exact retention", 100.0, "retained predictions (%)"),
        ("within3", "Within +/-3 beams", 100.0, "retained predictions (%)"),
        ("mae", "Circular MAE", 1.0, "mean circular error (beam steps)"),
    )
    baselines = ("amber_full", "rmbp_mm")
    fig, axes = plt.subplots(
        len(panels), len(baselines), figsize=(11.2, 9.6), sharey="row", constrained_layout=True
    )
    for column, baseline in enumerate(baselines):
        scope = f"pairwise:T2:{baseline}"
        scope_rows = [row for row in rows if row["scope"] == scope and row["status"] == "complete"]
        coverage = next((row for row in scope_rows if math.isclose(float(row["rate"]), 0.0)), None)
        coverage_note = (
            f"coverage {100.0 * float(coverage['coverage_micro_mean']):.1f}%, "
            f"mean n={float(coverage['subset_count_mean']):.0f}"
            if coverage is not None
            else "coverage unavailable"
        )
        for row_index, (metric, title, scale, ylabel) in enumerate(panels):
            axis = axes[row_index, column]
            for method in ("T2", baseline):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["method"] == method and row["scope"] == scope and row["status"] == "complete"
                    ),
                    key=lambda row: float(row["rate"]),
                )
                if not selected:
                    continue
                x = [100.0 * float(row["rate"]) for row in selected]
                y = [scale * float(row[f"{metric}_mean"]) for row in selected]
                std = [scale * float(row[f"{metric}_std"]) for row in selected]
                yerr = [value if math.isfinite(value) else 0.0 for value in std]
                axis.errorbar(x, y, yerr=yerr, marker="o", capsize=2.5, label=METHOD_LABELS[method])
            axis.set_title(f"T2 vs {METHOD_LABELS[baseline]}: {title}\n{coverage_note}")
            axis.set_xlabel("Missing modality-time cells (%)")
            axis.set_ylabel(ylabel)
            axis.set_xticks([0, 20, 40, 60, 80])
            if metric in {"exact", "within3"}:
                axis.set_ylim(0, 101)
            axis.grid(alpha=0.22)
            if row_index == 0:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(handles, labels)
    fig.suptitle("Frozen common-clean robustness under fixed modality-frame masks")
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_markdown(
    rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> str:
    lines = [
        "# MMW任务输出鲁棒性诊断",
        "",
        "> 本报告是统一四传感器协议下的本地验证证据，不等同于AMBER或RMBP原论文复现，也不自动得出T2全面最优的结论。",
        "",
        "## 协议",
        "",
        f"- 方法：{', '.join(METHOD_LABELS[method] for method in methods)}；随机种子：{', '.join(str(seed) for seed in seeds)}。",
        "- common-clean集合在每个seed和domain的clean条件一次冻结，后续所有缺失mask保持同一分母。",
        "- 本报告的逐样本曲线只使用每个非零缺失率的16个固定modality-frame masks；三mask-type主曲线由聚合评估另行报告。",
        "- 统计顺序为sample到domain-mask、mask等权、15个domain等权；micro coverage仅作补充。",
        "- true-margin只报告同一模型的missing减clean变化；JS除以ln(2)归一到[0,1]。",
        "- 相对clean保持率可能超过100%，表示该固定缺失mask下Top1高于同一模型clean Top1，不表示概率或样本保持率超过100%。",
        "",
        "## 全样本结果",
        "",
        "| 方法 | 缺失率 | Top1 | 相对clean保持率 | Margin变化 | JS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["scope_kind"] == "all":
            lines.append(
                f"| {row['method_label']} | {int(round(100 * float(row['rate'])))}% | "
                f"{_mean_std(row, 'top1')} | {_mean_std(row, 'relative_clean_top1')} | "
                f"{_mean_std(row, 'true_margin_delta')} | "
                f"{_mean_std(row, 'normalized_js')} |"
            )
    lines.extend(["", "## Common-clean保持", ""])
    for scope_kind, title in (
        ("pairwise_common_clean", "Pairwise common-clean"),
        ("three_way_common_clean", "Three-way common-clean"),
    ):
        selected = [row for row in rows if row["scope_kind"] == scope_kind]
        if not selected:
            continue
        lines.extend(
            [
                f"### {title}",
                "",
                "| 集合 | 方法 | 缺失率 | Exact | Within1 | Within3 | 圆周MAE | Margin变化 | JS | Domain-macro coverage | 空domain上限 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected.sort(key=lambda row: (row["scope"], METHODS.index(row["method"]), float(row["rate"])))
        for row in selected:
            lines.append(
                f"| {row['scope']} | {row['method_label']} | {int(round(100 * float(row['rate'])))}% | "
                f"{_mean_std(row, 'exact')} | {_mean_std(row, 'within1')} | {_mean_std(row, 'within3')} | "
                f"{_mean_std(row, 'mae')} | {_mean_std(row, 'true_margin_delta')} | "
                f"{_mean_std(row, 'normalized_js')} | {_fmt(row['coverage_domain_macro_mean'])} | "
                f"{row['empty_domain_count_max']} |"
            )
        lines.append("")
    empty_domains = sum(row["status"] != "available" for row in coverage_rows)
    drop80_deltas = []
    for baseline in ("amber_full", "rmbp_mm"):
        scope = f"pairwise:T2:{baseline}"
        selected = {
            row["method"]: row
            for row in rows
            if row["scope"] == scope
            and row["status"] == "complete"
            and math.isclose(float(row["rate"]), 0.8, abs_tol=1e-6)
        }
        if set(selected) == {"T2", baseline}:
            t2, other = selected["T2"], selected[baseline]
            drop80_deltas.append(
                f"- Drop80 T2-{METHOD_LABELS[baseline]}：Exact "
                f"{100.0 * (float(t2['exact_mean']) - float(other['exact_mean'])):+.2f} pp，"
                f"Within3 {100.0 * (float(t2['within3_mean']) - float(other['within3_mean'])):+.2f} pp，"
                f"圆周MAE {float(t2['mae_mean']) - float(other['mae_mean']):+.3f} beam steps（正值表示T2误差更大）。"
            )
    lines.extend(
        [
            "## 解读边界",
            "",
            f"- Common-clean空domain记录数：{empty_domains}；空集合不会被静默当作零或从覆盖率中删除。",
            "- Exact保持率与Within/MAE可能给出不同排序，论文中必须并列报告，不能据单项指标写成全面优于。",
            *drop80_deltas,
            f"- AMBER-Full范围：{BASELINE_SCOPES['amber_full']}。",
            f"- RMBP-MM范围：{BASELINE_SCOPES['rmbp_mm']}。",
            "- 15-domain配对差值见`t2_baseline_15domain_heatmap.png`和对应PDF；灰色单元表示共同clean集合不可用。",
            "",
        ]
    )
    return "\n".join(lines)


def _short_domain_label(domain_id: str) -> str:
    condition, _, scene = domain_id.partition("/")
    scene = scene.removeprefix("Town03_")
    scene = (
        scene.replace("_wiz_slope_seed42", "")
        .replace("_seed28", "")
        .replace("_seed40", "")
        .replace("_seed42", "")
    )
    scene = {"5wayroad": "5-way", "Tjunction": "T-junction", "gastation": "Gas station"}.get(scene, scene)
    return f"{condition.title()} | {scene.replace('_', ' ').title()}"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _identity_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return left_array.shape == right_array.shape and bool(np.array_equal(left_array, right_array))


def _first_mismatch(left: Any, right: Any) -> str:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return f" shape={left_array.shape} vs {right_array.shape}"
    unequal = left_array != right_array
    indices = np.argwhere(unequal)
    if indices.size == 0:
        return ""
    index = tuple(int(value) for value in indices[0])
    return f" at index={index}: {left_array[index]!r} vs {right_array[index]!r}"


def _scalar_text(value: Any, *, path: Path, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {field} must be scalar")
    text = str(array.item())
    if not text:
        raise ValueError(f"{path}: {field} must be non-empty")
    return text


def _finite_mean(values: Any) -> float:
    selected = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(selected.mean()) if selected.size else math.nan


def _difference(left: Any, right: Any) -> float:
    left_value = float(left)
    right_value = float(right)
    return left_value - right_value if math.isfinite(left_value) and math.isfinite(right_value) else math.nan


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"{_fmt(mean)} +/- {_fmt(std)}" if math.isfinite(std) else _fmt(mean)


def _fmt(value: Any) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
