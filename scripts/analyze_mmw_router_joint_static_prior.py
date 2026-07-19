#!/usr/bin/env python3
"""Falsify dynamic Router adaptation with frozen Clean-prior controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from kd_sensing.data.mmw.twc_router_joint_stress import JOINT_RATES, MASKS_PER_RATE, PROTOCOL_ID

from eval_mmw_router_oracle_gap import metrics_for, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/mmw_router_joint_stress_v1"
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"
DEFAULT_CONFIG = ROOT / "outputs/mmw_router_expected_utility_screen_v3/generated_configs/CurrentControl_seed1.yaml"
OUTPUT_NAME = "static_prior_falsification"
BRANCHES = ("uniform", "global_clean_prior", "per_sample_clean", "dynamic")
METRICS = (
    "adba",
    "top1",
    "normalized_gain",
    "spectral_efficiency_ratio_0db",
    "spectral_efficiency_ratio_10db",
    "spectral_efficiency_ratio_20db",
)
GATE_RATES = (0.4, 0.6, 0.8)
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    analyze(
        Path(args.root).resolve(),
        Path(args.cache).resolve(),
        Path(args.config).resolve(),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    return 0


def analyze(
    root: Path,
    cache_path: Path,
    config_path: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    manifest = _read_json(root / "evaluation_manifest.json")
    cache = _read_json(cache_path)
    _validate_parent(manifest, cache, cache_path, config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    conditions = list(cache["conditions"])
    condition_by_pattern = {str(item["pattern"]): item for item in conditions}
    traces, inventory = _load_traces(root, conditions, manifest)
    clean = traces["clean"]
    global_prior = global_clean_prior(clean)

    metric_rows: list[dict[str, Any]] = []
    alpha_rows_raw: list[dict[str, Any]] = []
    for pattern, domain_traces in traces.items():
        condition = condition_by_pattern[pattern]
        rate = float(condition["requested_stress_rate"])
        for domain_id, trace in domain_traces.items():
            clean_trace = clean[domain_id]
            _validate_pair(trace, clean_trace, pattern=pattern, domain_id=domain_id)
            availability = trace["available_modalities"].astype(bool)
            global_weights = normalize_available_weights(
                np.broadcast_to(global_prior, trace["router_weights"].shape), availability
            )
            sample_weights = normalize_available_weights(clean_trace["router_weights"], availability)
            branches = {
                "uniform": trace["uniform_logits"],
                "global_clean_prior": fuse_logits(global_weights, trace["unimodal_logits"]),
                "per_sample_clean": fuse_logits(sample_weights, trace["unimodal_logits"]),
                "dynamic": trace["learned_logits"],
            }
            for branch, logits in branches.items():
                values = metrics_for(logits, trace["target"], trace["beam_powers"], cfg)
                metric_rows.append(
                    {
                        "pattern": pattern,
                        "requested_stress_rate": rate,
                        "mask_set_index": int(condition["mask_set_index"]),
                        "domain_id": domain_id,
                        "branch": branch,
                        "sample_count": int(trace["target"].shape[0]),
                        **{metric: float(values[metric]) for metric in METRICS},
                    }
                )
            if rate in GATE_RATES:
                for alpha in ALPHAS:
                    weights = normalize_available_weights(
                        (1.0 - alpha) * global_weights + alpha * trace["router_weights"], availability
                    )
                    values = metrics_for(
                        fuse_logits(weights, trace["unimodal_logits"]),
                        trace["target"],
                        trace["beam_powers"],
                        cfg,
                    )
                    alpha_rows_raw.append(
                        {
                            "alpha": alpha,
                            "domain_id": domain_id,
                            "pattern": pattern,
                            **{metric: float(values[metric]) for metric in METRICS},
                        }
                    )

    domain_rate_rows = _domain_rate_rows(metric_rows)
    rate_rows = _rate_rows(domain_rate_rows)
    bootstrap_rows = _bootstrap_rows(
        domain_rate_rows,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    alpha_rows = _alpha_rows(alpha_rows_raw)
    decision = claim_decision(bootstrap_rows)
    provenance = {
        "protocol": "mmw_router_joint_static_prior_v1",
        "parent_protocol": PROTOCOL_ID,
        "parent_request_sha256": str(manifest["request_sha256"]),
        "parent_manifest_sha256": sha256(root / "evaluation_manifest.json"),
        "cache_sha256": sha256(cache_path),
        "cache_checksum": str(cache["checksum"]),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": str(manifest["request"]["checkpoint_sha256"]),
        "trace_count": len(inventory),
        "trace_inventory_sha256": _payload_sha256(inventory),
        "global_clean_prior": {
            name: float(global_prior[index])
            for index, name in enumerate(("image", "radar", "gps", "lidar"))
        },
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "claim_eligible": False,
        "post_hoc": True,
        "no_training_or_forward": True,
    }
    payload = {
        "provenance": provenance,
        "decision": decision,
        "rate_summary": rate_rows,
        "paired_domain_bootstrap": bootstrap_rows,
        "alpha_exploratory": alpha_rows,
    }
    output = root / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "rate_summary.csv", rate_rows)
    _write_csv(output / "paired_domain_bootstrap.csv", bootstrap_rows)
    _write_csv(output / "alpha_exploratory.csv", alpha_rows)
    write_json(output / "summary.json", payload)
    write_json(output / "provenance.json", provenance)
    (output / "README.md").write_text(
        _markdown(provenance, rate_rows, bootstrap_rows, alpha_rows, decision), encoding="utf-8"
    )
    return payload


def global_clean_prior(clean: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    weights = np.concatenate([trace["router_weights"] for trace in clean.values()], axis=0).mean(axis=0)
    if weights.shape != (4,) or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("Clean Router weights cannot define a four-modality prior.")
    return weights / weights.sum()


def normalize_available_weights(weights: np.ndarray, available: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64) * np.asarray(available, dtype=bool)
    denominator = values.sum(axis=1, keepdims=True)
    if values.ndim != 2 or values.shape != available.shape or bool((denominator <= 0.0).any()):
        raise ValueError("Fusion weights and availability must be [N,4] with positive available mass.")
    return values / denominator


def fuse_logits(weights: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3 or weights.shape != logits.shape[:2]:
        raise ValueError("Fusion requires weights [N,M] and logits [N,M,C].")
    return (weights[..., None] * logits).sum(axis=1).astype(np.float32)


def paired_bootstrap(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    deltas = np.asarray(values, dtype=np.float64)
    if deltas.shape != (15,) or not np.isfinite(deltas).all():
        raise ValueError("Static-prior paired bootstrap requires 15 finite domain deltas.")
    rng = np.random.default_rng(int(seed))
    draws = deltas[rng.integers(0, 15, size=(int(iterations), 15))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def claim_decision(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    combined = {
        (str(row["control"]), str(row["metric"])): row
        for row in rows
        if row["scope"] == "Joint40_60_80Combined"
    }
    required = [combined[("global_clean_prior", metric)] for metric in ("adba", "normalized_gain")]
    supported = all(float(row["ci_low"]) > 0.0 for row in required)
    return {
        "dynamic_adaptation_supported": supported,
        "pre_registered_uniform_gate_unchanged": True,
        "claim": "corruption_adaptive_reliability" if supported else "learned_non_uniform_fusion_only",
        "reason": (
            "Dynamic beats GlobalCleanPrior with positive paired-domain lower bounds."
            if supported
            else "Dynamic does not robustly beat the deployable GlobalCleanPrior control."
        ),
    }


def _validate_parent(
    manifest: Mapping[str, Any], cache: Mapping[str, Any], cache_path: Path, config_path: Path
) -> None:
    request = manifest.get("request", {})
    if manifest.get("status") != "complete" or request.get("protocol") != PROTOCOL_ID:
        raise ValueError("Parent joint-stress evaluation is not complete.")
    if manifest.get("summary", {}).get("status") != "complete":
        raise ValueError("Parent joint-stress summary is not complete.")
    if request.get("cache_checksum") != cache.get("checksum") or request.get("cache_sha256") != sha256(cache_path):
        raise ValueError("Parent cache identity mismatch.")
    if request.get("config_sha256") != sha256(config_path):
        raise ValueError("Parent config identity mismatch.")
    if int(request.get("condition_count", -1)) != 81 or len(cache.get("conditions", ())) != 81:
        raise ValueError("Static-prior analysis requires all 81 conditions.")
    if len(manifest.get("jobs", ())) != 8 or any(job.get("status") != "complete" for job in manifest["jobs"]):
        raise ValueError("Static-prior analysis requires all eight completed shards.")


def _load_traces(
    root: Path,
    conditions: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, str]]]:
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    inventory: list[dict[str, str]] = []
    request = manifest["request"]
    required = (
        "sample_id",
        "target",
        "beam_powers",
        "unimodal_logits",
        "router_weights",
        "uniform_logits",
        "learned_logits",
        "available_modalities",
    )
    for condition in conditions:
        pattern = str(condition["pattern"])
        complete = _read_json(root / pattern / "complete.json")
        if (
            complete.get("status") != "complete"
            or complete.get("request_sha256") != manifest["request_sha256"]
            or complete.get("checkpoint_sha256") != request["checkpoint_sha256"]
            or complete.get("cache_checksum") != request["cache_checksum"]
        ):
            raise ValueError(f"Condition completion identity mismatch: {pattern}")
        files = list(complete.get("trace_files", ()))
        if len(files) != 15:
            raise ValueError(f"Condition {pattern} must contain 15 traces.")
        domains: dict[str, dict[str, np.ndarray]] = {}
        for item in files:
            path = Path(item["path"])
            digest = sha256(path)
            if digest != item["sha256"]:
                raise ValueError(f"Trace checksum mismatch: {path}")
            with np.load(path) as payload:
                if any(key not in payload for key in required):
                    raise ValueError(f"Trace schema mismatch: {path}")
                domain_id = str(payload["domain_id"].item())
                if domain_id in domains:
                    raise ValueError(f"Duplicate trace domain: {pattern}/{domain_id}")
                domains[domain_id] = {key: np.asarray(payload[key]) for key in required}
            inventory.append({"path": str(path), "sha256": digest})
        result[pattern] = domains
    inventory.sort(key=lambda item: item["path"])
    if len(inventory) != 1215:
        raise ValueError("Static-prior analysis requires exactly 1,215 traces.")
    return result, inventory


def _validate_pair(
    trace: Mapping[str, np.ndarray], clean: Mapping[str, np.ndarray], *, pattern: str, domain_id: str
) -> None:
    if not np.array_equal(trace["sample_id"].astype(str), clean["sample_id"].astype(str)):
        raise ValueError(f"Sample identity mismatch: {pattern}/{domain_id}")
    if not np.array_equal(trace["target"], clean["target"]) or not np.array_equal(
        trace["beam_powers"], clean["beam_powers"]
    ):
        raise ValueError(f"Target or beam-power identity mismatch: {pattern}/{domain_id}")


def _domain_rate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["requested_stress_rate"]), str(row["domain_id"]), str(row["branch"]))].append(row)
    domains = sorted({str(row["domain_id"]) for row in rows})
    result = []
    for rate in (0.0, *JOINT_RATES):
        expected = 1 if rate == 0.0 else MASKS_PER_RATE
        for domain_id in domains:
            for branch in BRANCHES:
                selected = grouped[(float(rate), domain_id, branch)]
                if len(selected) != expected:
                    raise ValueError(f"Incomplete domain/rate branch: {rate}/{domain_id}/{branch}")
                result.append(
                    {
                        "requested_stress_rate": float(rate),
                        "domain_id": domain_id,
                        "branch": branch,
                        "mask_count": expected,
                        **{metric: _mean(row[metric] for row in selected) for metric in METRICS},
                    }
                )
    return result


def _rate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["requested_stress_rate"]), str(row["branch"]))].append(row)
    result = []
    for rate in (0.0, *JOINT_RATES):
        for branch in BRANCHES:
            selected = grouped[(float(rate), branch)]
            if len(selected) != 15:
                raise ValueError("Rate summary requires 15 domain aggregates.")
            result.append(
                {
                    "cell": "Clean" if rate == 0.0 else f"Joint{int(rate * 100)}",
                    "requested_stress_rate": float(rate),
                    "branch": branch,
                    "domain_count": 15,
                    **{metric: _mean(row[metric] for row in selected) for metric in METRICS},
                }
            )
    return result


def _bootstrap_rows(
    rows: Iterable[Mapping[str, Any]], *, iterations: int, seed: int
) -> list[dict[str, Any]]:
    indexed = {
        (float(row["requested_stress_rate"]), str(row["domain_id"]), str(row["branch"])): row
        for row in rows
    }
    domains = sorted({str(row["domain_id"]) for row in rows})
    result = []
    for control in ("global_clean_prior", "per_sample_clean", "uniform"):
        for metric in ("adba", "normalized_gain"):
            for rate in GATE_RATES:
                deltas = np.asarray(
                    [indexed[(rate, domain, "dynamic")][metric] - indexed[(rate, domain, control)][metric] for domain in domains]
                )
                low, high = paired_bootstrap(
                    deltas, iterations=iterations, seed=_stable_seed(seed, control, metric, str(rate))
                )
                result.append(_bootstrap_row(rate, control, metric, deltas, low, high, iterations))
            combined = np.asarray(
                [
                    _mean(
                        indexed[(rate, domain, "dynamic")][metric] - indexed[(rate, domain, control)][metric]
                        for rate in GATE_RATES
                    )
                    for domain in domains
                ]
            )
            low, high = paired_bootstrap(
                combined, iterations=iterations, seed=_stable_seed(seed, control, metric, "combined")
            )
            result.append(_bootstrap_row(None, control, metric, combined, low, high, iterations))
    return result


def _bootstrap_row(
    rate: float | None,
    control: str,
    metric: str,
    deltas: np.ndarray,
    low: float,
    high: float,
    iterations: int,
) -> dict[str, Any]:
    return {
        "scope": "Joint40_60_80Combined" if rate is None else f"Joint{int(rate * 100)}",
        "contrast": f"dynamic_minus_{control}",
        "control": control,
        "metric": metric,
        "mean_delta": float(deltas.mean()),
        "ci_low": low,
        "ci_high": high,
        "domain_wins": int((deltas > 0.0).sum()),
        "paired_domain_count": 15,
        "bootstrap_iterations": iterations,
    }


def _alpha_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["alpha"])].append(row)
    result = []
    for alpha in ALPHAS:
        selected = grouped[alpha]
        if len(selected) != 15 * 20 * len(GATE_RATES):
            raise ValueError("Exploratory alpha curve is incomplete.")
        result.append({"alpha": alpha, **{metric: _mean(row[metric] for row in selected) for metric in METRICS}})
    baseline = next(row for row in result if row["alpha"] == 0.0)
    for row in result:
        for metric in METRICS:
            row[f"delta_vs_global_prior_{metric}"] = row[metric] - baseline[metric]
    return result


def _markdown(
    provenance: Mapping[str, Any],
    rates: Iterable[Mapping[str, Any]],
    bootstraps: Iterable[Mapping[str, Any]],
    alphas: Iterable[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    indexed = {(row["cell"], row["branch"]): row for row in rates}
    lines = [
        "# Joint Router Static-Prior Falsification",
        "",
        "该 post-hoc control 不训练、不重新前向，也不修改预注册 Uniform Gate。GlobalCleanPrior 是 Clean trace 上全部样本 Router 权重的全局均值；PerSampleClean 只作为同样本反事实上界。",
        "",
        "| Cell | Uniform gain | Global prior | Per-sample Clean | Dynamic | Dynamic-Global | Uniform ADBA | Global ADBA | Dynamic ADBA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ("Clean", "Joint20", "Joint40", "Joint60", "Joint80"):
        u, g, p, d = (indexed[(cell, branch)] for branch in BRANCHES)
        lines.append(
            f"| {cell} | {u['normalized_gain']:.4f} | {g['normalized_gain']:.4f} | "
            f"{p['normalized_gain']:.4f} | {d['normalized_gain']:.4f} | "
            f"{d['normalized_gain'] - g['normalized_gain']:+.4f} | {u['adba']:.4f} | "
            f"{g['adba']:.4f} | {d['adba']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired-domain combined contrasts",
            "",
            "| Contrast | Metric | Mean | 95% CI | Domain wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstraps:
        if row["scope"] == "Joint40_60_80Combined":
            lines.append(
                f"| {row['contrast']} | {row['metric']} | {row['mean_delta']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['domain_wins']}/15 |"
            )
    best = max(alphas, key=lambda row: row["normalized_gain"])
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"- Dynamic adaptation supported: `{decision['dynamic_adaptation_supported']}`",
            f"- Allowed claim: `{decision['claim']}`",
            f"- Reason: {decision['reason']}",
            f"- Exploratory best alpha by normalized gain: `{best['alpha']}`; delta vs static prior `{best['delta_vs_global_prior_normalized_gain']:+.6f}`。该曲线为 post-hoc，不用于选择模型。",
            "",
            "## Provenance",
            "",
            f"- Parent request: `{provenance['parent_request_sha256']}`",
            f"- Trace inventory: `{provenance['trace_inventory_sha256']}` ({provenance['trace_count']} files)",
            f"- Global Clean prior: `{json.dumps(provenance['global_clean_prior'], sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    return int.from_bytes(hashlib.sha256("::".join((str(base), *parts)).encode()).digest()[:8], "big")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot average an empty collection.")
    return float(sum(items) / len(items))


if __name__ == "__main__":
    raise SystemExit(main())
