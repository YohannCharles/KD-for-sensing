#!/usr/bin/env python3
"""Build the ADBA-first summary for the inner-only tie-aware Router screen."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml


PROTOCOL_ID = "mmw_tie_aware_router_screen_v1"
CANDIDATES = (
    "HardFirstControl",
    "HardConfidenceTie",
    "SoftUniformTie",
    "SoftConfidenceTie",
    "DistanceSoftT05",
    "DistanceSoftT10",
    "DistanceConfidenceT10",
    "UniformFusion",
)
PRIMARY_METRIC = "adba"
SECONDARY_METRIC = "top1"
MAIN_CELLS = ("Clean", "Drop1", "Drop2", "Drop3", "Block80")
CURVE_CELLS = ("Clean", "Drop1", "Drop2", "Drop3", "Token20", "Token40", "Token60", "Token80")
EXPECTED_DOMAINS = 15
EXPECTED_ROWS = 1860
ADBA_DELTA = 5.0
DISTANCE_MODE = "circular"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", default="outputs/mmw_tie_aware_router_screen_v1")
    parser.add_argument("--output-dir", default="outputs/mmw_tie_aware_router_screen_v1/adba_first_summary")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()
    summarize(
        Path(args.screen_dir),
        Path(args.output_dir),
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    return 0


def summarize(
    screen_dir: Path,
    output_dir: Path,
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260718,
) -> dict[str, Any]:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    manifest_path = screen_dir / "training_manifest_tie_aware_seed1.json"
    manifest = _read_json(manifest_path)
    jobs = _validate_manifest(manifest, screen_dir)
    rows_by_candidate = {
        candidate: _read_csv(screen_dir / "eval_inner" / candidate / "metrics.csv")
        for candidate in CANDIDATES
    }
    shared = _validate_evidence(rows_by_candidate)
    domain_cells = _domain_cells(rows_by_candidate)
    table = _primary_table(domain_cells)
    paired = _paired_deltas(
        domain_cells,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    provenance = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "screening_role": "frozen_inner_validation_only",
        "claim_eligible": False,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "adba_definition": "progressive_top3_minimum_circular_beam_distance",
        "adba_delta": ADBA_DELTA,
        "dba_distance_mode": DISTANCE_MODE,
        "main_cells": list(MAIN_CELLS),
        "curve_cells": list(CURVE_CELLS),
        "candidate_count": len(CANDIDATES),
        "domain_count": EXPECTED_DOMAINS,
        "rows_per_candidate": EXPECTED_ROWS,
        "metric_profile": shared["metric_profile"],
        "mask_cache_checksums": shared["mask_cache_checksums"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_metrics": {
            candidate: {
                "path": str((screen_dir / "eval_inner" / candidate / "metrics.csv").resolve()),
                "sha256": _sha256_file(screen_dir / "eval_inner" / candidate / "metrics.csv"),
                "checkpoint_sha256": jobs[candidate]["checkpoint_sha256"],
            }
            for candidate in CANDIDATES
        },
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "selection_warning": "Development-only metric reprioritization; existing outer evidence is not reinterpreted.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "domain_cells.csv", domain_cells)
    _write_csv(output_dir / "adba_primary_table.csv", table)
    _write_csv(output_dir / "paired_adba_deltas.csv", paired)
    _write_json(output_dir / "provenance.json", provenance)
    (output_dir / "summary.md").write_text(_markdown(table, paired), encoding="utf-8")
    return {"table": table, "paired": paired, "domain_cells": domain_cells, "provenance": provenance}


def _validate_manifest(manifest: dict[str, Any], screen_dir: Path) -> dict[str, dict[str, str]]:
    request = manifest.get("request", {})
    if manifest.get("protocol") != PROTOCOL_ID or request.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Tie-aware screen protocol identity mismatch")
    if request.get("claim_eligible") is not False or request.get("selection_split") != "frozen_inner_validation_only":
        raise ValueError("ADBA-first summary accepts only claim-ineligible frozen inner validation")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or {job.get("method") for job in jobs} != set(CANDIDATES):
        raise ValueError("Tie-aware screen manifest must contain exactly eight candidates")
    result: dict[str, dict[str, str]] = {}
    for job in jobs:
        candidate = str(job["method"])
        if job.get("status") != "done" or job.get("evaluation_status") != "done":
            raise ValueError(f"Incomplete tie-aware screen job: {candidate}")
        if job.get("training_return_code") != 0 or job.get("evaluation_return_code") != 0:
            raise ValueError(f"Failed tie-aware screen job: {candidate}")
        config_path = Path(str(job["config_path"]))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        evaluation = config.get("evaluation", {})
        if float(evaluation.get("dba_delta", math.nan)) != ADBA_DELTA:
            raise ValueError(f"ADBA delta mismatch for {candidate}")
        if evaluation.get("dba_distance_mode") != DISTANCE_MODE:
            raise ValueError(f"ADBA distance mode mismatch for {candidate}")
        checkpoint = screen_dir / candidate / "seed1" / "checkpoints" / "last.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing last checkpoint for {candidate}: {checkpoint}")
        result[candidate] = {"checkpoint_sha256": _sha256_file(checkpoint)}
    return result


def _validate_evidence(rows_by_candidate: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    reference_identities: set[tuple[str, ...]] | None = None
    profiles: set[str] = set()
    cache_checksums: set[str] = set()
    for candidate in CANDIDATES:
        rows = rows_by_candidate.get(candidate, [])
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"{candidate} requires exactly {EXPECTED_ROWS} evaluation rows, found {len(rows)}")
        domains = {row.get("domain_id", "") for row in rows}
        if len(domains) != EXPECTED_DOMAINS:
            raise ValueError(f"{candidate} requires exactly {EXPECTED_DOMAINS} domains")
        identities: set[tuple[str, ...]] = set()
        for row in rows:
            if row.get("method") != candidate or row.get("seed") != "1":
                raise ValueError(f"Method/seed mismatch in {candidate}")
            if row.get("coverage_status") != "complete" or _truthy(row.get("partial_request")):
                raise ValueError(f"Partial evidence is not admissible for {candidate}")
            if row.get("screening_role") != "local_validation" or "inner_validation.csv" not in row.get("sample_csv", ""):
                raise ValueError(f"Outer or non-inner evidence is not admissible for {candidate}")
            if row.get("checkpoint_role") != "last" or Path(row.get("checkpoint", "")).name != "last.pth":
                raise ValueError(f"Fixed last checkpoint is required for {candidate}")
            if row.get("dba_distance_mode") != DISTANCE_MODE:
                raise ValueError(f"Non-circular ADBA row found for {candidate}")
            if int(float(row.get("sample_count", 0))) != int(float(row.get("expected_sample_count", -1))):
                raise ValueError(f"Sample coverage mismatch for {candidate}/{row.get('domain_id')}")
            for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
                if not math.isfinite(float(row.get(metric, "nan"))):
                    raise ValueError(f"Non-finite {metric} for {candidate}")
            identity = tuple(
                row.get(field, "")
                for field in (
                    "domain_id", "sample_csv_sha256", "eval_family", "pattern", "available_modalities",
                    "missing_rate", "drop_count", "mask_index", "mask_type", "mask_digest",
                    "mask_cache_checksum", "mask_cache_seed", "sample_count", "expected_sample_count",
                )
            )
            if identity in identities:
                raise ValueError(f"Duplicate evaluation identity for {candidate}: {identity}")
            identities.add(identity)
            profiles.add(row.get("metric_profile", ""))
            cache_checksums.add(row.get("mask_cache_checksum", ""))
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError(f"Fixed sample/mask identities differ for {candidate}")
    if len(profiles) != 1 or not next(iter(profiles), ""):
        raise ValueError(f"Metric profile mismatch: {sorted(profiles)}")
    return {"metric_profile": next(iter(profiles)), "mask_cache_checksums": sorted(cache_checksums)}


def _domain_cells(rows_by_candidate: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows_by_candidate[candidate]:
            by_domain[row["domain_id"]].append(row)
        for domain_id, rows in sorted(by_domain.items()):
            for cell, selector in _cell_selectors().items():
                selected = [row for row in rows if selector(row)]
                if not selected:
                    raise ValueError(f"Missing {cell} rows for {candidate}/{domain_id}")
                result.append(
                    {
                        "candidate": candidate,
                        "domain_id": domain_id,
                        "cell": cell,
                        "mask_count": len(selected),
                        PRIMARY_METRIC: statistics.fmean(float(row[PRIMARY_METRIC]) for row in selected),
                        SECONDARY_METRIC: statistics.fmean(float(row[SECONDARY_METRIC]) for row in selected),
                    }
                )
    return result


def _cell_selectors() -> dict[str, Callable[[dict[str, str]], bool]]:
    selectors: dict[str, Callable[[dict[str, str]], bool]] = {
        "Clean": lambda row: row["eval_family"] == "whole_modality" and row["drop_count"] == "0",
        "Drop1": lambda row: row["eval_family"] == "whole_modality" and row["drop_count"] == "1",
        "Drop2": lambda row: row["eval_family"] == "whole_modality" and row["drop_count"] == "2",
        "Drop3": lambda row: row["eval_family"] == "whole_modality" and row["drop_count"] == "3",
        "Block80": lambda row: row["mask_type"] == "block" and float(row["missing_rate"]) == 0.8,
    }
    selectors.update(
        {
            f"Token{rate}": lambda row, value=rate / 100: (
                row["mask_type"] == "modality_frame" and float(row["missing_rate"]) == value
            )
            for rate in (20, 40, 60, 80)
        }
    )
    return selectors


def _primary_table(domain_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["candidate"], row["domain_id"], row["cell"]): row for row in domain_cells}
    domains = sorted({row["domain_id"] for row in domain_cells})
    result = []
    for candidate in CANDIDATES:
        out: dict[str, Any] = {"candidate": candidate}
        for cell in (*MAIN_CELLS, "Token20", "Token40", "Token60", "Token80"):
            for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
                out[f"{metric}_{cell}"] = statistics.fmean(index[candidate, domain, cell][metric] for domain in domains)
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            out[f"{metric}_Main5"] = statistics.fmean(
                statistics.fmean(index[candidate, domain, cell][metric] for cell in MAIN_CELLS) for domain in domains
            )
            out[f"{metric}_Curve8"] = statistics.fmean(
                statistics.fmean(index[candidate, domain, cell][metric] for cell in CURVE_CELLS) for domain in domains
            )
        result.append(out)
    result.sort(key=lambda row: (-row["adba_Main5"], -row["top1_Main5"], row["candidate"]))
    for rank, row in enumerate(result, 1):
        row["adba_primary_rank"] = rank
    return result


def _paired_deltas(
    domain_cells: list[dict[str, Any]], *, bootstrap_iterations: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    index = {(row["candidate"], row["domain_id"], row["cell"]): row for row in domain_cells}
    domains = sorted({row["domain_id"] for row in domain_cells})
    result = []
    for baseline in ("HardFirstControl", "UniformFusion"):
        for candidate in CANDIDATES:
            if candidate == baseline:
                continue
            for composite, cells in (("Main5", MAIN_CELLS), ("Curve8", CURVE_CELLS)):
                deltas = [
                    statistics.fmean(index[candidate, domain, cell][PRIMARY_METRIC] for cell in cells)
                    - statistics.fmean(index[baseline, domain, cell][PRIMARY_METRIC] for cell in cells)
                    for domain in domains
                ]
                low, high = _bootstrap_interval(
                    deltas,
                    iterations=bootstrap_iterations,
                    seed=(
                        bootstrap_seed
                        + 100 * CANDIDATES.index(baseline)
                        + 10 * CANDIDATES.index(candidate)
                        + int(composite == "Curve8")
                    ),
                )
                result.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": f"adba_{composite}",
                        "mean_delta": statistics.fmean(deltas),
                        "ci95_low": low,
                        "ci95_high": high,
                        "domain_wins": sum(value > 0 for value in deltas),
                        "domain_losses": sum(value < 0 for value in deltas),
                        "domain_count": len(deltas),
                    }
                )
    return result


def _bootstrap_interval(values: list[float], *, iterations: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]


def _markdown(table: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    lines = [
        "# Tie-Aware Router ADBA-First Inner-Validation Summary",
        "",
        "主指标：circular progressive Top-3 ADBA（delta=5）；Top-1 为支线。所有结果仅用于 inner-development，`claim_eligible=false`。",
        "",
        "## ADBA 主表",
        "",
        "| Rank | Candidate | Clean | Drop1 | Drop2 | Drop3 | Block80 | Main5 | Curve8 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['adba_primary_rank']} | {row['candidate']} | "
            + " | ".join(_pct(row[f"adba_{cell}"]) for cell in MAIN_CELLS)
            + f" | {_pct(row['adba_Main5'])} | {_pct(row['adba_Curve8'])} |"
        )
    lines.extend(
        [
            "",
            "## Top-1 支线",
            "",
            "| Candidate | Clean | Drop1 | Drop2 | Drop3 | Block80 | Main5 | Curve8 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table:
        lines.append(
            f"| {row['candidate']} | "
            + " | ".join(_pct(row[f"top1_{cell}"]) for cell in MAIN_CELLS)
            + f" | {_pct(row['top1_Main5'])} | {_pct(row['top1_Curve8'])} |"
        )
    best = table[0]
    control = next(
        row
        for row in paired
        if row["candidate"] == best["candidate"]
        and row["baseline"] == "HardFirstControl"
        and row["metric"] == "adba_Main5"
    )
    uniform = next(
        (
            row
            for row in paired
            if row["candidate"] == best["candidate"]
            and row["baseline"] == "UniformFusion"
            and row["metric"] == "adba_Main5"
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## 配对结论",
            "",
            f"ADBA 主排序第一为 `{best['candidate']}`，Main5={_pct(best['adba_Main5'])}。",
            f"相对 HardFirstControl 的逐域配对差为 {_pp(control['mean_delta'])}，95% bootstrap CI [{_pp(control['ci95_low'])}, {_pp(control['ci95_high'])}]。",
        ]
    )
    if uniform is not None:
        lines.append(
            f"相对 UniformFusion 的逐域配对差为 {_pp(uniform['mean_delta'])}，95% bootstrap CI [{_pp(uniform['ci95_low'])}, {_pp(uniform['ci95_high'])}]。"
        )
    lines.extend(
        [
            "",
            "该排序复用既有 fixed-mask 预测，没有重新推理。既有 outer evidence 不得追溯改写为 ADBA 主张；候选冻结后需运行新的 confirmation protocol。",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


if __name__ == "__main__":
    raise SystemExit(main())
