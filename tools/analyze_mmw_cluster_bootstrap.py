"""Validation-only paired cluster bootstrap for the frozen MMW panel.

This tool deliberately consumes already-written prediction/probing evidence.  It
does not train, evaluate a model, read the outer test split, or silently inner
join mismatched evidence.  The independent unit is a trajectory (``group_id``)
or a domain; expanded missing-pattern rows are observations within a cluster.

The default paths point at the canonical masked-feature Prototype-only, matched
masked-feature hard-control, and RMBP-MM-local validation evidence.  The command
is intentionally a small standalone analysis utility rather than a public CLI.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mmw_frozen_validation_cluster_bootstrap"

SEEDS = (1, 2, 3)
PROBE_METHODS = ("Direct Prediction", "Posterior Top-3", "TBCP-3")
METRICS = ("correct", "normalized_gain")
CLUSTER_UNITS = ("trajectory", "domain")
EXPECTED_ROWS = 5_931 * 15  # keep the 5,931 x 15 contract visible
EXPECTED_SAMPLES = 5_931
EXPECTED_PATTERNS = 15
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_813


def _default_paths(kind: str, seed: int) -> tuple[Path, Path]:
    """Return the canonical ``(ledger, matrix/sample-records)`` paths."""

    if kind == "prototype_only":
        base = ROOT / "outputs/four_modal_topology_predictor_masked_feature_fusion"
        return (
            base / "probing" / f"masked_feature_fusion_prototype_only_seed{seed}" / "per_sample_results.csv.gz",
            base / "reports" / f"masked_feature_fusion_prototype_only_seed{seed}_matrix_sample_records.pt",
        )
    if kind == "hard":
        base = ROOT / "outputs/four_modal_topology_predictor_masked_feature_fusion"
        return (
            base / "probing" / f"masked_feature_fusion_off_seed{seed}" / "per_sample_results.csv.gz",
            base / "reports" / f"masked_feature_fusion_off_seed{seed}_matrix_sample_records.pt",
        )
    if kind == "rmbp":
        base = ROOT / "outputs/fair_ablation_baseline_panel/baseline_evaluations"
        return (
            base / f"rmbp_mm_seed{seed}" / "probe_diagnostic" / "per_sample_results.csv.gz",
            base / f"rmbp_mm_seed{seed}" / "baseline_sample_records.pt",
        )
    raise ValueError(f"unknown evidence kind: {kind}")


@dataclass(frozen=True)
class MatrixEvidence:
    """Stable identity and cluster metadata for one 15-mask matrix."""

    keys: tuple[tuple[str, str], ...]
    labels: np.ndarray
    group_id: np.ndarray
    domain: np.ndarray
    sample_ids: np.ndarray
    patterns: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class LedgerEvidence:
    """Selected method rows aligned to a canonical matrix key order."""

    values: Mapping[str, Mapping[str, np.ndarray]]
    path: Path


def _to_list(value: object) -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    elif hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    return list(value)  # type: ignore[arg-type]


def _load_matrix(path: Path, *, expected_rows: int, expected_samples: int, expected_patterns: int) -> MatrixEvidence:
    """Load and validate a matrix sample-records PT file.

    The function does not accept an incomplete matrix.  In particular, it checks
    that every sample has exactly the same set of missing patterns and that the
    trajectory/domain labels are constant over those patterns.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project environment supplies torch
        raise RuntimeError("读取 *.pt matrix evidence 需要项目已有的 torch 依赖。") from exc

    if not path.is_file():
        raise FileNotFoundError(f"matrix evidence 不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("labels", "sample_id", "pattern", "group_id", "domain")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} 缺少 matrix 字段: {missing}")

    labels = np.asarray(_to_list(payload["labels"]), dtype=np.int64)
    sample_ids = np.asarray([str(x) for x in _to_list(payload["sample_id"])], dtype=object)
    patterns_raw = [str(x) for x in _to_list(payload["pattern"])]
    group_id = np.asarray([str(x) for x in _to_list(payload["group_id"])], dtype=object)
    domain = np.asarray([str(x) for x in _to_list(payload["domain"])], dtype=object)
    lengths = {len(labels), len(sample_ids), len(patterns_raw), len(group_id), len(domain)}
    if lengths != {expected_rows}:
        raise ValueError(f"{path} matrix 行数不满足 {expected_rows}: {sorted(lengths)}")

    keys = tuple(zip(sample_ids.tolist(), patterns_raw))
    if len(set(keys)) != expected_rows:
        raise ValueError(f"{path} matrix 存在重复 (sample_id, missing_pattern) key")
    unique_samples = set(sample_ids.tolist())
    unique_patterns = tuple(sorted(set(patterns_raw)))
    if len(unique_samples) != expected_samples:
        raise ValueError(f"{path} sample 数量应为 {expected_samples}，实际 {len(unique_samples)}")
    if len(unique_patterns) != expected_patterns:
        raise ValueError(f"{path} missing pattern 数量应为 {expected_patterns}，实际 {len(unique_patterns)}")
    expected_per_sample = expected_patterns
    pattern_counts: dict[str, int] = {}
    sample_group: dict[str, str] = {}
    sample_domain: dict[str, str] = {}
    for sample, pattern, group, domain_value in zip(sample_ids, patterns_raw, group_id, domain):
        sample = str(sample)
        pattern_counts[sample] = pattern_counts.get(sample, 0) + 1
        previous_group = sample_group.setdefault(sample, str(group))
        previous_domain = sample_domain.setdefault(sample, str(domain_value))
        if previous_group != str(group) or previous_domain != str(domain_value):
            raise ValueError(f"{path} cluster metadata 对同一 sample 不一致: {sample}")
    bad_samples = [sample for sample, count in pattern_counts.items() if count != expected_per_sample]
    if bad_samples:
        raise ValueError(f"{path} 存在未覆盖全部 missing pattern 的 sample，例如 {bad_samples[:3]}")
    if len(set(group_id.tolist())) != 16:
        raise ValueError(f"{path} trajectory/group_id 应为 16 个簇，实际 {len(set(group_id.tolist()))}")
    if len(set(domain.tolist())) != 15:
        raise ValueError(f"{path} domain 应为 15 个簇，实际 {len(set(domain.tolist()))}")

    return MatrixEvidence(
        keys=keys,
        labels=labels,
        group_id=group_id,
        domain=domain,
        sample_ids=sample_ids,
        patterns=unique_patterns,
        path=path,
    )


def _parse_float(row: Mapping[str, str], key: str, *, path: Path, row_number: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number} 的 {key} 不是数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}:{row_number} 的 {key} 非有限")
    return value


def _open_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"ledger 不存在: {path}")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _load_ledger(path: Path, matrix: MatrixEvidence) -> LedgerEvidence:
    """Read only the three registered probe methods and reject inner drops."""

    expected = set(matrix.keys)
    selected: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {
        method: {} for method in PROBE_METHODS
    }
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "missing_pattern", "gt_beam", "method", "correct", "normalized_gain"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path} 缺少 ledger 字段: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            method = str(row.get("method", "")).strip()
            if method not in selected:
                continue
            key = (str(row["sample_id"]), str(row["missing_pattern"]))
            if key in selected[method]:
                raise ValueError(f"{path}:{row_number} 方法 {method} 重复 key: {key}")
            gt = _parse_float(row, "gt_beam", path=path, row_number=row_number)
            correct = _parse_float(row, "correct", path=path, row_number=row_number)
            gain = _parse_float(row, "normalized_gain", path=path, row_number=row_number)
            if gt != int(gt) or not 0 <= int(gt) < 64:
                raise ValueError(f"{path}:{row_number} gt_beam 非法: {gt}")
            if correct not in (0.0, 1.0):
                raise ValueError(f"{path}:{row_number} correct 必须为0/1: {correct}")
            if not 0.0 <= gain <= 1.0:
                raise ValueError(f"{path}:{row_number} normalized_gain 应在[0,1]: {gain}")
            selected[method][key] = (gt, correct, gain)

    output: dict[str, Mapping[str, np.ndarray]] = {}
    expected_labels = matrix.labels
    for method, records in selected.items():
        if len(records) != len(expected):
            raise ValueError(
                f"{path} 方法 {method} 应有 {len(expected)} 行，实际 {len(records)}；禁止静默 inner drop"
            )
        keys = set(records)
        if keys != expected:
            missing = sorted(expected.difference(keys))[:3]
            extra = sorted(keys.difference(expected))[:3]
            raise ValueError(f"{path} 方法 {method} key 集合不一致；missing={missing}, extra={extra}")
        gt = np.fromiter((records[key][0] for key in matrix.keys), dtype=np.float64, count=len(matrix.keys))
        correct = np.fromiter((records[key][1] for key in matrix.keys), dtype=np.float64, count=len(matrix.keys))
        gain = np.fromiter((records[key][2] for key in matrix.keys), dtype=np.float64, count=len(matrix.keys))
        if not np.array_equal(gt.astype(np.int64), expected_labels):
            mismatch = int(np.flatnonzero(gt.astype(np.int64) != expected_labels)[0])
            raise ValueError(f"{path} 方法 {method} GT 与 matrix 不一致，key={matrix.keys[mismatch]}")
        output[method] = {"correct": correct, "normalized_gain": gain}
    return LedgerEvidence(values=output, path=path)


def _assert_matrix_identity(reference: MatrixEvidence, candidate: MatrixEvidence, *, label: str) -> None:
    if reference.keys != candidate.keys:
        raise ValueError(f"{label} matrix sample/pattern identity 与 reference 不一致")
    for field in ("labels", "group_id", "domain"):
        if not np.array_equal(getattr(reference, field), getattr(candidate, field)):
            raise ValueError(f"{label} matrix {field} 与 reference 不一致")


def cluster_bootstrap(
    delta: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap cluster means while retaining row-weighted and macro estimates.

    A resample samples whole clusters with replacement.  ``point_estimate`` and
    its percentile interval use the original row weighting (cluster means are
    weighted by their original row counts); ``cluster_macro_point`` gives the
    equal-cluster diagnostic requested by the protocol.
    """

    values = np.asarray(delta, dtype=np.float64)
    labels = np.asarray([str(x) for x in clusters], dtype=object)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels) or len(values) == 0:
        raise ValueError("delta 与 clusters 必须为同长度的非空一维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("delta 含非有限值")
    if replicates <= 0:
        raise ValueError("bootstrap replicates 必须为正")
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    cluster_means = np.asarray([values[inverse == index].mean() for index in range(len(unique))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    drawn_means = cluster_means[draws]
    drawn_counts = counts[draws]
    # A cluster bootstrap resamples complete clusters.  Since cluster sizes can
    # differ, the row-weighted denominator is the number of rows in the sampled
    # cluster multiset, not the original total row count.
    row_bootstrap = (drawn_means * drawn_counts).sum(axis=1) / drawn_counts.sum(axis=1)
    macro_bootstrap = drawn_means.mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "cluster_macro_point": float(cluster_means.mean()),
        "ci_low": float(np.quantile(row_bootstrap, 0.025)),
        "ci_high": float(np.quantile(row_bootstrap, 0.975)),
        "cluster_macro_ci_low": float(np.quantile(macro_bootstrap, 0.025)),
        "cluster_macro_ci_high": float(np.quantile(macro_bootstrap, 0.975)),
        "rows": int(len(values)),
        "clusters": int(len(unique)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_equal_matrix_set(
    matrices: Mapping[tuple[str, int], MatrixEvidence],
    *,
    expected_rows: int,
    expected_samples: int,
    expected_patterns: int,
) -> MatrixEvidence:
    reference: MatrixEvidence | None = None
    for key in sorted(matrices):
        matrix = matrices[key]
        if len(matrix.keys) != expected_rows:
            raise ValueError(f"{key} matrix 未满足 {expected_rows} 行")
        if reference is None:
            reference = matrix
        else:
            _assert_matrix_identity(reference, matrix, label=f"{key}")
    if reference is None:
        raise ValueError("没有 matrix evidence")
    return reference


def analyze(
    *,
    paths: Mapping[str, Sequence[tuple[Path, Path]]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_rows: int = EXPECTED_ROWS,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_patterns: int = EXPECTED_PATTERNS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run strict loading, paired deltas, two cluster bootstrap units, and write reports."""

    if set(paths) != {"prototype_only", "hard", "rmbp"}:
        raise ValueError("paths 必须恰好包含 prototype_only、hard、rmbp")
    for kind in paths:
        if len(paths[kind]) != len(SEEDS):
            raise ValueError(f"{kind} 必须提供三条 seed ledger/matrix")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paired_cluster_bootstrap.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，避免覆盖：{manifest_path}（使用 --overwrite 明确覆盖）")

    matrices: dict[tuple[str, int], MatrixEvidence] = {}
    ledgers: dict[tuple[str, int], LedgerEvidence] = {}
    for kind in ("prototype_only", "hard", "rmbp"):
        for seed, (ledger_path, matrix_path) in zip(SEEDS, paths[kind]):
            matrix = _load_matrix(
                Path(matrix_path).expanduser().resolve(),
                expected_rows=expected_rows,
                expected_samples=expected_samples,
                expected_patterns=expected_patterns,
            )
            ledger = _load_ledger(Path(ledger_path).expanduser().resolve(), matrix)
            matrices[(kind, seed)] = matrix
            ledgers[(kind, seed)] = ledger
    reference = _validate_equal_matrix_set(
        matrices,
        expected_rows=expected_rows,
        expected_samples=expected_samples,
        expected_patterns=expected_patterns,
    )

    rows: list[dict[str, object]] = []
    comparisons = (("Hard", "hard"), ("RMBP", "rmbp"))
    for seed in SEEDS:
        prototype = ledgers[("prototype_only", seed)].values
        for comparator_name, comparator_kind in comparisons:
            comparator = ledgers[(comparator_kind, seed)].values
            for method in PROBE_METHODS:
                for metric in METRICS:
                    delta = prototype[method][metric] - comparator[method][metric]
                    for unit, labels in (("trajectory", reference.group_id), ("domain", reference.domain)):
                        result = cluster_bootstrap(
                            delta,
                            labels,
                            replicates=replicates,
                            seed=bootstrap_seed,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "comparison": f"Prototype-only - {comparator_name}",
                                "method": method,
                                "metric": metric,
                                "cluster_unit": unit,
                                **result,
                                "claim_ineligible": True,
                                "outer_test_accessed": False,
                                "test_sealed": True,
                            }
                        )

    summary: list[dict[str, object]] = []
    group_keys = [(row["comparison"], row["method"], row["metric"], row["cluster_unit"]) for row in rows]
    for key in dict.fromkeys(group_keys):
        selected = [row for row in rows if tuple(row[field] for field in ("comparison", "method", "metric", "cluster_unit")) == key]
        def mean_std(name: str) -> tuple[float, float]:
            values = np.asarray([float(row[name]) for row in selected], dtype=np.float64)
            return float(values.mean()), float(values.std(ddof=0))
        point_mean, point_std = mean_std("point_estimate")
        macro_mean, macro_std = mean_std("cluster_macro_point")
        low_mean, low_std = mean_std("ci_low")
        high_mean, high_std = mean_std("ci_high")
        summary.append(
            {
                "comparison": key[0],
                "method": key[1],
                "metric": key[2],
                "cluster_unit": key[3],
                "seed_count": len(selected),
                "point_estimate_mean": point_mean,
                "point_estimate_std": point_std,
                "cluster_macro_point_mean": macro_mean,
                "cluster_macro_point_std": macro_std,
                "ci_low_mean": low_mean,
                "ci_low_std": low_std,
                "ci_high_mean": high_mean,
                "ci_high_std": high_std,
                "claim_ineligible": True,
                "outer_test_accessed": False,
                "test_sealed": True,
            }
        )

    detail_csv = output_dir / "paired_cluster_bootstrap.csv"
    summary_csv = output_dir / "paired_cluster_bootstrap_summary.csv"
    _write_csv(detail_csv, rows)
    _write_csv(summary_csv, summary)
    input_records = []
    for kind in ("prototype_only", "hard", "rmbp"):
        for seed, (ledger_path, matrix_path) in zip(SEEDS, paths[kind]):
            ledger_path = Path(ledger_path).expanduser().resolve()
            matrix_path = Path(matrix_path).expanduser().resolve()
            input_records.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "ledger": str(ledger_path),
                    "matrix": str(matrix_path),
                    "ledger_sha256": _sha256(ledger_path),
                    "matrix_sha256": _sha256(matrix_path),
                    "rows": expected_rows,
                }
            )
    manifest = {
        "schema_version": 1,
        "analysis": "mmw_frozen_validation_paired_cluster_bootstrap",
        "claim_ineligible": True,
        "validation_only": True,
        "outer_test_accessed": False,
        "test_sealed": True,
        "test_evaluated": False,
        "independent_units": {"trajectory": 16, "domain": 15},
        "expected_contract": {
            "validation_samples": expected_samples,
            "missing_patterns": expected_patterns,
            "rows_per_method_seed": expected_rows,
            "methods": list(PROBE_METHODS),
            "metrics": list(METRICS),
        },
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed, "ci": "percentile_95"},
        "comparisons": ["Prototype-only - Hard", "Prototype-only - RMBP"],
        "inputs": input_records,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "result_count": len(rows),
        "summary_count": len(summary),
        "results": rows,
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "paired_cluster_bootstrap.md", manifest)
    return manifest


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"不能写空 CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, manifest: Mapping[str, object]) -> None:
    rows = manifest["results"]
    summary = manifest["summary"]
    lines = [
        "# MMW frozen validation paired cluster bootstrap",
        "",
        "- **状态**：validation-only；`claim_ineligible=true`；outer test 未访问，test 保持 sealed。",
        "- **配对**：Prototype-only 相对 matched Hard 与 RMBP-MM-local。",
        "- **独立单位**：trajectory/group_id（16 簇）与 domain（15 簇）；不把 88965 行视为独立样本。",
        f"- **Bootstrap**：{manifest['bootstrap']['replicates']} 次，seed `{manifest['bootstrap']['seed']}`，percentile 95% CI。",
        "",
        "## 每 seed：row-weighted delta 与 cluster-macro 诊断",
        "",
        "| seed | comparison | method | metric | unit | point | macro | 95% CI |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['comparison']} | {row['method']} | {row['metric']} | {row['cluster_unit']} | "
            f"{float(row['point_estimate']):+.6f} | {float(row['cluster_macro_point']):+.6f} | "
            f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## 三 seed 均值/标准差",
            "",
            "| comparison | method | metric | unit | point mean±std | macro mean±std | CI low mean±std | CI high mean±std |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['comparison']} | {row['method']} | {row['metric']} | {row['cluster_unit']} | "
            f"{float(row['point_estimate_mean']):+.6f}±{float(row['point_estimate_std']):.6f} | "
            f"{float(row['cluster_macro_point_mean']):+.6f}±{float(row['cluster_macro_point_std']):.6f} | "
            f"{float(row['ci_low_mean']):+.6f}±{float(row['ci_low_std']):.6f} | "
            f"{float(row['ci_high_mean']):+.6f}±{float(row['ci_high_std']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed_paths(values: Sequence[str], *, name: str) -> list[tuple[Path, Path]]:
    """Parse three ``ledger,matrix`` pairs for a compact CLI override."""

    if len(values) != len(SEEDS):
        raise ValueError(f"--{name} 需要三个 seed 的 ledger,matrix 对")
    result = []
    for value in values:
        parts = value.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"--{name} 每项必须是 ledger,matrix：{value}")
        result.append((Path(parts[0]), Path(parts[1])))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", nargs=3, metavar="LEDGER,MATRIX", help="Prototype-only seed1/2/3 ledger,matrix")
    parser.add_argument("--hard", nargs=3, metavar="LEDGER,MATRIX", help="matched Hard seed1/2/3 ledger,matrix")
    parser.add_argument("--rmbp", nargs=3, metavar="LEDGER,MATRIX", help="RMBP seed1/2/3 ledger,matrix")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument("--expected-patterns", type=int, default=EXPECTED_PATTERNS)
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖同名 report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths: dict[str, Sequence[tuple[Path, Path]]] = {}
    for kind, values in (("prototype_only", args.prototype), ("hard", args.hard), ("rmbp", args.rmbp)):
        if values is None:
            paths[kind] = [_default_paths(kind, seed) for seed in SEEDS]
        else:
            paths[kind] = _parse_seed_paths(values, name={"prototype_only": "prototype", "hard": "hard", "rmbp": "rmbp"}[kind])
    try:
        manifest = analyze(
            paths=paths,
            output_dir=args.output_dir,
            replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            expected_rows=args.expected_rows,
            expected_samples=args.expected_samples,
            expected_patterns=args.expected_patterns,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "result_count": manifest["result_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
