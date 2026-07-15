import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def analysis():
    path = ROOT / "scripts" / "summarize_mmw_task_output_robustness.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("sample_ids", lambda value: value[::-1].copy()),
        ("labels", lambda value: np.roll(value, 1)),
        ("sample_csv_sha256", lambda _value: np.asarray("different-csv")),
        ("mask_digests", lambda value: np.asarray(["different-mask", *value[1:]])),
        ("cache_checksums", lambda value: np.asarray(["different-cache", *value[1:]])),
    ],
)
def test_strict_identity_rejects_same_length_misalignment(tmp_path, analysis, field, mutate):
    labels = np.asarray([0, 63, 10, 20], dtype=np.int16)
    for method in analysis.METHODS:
        _write_domain(tmp_path, seed=1, method=method, domain_id="sunny/scene", labels=labels)

    bundles = analysis.load_bundles(tmp_path, methods=analysis.METHODS, seeds=(1,), expected_domains=1)
    assert bundles[1]["T2"]["sunny/scene"]["checkpoint_sha256"] != bundles[1]["amber_full"]["sunny/scene"]["checkpoint_sha256"]

    path = next((tmp_path / "seed1" / "amber_full" / "domains").glob("*.npz"))
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays[field] = mutate(arrays[field])
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match=rf"identity mismatch field={field}"):
        analysis.load_bundles(tmp_path, methods=analysis.METHODS, seeds=(1,), expected_domains=1)


def test_common_clean_is_frozen_and_outputs_are_cautious(tmp_path, analysis):
    labels = np.asarray([0, 63, 10, 20], dtype=np.int16)
    predictions = {
        "T2": _prediction_matrix(labels, clean=[0, 63, 10, 21], missing=[0, 10, 10, 20]),
        "amber_full": _prediction_matrix(labels, clean=[0, 63, 11, 20], missing=[1, 0, 10, 20]),
        "rmbp_mm": _prediction_matrix(labels, clean=[1, 63, 10, 20], missing=[0, 63, 11, 20]),
    }
    for method in analysis.METHODS:
        _write_domain(
            tmp_path,
            seed=1,
            method=method,
            domain_id="sunny/scene",
            labels=labels,
            predictions=predictions[method],
        )

    bundles = analysis.load_bundles(tmp_path, methods=analysis.METHODS, seeds=(1,), expected_domains=1)
    rows, coverage = analysis.build_domain_mask_metrics(bundles)
    pair_scope = "pairwise:T2:amber_full"
    t2_drop80 = [
        row
        for row in rows
        if row["method"] == "T2" and row["scope"] == pair_scope and row["rate"] == pytest.approx(0.8)
    ]
    assert len(t2_drop80) == 16
    assert {row["subset_count"] for row in t2_drop80} == {2}
    assert all(row["coverage"] == pytest.approx(0.5) for row in t2_drop80)
    assert all(row["exact"] == pytest.approx(0.5) for row in t2_drop80)
    assert all(row["within1"] == pytest.approx(0.5) for row in t2_drop80)
    assert all(row["mae"] == pytest.approx(5.5) for row in t2_drop80)
    assert all(row["true_margin_delta"] == pytest.approx(-8.0) for row in t2_drop80)
    assert all(row["normalized_js"] > 0 for row in t2_drop80)
    amber_drop80 = [
        row
        for row in rows
        if row["method"] == "amber_full" and row["scope"] == pair_scope and row["rate"] == pytest.approx(0.8)
    ]
    assert all(row["exact"] == pytest.approx(0.0) for row in amber_drop80)
    assert all(row["mae"] == pytest.approx(1.0) for row in amber_drop80)
    three_way = next(row for row in coverage if row["scope_kind"] == "three_way_common_clean")
    assert three_way["common_count"] == 1
    assert three_way["coverage"] == pytest.approx(0.25)

    clean_logits = np.asarray([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    missing_logits = np.asarray([[1.0, 2.0, 0.0], [0.0, 3.0, 2.0]])
    target = np.asarray([0, 2])
    np.testing.assert_allclose(analysis.true_class_margin(clean_logits, target), [2.0, 2.0])
    np.testing.assert_allclose(analysis.true_class_margin(missing_logits, target), [-1.0, -1.0])
    np.testing.assert_allclose(analysis.normalized_js_divergence(clean_logits, clean_logits), 0.0)
    js = analysis.normalized_js_divergence(clean_logits, missing_logits)
    np.testing.assert_allclose(js, analysis.normalized_js_divergence(missing_logits, clean_logits))
    assert np.all((0.0 <= js) & (js <= 1.0))
    np.testing.assert_array_equal(analysis.circular_beam_distance([63, 0], [0, 63]), [1, 1])

    output_dir = tmp_path / "summary"
    result = analysis.summarize_task_outputs(
        tmp_path,
        output_dir,
        methods=analysis.METHODS,
        seeds=(1,),
        expected_domains=1,
    )
    amber_delta = next(row for row in result["deltas"] if row["baseline"] == "amber_full")
    assert amber_delta["baseline_reproduction_scope"] == "amber_full_local_adaptation"
    assert amber_delta["baseline_paper_equivalent"] is False
    for name in (
        "domain_mask_metrics.csv",
        "domain_rate_metrics.csv",
        "seed_rate_metrics.csv",
        "multiseed_rate_summary.csv",
        "common_clean_coverage.csv",
        "t2_baseline_domain_deltas.csv",
        "all_sample_robustness_curves.png",
        "all_sample_robustness_curves.pdf",
        "common_clean_robustness_curves.png",
        "common_clean_robustness_curves.pdf",
        "t2_baseline_15domain_heatmap.png",
        "t2_baseline_15domain_heatmap.pdf",
        "summary.md",
    ):
        assert (output_dir / name).stat().st_size > 0
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "不自动得出T2全面最优" in markdown
    assert "必须并列报告" in markdown
    assert "Drop80 T2-AMBER-Full" in markdown


def test_domain_and_mask_aggregation_is_equal_weighted_not_sample_micro(tmp_path, analysis):
    small_labels = np.asarray([0, 1], dtype=np.int16)
    large_labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
    small_predictions = _prediction_matrix(small_labels)
    small_predictions[1:9] = small_labels
    small_predictions[9:17] = (small_labels + 1) % 64
    large_predictions = _prediction_matrix(large_labels)
    large_predictions[1:17] = (large_labels + 1) % 64
    large_predictions[1:17, :2] = large_labels[:2]
    _write_domain(
        tmp_path,
        seed=1,
        method="T2",
        domain_id="sunny/small",
        labels=small_labels,
        predictions=small_predictions,
    )
    _write_domain(
        tmp_path,
        seed=1,
        method="T2",
        domain_id="rainy/large",
        labels=large_labels,
        predictions=large_predictions,
    )

    bundles = analysis.load_bundles(tmp_path, methods=("T2",), seeds=(1,), expected_domains=2)
    mask_rows, _ = analysis.build_domain_mask_metrics(bundles)
    domain_rows = analysis.aggregate_domain_rates(mask_rows)
    seed_rows = analysis.aggregate_seed_rates(domain_rows)
    small = next(
        row
        for row in domain_rows
        if row["scope"] == "all:T2" and row["domain_id"] == "sunny/small" and row["rate"] == pytest.approx(0.2)
    )
    large = next(
        row
        for row in domain_rows
        if row["scope"] == "all:T2" and row["domain_id"] == "rainy/large" and row["rate"] == pytest.approx(0.2)
    )
    macro = next(
        row for row in seed_rows if row["scope"] == "all:T2" and row["rate"] == pytest.approx(0.2)
    )

    assert small["mask_count"] == 16
    assert small["top1"] == pytest.approx(0.5)
    assert large["top1"] == pytest.approx(0.25)
    assert macro["top1"] == pytest.approx((0.5 + 0.25) / 2)
    assert macro["top1"] != pytest.approx((2 * 0.5 + 8 * 0.25) / 10)
    assert macro["domain_count"] == 2
    assert macro["coverage_domain_macro"] == pytest.approx(1.0)


def test_loader_rejects_mixed_domain_checkpoint_and_duplicate_masks(tmp_path, analysis):
    labels = np.asarray([0, 1], dtype=np.int16)
    first = _write_domain(tmp_path, seed=1, method="T2", domain_id="sunny/a", labels=labels)
    second = _write_domain(tmp_path, seed=1, method="T2", domain_id="rainy/b", labels=labels)
    _rewrite_npz(second, "checkpoint_sha256", np.asarray("other-checkpoint"))
    with pytest.raises(ValueError, match="domain provenance mismatch field=checkpoint_sha256"):
        analysis.load_bundles(tmp_path, methods=("T2",), seeds=(1,), expected_domains=2)

    with np.load(first, allow_pickle=False) as payload:
        checkpoint_sha256 = payload["checkpoint_sha256"].copy()
    _rewrite_npz(second, "checkpoint_sha256", checkpoint_sha256)
    with np.load(first, allow_pickle=False) as payload:
        duplicate = payload["mask_digests"].copy()
    duplicate[2] = duplicate[1]
    _rewrite_npz(first, "mask_digests", duplicate)
    with pytest.raises(ValueError, match="mask_digests must contain 16 unique values"):
        analysis.load_bundles(tmp_path, methods=("T2",), seeds=(1,), expected_domains=2)


def test_loader_rejects_stale_checkpoint_artifacts(tmp_path, analysis):
    _write_domain(tmp_path, seed=1, method="T2", domain_id="sunny/a", labels=np.asarray([0, 1]))
    checkpoint = tmp_path / "_runs" / "T2" / "seed1" / "checkpoints" / "last.pth"
    checkpoint.write_bytes(b"changed-after-extraction")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        analysis.load_bundles(tmp_path, methods=("T2",), seeds=(1,), expected_domains=1)


def test_empty_common_domain_invalidates_macro_and_multiseed_value(analysis):
    def row(domain_id, subset_count, top1):
        item = {
            "seed": 1,
            "method": "T2",
            "method_label": "T2",
            "reproduction_scope": "project_mainline",
            "paper_equivalent": False,
            "temporal_result_scope": "mainline_local_validation",
            "domain_id": domain_id,
            "rate": 0.8,
            "scope": "pairwise:T2:amber_full",
            "scope_kind": "pairwise_common_clean",
            "scope_methods": "T2,amber_full",
            "sample_count": 4,
            "subset_count": subset_count,
            "coverage": subset_count / 4,
            "status": "available" if subset_count else "unavailable_empty_common_clean",
        }
        item.update({metric: top1 for metric in analysis.METRICS})
        return item

    seed_row = analysis.aggregate_seed_rates([row("domain-a", 2, 0.5), row("domain-b", 0, np.nan)])[0]
    assert seed_row["status"] == "unavailable_empty_domains"
    assert seed_row["eligible_domain_count"] == 1
    assert seed_row["empty_domain_count"] == 1
    assert seed_row["coverage_domain_macro"] == pytest.approx(0.25)
    assert np.isnan(seed_row["top1"])
    multiseed = analysis.aggregate_multiseed_rates([seed_row], requested_seed_count=1)[0]
    assert multiseed["status"] == "partial"
    assert np.isnan(multiseed["top1_mean"])


def test_nonfinite_domain_metric_invalidates_seed_macro(analysis):
    rows = []
    for domain_id in ("domain-a", "domain-b"):
        row = {
            "seed": 1,
            "method": "T2",
            "method_label": "T2",
            "reproduction_scope": "project_mainline",
            "paper_equivalent": False,
            "temporal_result_scope": "mainline_local_validation",
            "domain_id": domain_id,
            "rate": 0.8,
            "scope": "all:T2",
            "scope_kind": "all",
            "scope_methods": "T2",
            "sample_count": 4,
            "subset_count": 4,
            "coverage": 1.0,
            "status": "available",
        }
        row.update({metric: 0.5 for metric in analysis.METRICS})
        rows.append(row)
    rows[1]["relative_clean_top1"] = np.nan
    result = analysis.aggregate_seed_rates(rows)[0]
    assert result["status"] == "unavailable_nonfinite_metrics"
    assert result["nonfinite_metrics"] == "relative_clean_top1"
    assert np.isnan(result["top1"])


def _prediction_matrix(labels, *, clean=None, missing=None):
    labels = np.asarray(labels, dtype=np.int16)
    result = np.repeat(labels[None, :], 65, axis=0)
    if clean is not None:
        result[0] = np.asarray(clean, dtype=np.int16)
    if missing is not None:
        result[1:] = np.asarray(missing, dtype=np.int16)
    return result


def _write_domain(
    root: Path,
    *,
    seed: int,
    method: str,
    domain_id: str,
    labels: np.ndarray,
    predictions: np.ndarray | None = None,
) -> Path:
    labels = np.asarray(labels, dtype=np.int16)
    predictions = _prediction_matrix(labels) if predictions is None else np.asarray(predictions, dtype=np.int16)
    logits = np.full((*predictions.shape, 64), -4.0, dtype=np.float32)
    rows = np.arange(predictions.shape[0])[:, None]
    samples = np.arange(predictions.shape[1])[None, :]
    logits[rows, samples, predictions] = 4.0
    rates = np.asarray([0.0, *([0.2] * 16), *([0.4] * 16), *([0.6] * 16), *([0.8] * 16)], dtype=np.float32)
    mask_digests = np.asarray(["clean", *[f"m{index}" for index in range(1, 65)]])
    cache_checksums = np.asarray([f"cache-{int(round(rate * 100))}" for rate in rates])
    condition, scene = domain_id.split("/", 1)
    checkpoint = root / "_runs" / method / f"seed{seed}" / "checkpoints" / "last.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(f"checkpoint-{method}-seed{seed}".encode())
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    method_dir = root / f"seed{seed}" / method
    target = method_dir / "domains" / f"{condition}__{scene}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        logits=logits,
        predictions=predictions,
        labels=labels,
        sample_ids=np.asarray([f"{domain_id}:{index}" for index in range(labels.size)]),
        rates=rates,
        mask_digests=mask_digests,
        cache_checksums=cache_checksums,
        domain_id=np.asarray(domain_id),
        condition=np.asarray(condition),
        scene=np.asarray(scene),
        sample_csv_sha256=np.asarray(f"csv-{domain_id}"),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        seed=np.asarray(seed, dtype=np.int16),
    )
    worker_path = method_dir / "worker_0_of_1.json"
    completed_domains = []
    if worker_path.exists():
        completed_domains = json.loads(worker_path.read_text(encoding="utf-8"))["completed_domains"]
    if domain_id not in completed_domains:
        completed_domains.append(domain_id)
    worker_path.write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "domain_shard_index": 0,
                "domain_shard_count": 1,
                "completed_domains": completed_domains,
            }
        ),
        encoding="utf-8",
    )
    return target


def _rewrite_npz(path: Path, field: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays[field] = value
    np.savez_compressed(path, **arrays)
