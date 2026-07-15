import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def analysis():
    path = ROOT / "scripts" / "summarize_mmw_t2_bpa_cma_ablation.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slices_metrics_auc_and_preregistered_delta_direction(analysis):
    bundles = _bundles(analysis, seeds=(1, 2, 3))
    seed_rows = analysis.build_seed_rate_slice_metrics(bundles)

    assert len(seed_rows) == 3 * len(analysis.METHODS) * len(analysis.SLICES) * 5
    endpoint_linear = _row(
        seed_rows,
        seed=1,
        method="T2-Linear",
        slice="exact_endpoint",
        rate=0.8,
    )
    assert endpoint_linear["domain_count"] == 2
    assert endpoint_linear["eligible_domain_count"] == 1
    assert endpoint_linear["sample_count"] == 2
    assert endpoint_linear["top1"] == pytest.approx(0.0)
    assert endpoint_linear["within1"] == pytest.approx(1.0)
    assert endpoint_linear["circular_mae"] == pytest.approx(1.0)

    all_no_bpa = _row(seed_rows, seed=1, method="T2-NoBPA", slice="all", rate=0.8)
    assert all_no_bpa["top1"] == pytest.approx(((5 / 6) + 1.0) / 2)

    auc_rows = analysis.build_missing_auc(seed_rows)
    t2_auc = _row(auc_rows, seed=1, method="T2", slice="all")
    assert t2_auc["rate_min"] == pytest.approx(0.2)
    assert t2_auc["rate_max"] == pytest.approx(0.8)
    assert t2_auc["top1_auc"] == pytest.approx(1.0)

    deltas = analysis.build_paired_deltas(seed_rows, value_suffix="")
    topology = _row(deltas, seed=1, comparison="circular_vs_linear", slice="exact_endpoint", rate=0.8)
    assert topology["left_method"] == "T2"
    assert topology["right_method"] == "T2-Linear"
    assert topology["top1_delta"] == pytest.approx(1.0)
    assert topology["within1_delta"] == pytest.approx(0.0)
    assert topology["circular_mae_delta"] == pytest.approx(-1.0)
    cma = _row(deltas, seed=1, comparison="cma_without_prototypes", slice="all", rate=0.8)
    assert cma["top1_delta"] > 0


def test_summary_reuses_strict_loader_and_writes_all_paper_artifacts(tmp_path, monkeypatch, analysis):
    bundles = _bundles(analysis, seeds=(1,))
    called = {}

    def fake_load(raw_root, *, methods, seeds, expected_domains):
        called.update(raw_root=raw_root, methods=methods, seeds=seeds, expected_domains=expected_domains)
        return bundles

    monkeypatch.setattr(analysis.task_output, "load_bundles", fake_load)
    output_dir = tmp_path / "summary"
    result = analysis.summarize_ablation(tmp_path / "raw", output_dir, seeds=(1,), expected_domains=2)

    assert called == {
        "raw_root": tmp_path / "raw",
        "methods": analysis.METHODS,
        "seeds": (1,),
        "expected_domains": 2,
    }
    assert len(result["multiseed_rate"]) == len(analysis.METHODS) * len(analysis.SLICES) * 5
    expected = (
        "per_seed_rate_slice_metrics.csv",
        "multiseed_rate_slice_metrics.csv",
        "per_seed_missing_auc.csv",
        "multiseed_missing_auc.csv",
        "per_seed_paired_rate_deltas.csv",
        "multiseed_paired_rate_deltas.csv",
        "per_seed_paired_auc_deltas.csv",
        "multiseed_paired_auc_deltas.csv",
        "objective_ablation_curves.png",
        "objective_ablation_curves.pdf",
        "topology_endpoint_deltas.png",
        "topology_endpoint_deltas.pdf",
        "classifier_cma_curves.png",
        "classifier_cma_curves.pdf",
        "summary.json",
        "summary.md",
    )
    for name in expected:
        assert (output_dir / name).stat().st_size > 0
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["cma_scope"].startswith("AMBER-style objective analogue")
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "不筛选样本、domain 或 seed" in markdown
    assert "不是完整 AMBER Class-Former 复现" in markdown
    assert "T2 - T2: linear BPA" in markdown


def _bundles(analysis, *, seeds):
    rates = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    domain_labels = {
        "sunny/endpoint": np.asarray([0, 63, 62, 1, 10, 20], dtype=np.int64),
        "rainy/interior": np.asarray([2, 3, 4, 5], dtype=np.int64),
    }
    result = {}
    for seed in seeds:
        result[seed] = {}
        for method in analysis.METHODS:
            result[seed][method] = {}
            for domain_id, labels in domain_labels.items():
                predictions = np.tile(labels, (rates.size, 1))
                if domain_id == "sunny/endpoint":
                    if method == "T2-NoBPA":
                        predictions[-1, 0] = 1
                    elif method == "T2-Linear":
                        predictions[-1, :2] = (63, 0)
                    elif method == "T2-CLS":
                        predictions[-1, -1] = 24
                result[seed][method][domain_id] = {
                    "labels": labels.copy(),
                    "predictions": predictions,
                    "rates": rates.copy(),
                }
    return result


def _row(rows, **expected):
    selected = [row for row in rows if all(row.get(key) == pytest.approx(value) if isinstance(value, float) else row.get(key) == value for key, value in expected.items())]
    assert len(selected) == 1
    return selected[0]
