import csv
from pathlib import Path

from kd_sensing.diagnostics.scene31_34_final_analysis import error_cdf
from kd_sensing.diagnostics.scene31_34_final_analysis import final_paper_tables as final_tables
from kd_sensing.diagnostics.scene31_34_final_analysis import pattern_heatmap as heatmap
from kd_sensing.diagnostics.scene31_34_final_analysis import presentation_artifacts as presentation
from kd_sensing.diagnostics.scene31_34_final_analysis import profile
from kd_sensing.diagnostics.scene31_34_final_analysis import sampling_distribution as sampling
from kd_sensing.diagnostics.scene31_34_final_analysis import significance


SUBSET = "scenes31_34_proto_randomdrop_subset_es40"
BERNOULLI = "scenes31_34_proto_randomdrop_bernoulli_k075_es40"
CLASSIFIER = "scenes31_34_classifier_randomdrop_subset_es40"
NATURAL = "scenes31_34_proto_natural_es40"
UNIFORM = "scenes31_34_proto_sampler_uniform_es40"


def test_significance_reads_summary_fallback(tmp_path: Path):
    root = tmp_path / "main"
    summary = root / "summary"
    summary.mkdir(parents=True)
    _write_csv(
        summary / "method_mean_std.csv",
        [
            _summary_row(SUBSET, 0.36, 0.25, 2.6),
            _summary_row(BERNOULLI, 0.34, 0.22, 3.1),
        ],
    )
    out = tmp_path / "stats"
    significance.main(["--root", str(root), "--out", str(out), "--paper-table-root", str(tmp_path / "paper"), "--bootstrap", "10"])
    rows = _read_csv(out / "significance_summary.csv")
    assert rows
    avg = next(row for row in rows if row["metric"] == "avg_missing_top1")
    assert "seed_mean_delta_fraction" in avg
    assert "bootstrap_mean_delta_fraction" in avg
    assert "bootstrap_ci_low_pp" in avg
    assert "delta" not in avg
    assert (out / "warnings.txt").read_text(encoding="utf-8").startswith("WARNING:")


def test_heatmap_outputs_nonempty(tmp_path: Path):
    root = tmp_path / "main"
    classifier_root = tmp_path / "classifier"
    for idx, method in enumerate([SUBSET, BERNOULLI, NATURAL, UNIFORM]):
        _write_eval(root, method, 1, top1=0.4 + idx * 0.01, mae=3.0 - idx * 0.1)
    _write_eval(classifier_root, CLASSIFIER, 1, top1=0.35, mae=3.4)
    out = tmp_path / "pattern"
    heatmap.main(["--root", str(root), "--classifier-root", str(classifier_root), "--out", str(out)])
    rows = _read_csv(out / "pattern_win_count_summary.csv")
    assert rows
    assert (out / "fig_pattern_heatmap_top1.png").exists()
    assert (out / "fig_pattern_delta_vs_bernoulli_top1_paper.png").exists()
    assert (out / "fig_pattern_heatmap_top1_grouped_paper.png").exists()


def test_sampling_distribution_probabilities_sum_to_one(tmp_path: Path):
    out = tmp_path / "sampling"
    sampling.main(["--root", str(tmp_path / "main"), "--out", str(out)])
    rows = _read_csv(out / "sampling_distribution_by_missing_count.csv")
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], 0.0)
        by_method[row["method"]] += float(row["probability"])
    assert by_method
    assert all(abs(total - 1.0) < 1e-6 for total in by_method.values())


def test_error_cdf_is_monotonic(tmp_path: Path):
    root = tmp_path / "main"
    for idx, method in enumerate([SUBSET, BERNOULLI, NATURAL, UNIFORM, CLASSIFIER]):
        _write_eval(root, method, 1, top1=0.4, mae=2.0 + idx)
    out = tmp_path / "cdf"
    error_cdf.main(["--root", str(root), "--out", str(out)])
    rows = _read_csv(out / "abs_error_cdf_data.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["condition"]), []).append((int(row["abs_error_threshold"]), float(row["cdf"])))
    assert grouped
    for values in grouped.values():
        ordered = [cdf for _, cdf in sorted(values)]
        assert ordered == sorted(ordered)
    assert (out / "fig_abs_error_cdf_all_missing_paper.png").exists()
    assert (out / "fig_abs_error_cdf_all_missing_presentation.png").exists()
    assert "higher and left-shifted" in (out / "caption_notes.txt").read_text(encoding="utf-8")


def test_final_notes_and_mask_suspect_external_exclusion(tmp_path: Path):
    summary = tmp_path / "summary"
    paper = tmp_path / "paper"
    summary.mkdir()
    _write_csv(
        summary / "final_method_mean_std.csv",
        [
            _final_row(SUBSET, "proto", 5, 0.36, mask_suspect=0, official="true"),
            _final_row("scenes31_34_amber_lite_uniform_es40", "external_lite", 1, 0.30, mask_suspect=1, official="false"),
        ],
    )
    final_tables.main(
        [
            "--summary-root",
            str(summary),
            "--statistics-root",
            str(tmp_path / "stats"),
            "--pattern-root",
            str(tmp_path / "pattern"),
            "--profile-root",
            str(tmp_path / "profile"),
            "--cdf-root",
            str(tmp_path / "cdf"),
            "--sampling-root",
            str(tmp_path / "sampling"),
            "--paper-table-root",
            str(paper),
        ]
    )
    assert "Final trusted method: prototype + random non-empty subset exposure." in (
        paper / "scenes31_34_final_paper_notes.txt"
    ).read_text(encoding="utf-8")
    assert "No further model-search experiments are recommended." in (
        paper / "scenes31_34_final_paper_notes.txt"
    ).read_text(encoding="utf-8")
    assert "excluded" in (paper / "table_scenes31_34_external_baselines.md").read_text(encoding="utf-8")


def test_compute_cost_table_uses_final_columns():
    rows = profile._paper_rows(
        [
            {
                "method": SUBSET,
                "family": "proto",
                "num_params": 10,
                "model_size_mb": 1.25,
                "eval_latency_per_sample_ms": 0.5,
                "eval_samples_per_second": 2000,
                "gpu_memory_peak_mb": 64,
                "extra_inference_cost": "None; training-only exposure strategy",
            }
        ]
    )
    assert list(rows[0]) == [
        "Method",
        "Family",
        "Params",
        "Model size",
        "Latency / sample",
        "Samples / second",
        "GPU memory",
        "Extra inference cost",
    ]


def test_presentation_artifacts_outputs(tmp_path: Path):
    stats = tmp_path / "stats"
    _write_csv(
        stats / "significance_summary.csv",
        [
            {
                "comparison": "Proto random subset vs Proto Bernoulli randomdrop",
                "metric": "avg_missing_top1",
                "bootstrap_mean_delta_pp": "2.56",
            },
            {
                "comparison": "Proto random subset vs Classifier random subset",
                "metric": "avg_missing_top1",
                "bootstrap_mean_delta_pp": "1.96",
            },
        ],
    )
    out = tmp_path / "presentation"
    paper = tmp_path / "paper"
    presentation.main(["--statistics-root", str(stats), "--paper-table-root", str(paper), "--out", str(out)])
    assert (out / "fig_method_overview_presentation.png").exists()
    assert (out / "fig_related_work_positioning_presentation.png").exists()
    assert (paper / "table_related_work_positioning.md").exists()


def _write_eval(root: Path, method: str, seed: int, *, top1: float, mae: float) -> None:
    run_name = f"{method}_seed{seed}"
    eval_dir = root / "fresh_eval_with_scene" / run_name
    eval_dir.mkdir(parents=True)
    pattern_rows = [
        _pattern_row(run_name, method, seed, "full", 0, top1, mae),
        _pattern_row(run_name, method, seed, "missing_gps", 1, top1 - 0.05, mae + 0.5),
        _pattern_row(run_name, method, seed, "gps_only", 3, top1 - 0.12, mae + 1.0),
    ]
    _write_csv(eval_dir / "pattern_metrics.csv", pattern_rows)
    pred_rows = []
    for scene in ("Scene31", "Scene32"):
        for sample in range(3):
            pred_rows.extend(
                [
                    _pred_row(run_name, method, seed, scene, sample, "full", 0, 1, 0),
                    _pred_row(run_name, method, seed, scene, sample, "missing_gps", 1, int(sample % 2 == 0), 2),
                    _pred_row(run_name, method, seed, scene, sample, "gps_only", 3, int(sample == 0), 4),
                ]
            )
    _write_csv(eval_dir / "predictions_by_pattern.csv", pred_rows)
    (eval_dir / "checkpoint_manifest.json").write_text("{}\n", encoding="utf-8")


def _pattern_row(run_name, method, seed, pattern, missing_count, top1, mae):
    return {
        "status": "ok",
        "run_name": run_name,
        "method": method,
        "seed": seed,
        "pattern": pattern,
        "missing_count": missing_count,
        "missing_ratio": missing_count / 4,
        "available_modalities": "image,radar,gps,lidar",
        "missing_modalities": "",
        "top1": top1,
        "within3": min(1.0, top1 + 0.4),
        "mae": mae,
        "num_samples": 3,
        "full_top1": top1,
        "miss1_top1": top1 - 0.05,
        "miss2_top1": top1 - 0.08,
        "miss3_top1": top1 - 0.12,
        "avg_missing_top1": top1 - 0.08,
        "overall_mean_top1": top1 - 0.06,
        "avg_missing_within@3": min(1.0, top1 + 0.3),
        "avg_missing_MAE": mae + 0.5,
        "balanced": top1,
        "mask_suspect": "false",
    }


def _pred_row(run_name, method, seed, scene, sample, pattern, missing_count, correct, abs_error):
    return {
        "run_name": run_name,
        "method": method,
        "seed": seed,
        "scene": scene,
        "sample_id": f"{scene}:{sample}",
        "pattern": pattern,
        "target": 10,
        "pred": 10 + abs_error,
        "top1_correct": correct,
        "top3_correct": int(abs_error <= 2),
        "top5_correct": int(abs_error <= 4),
        "within3_correct": int(abs_error <= 3),
        "abs_error": abs_error,
        "missing_count": missing_count,
        "missing_ratio": missing_count / 4,
        "available_modalities": "image,radar,gps,lidar",
        "missing_modalities": "",
    }


def _summary_row(method: str, avg_missing: float, miss3: float, mae: float):
    return {
        "method": method,
        "n": "5",
        "full_top1_mean": "0.45",
        "miss1_top1_mean": "0.42",
        "miss2_top1_mean": "0.35",
        "miss3_top1_mean": str(miss3),
        "avg_missing_top1_mean": str(avg_missing),
        "overall_mean_top1_mean": str(avg_missing),
        "avg_missing_within@3_mean": "0.8",
        "avg_missing_MAE_mean": str(mae),
    }


def _final_row(method: str, family: str, n: int, avg_missing: float, *, mask_suspect: int, official: str):
    row = _summary_row(method, avg_missing, avg_missing - 0.1, 2.0)
    row.update(
        {
            "family": family,
            "n": str(n),
            "top1_drop_0_to_75_mean": "0.2",
            "mae_at_75_mean": "4.0",
            "mask_suspect_count": str(mask_suspect),
            "official_ranking_included": official,
            "main_read": "fixture",
        }
    )
    return row


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
