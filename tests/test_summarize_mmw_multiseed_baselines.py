import csv
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_mmw_multiseed_baselines.py"
METHODS = ("T2", "amber_full", "rmbp_mm")
SCOPES = {
    "T2": ("project_mainline", "mainline_local_validation"),
    "amber_full": ("amber_full_local_adaptation", "local_adaptation_diagnostic"),
    "rmbp_mm": ("rmbp_mm_channel_attention_local", "out_of_paper_scope_diagnostic"),
}


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_mmw_multiseed_baselines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(
    method: str,
    seed: int,
    domain: int,
    checkpoint: Path,
    *,
    rate: float,
    mask_type: str,
    top1: float,
    family: str = "temporal_missing",
    mask_index: int = 0,
) -> dict:
    reproduction_scope, temporal_scope = SCOPES[method]
    return {
        "method": method,
        "seed": seed,
        "domain_id": f"sunny/domain{domain:02d}",
        "condition": "sunny",
        "scene": f"domain{domain:02d}",
        "sample_count": 10,
        "sample_csv_sha256": f"csv-{domain:02d}",
        "reproduction_scope": reproduction_scope,
        "paper_equivalent": "False",
        "temporal_result_scope": temporal_scope,
        "checkpoint": str(checkpoint),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "eval_family": family,
        "pattern": "full" if family == "whole_modality" else mask_type,
        "available_modalities": "image,radar,lidar,gps",
        "missing_rate": rate,
        "drop_count": 0,
        "mask_index": mask_index,
        "mask_type": "whole_modality" if family == "whole_modality" else mask_type,
        "mask_digest": "whole-full" if family == "whole_modality" else f"{rate:g}-{mask_type}-{mask_index}",
        "mask_cache_checksum": "whole-cache" if family == "whole_modality" else "shared-main-cache",
        "mask_cache_seed": 20260713,
        "observed_missing_rate": rate,
        "last_frame_available": "True",
        "last_frame_available_modalities": 4 if rate == 0.0 else 1,
        "trailing_fully_missing_frames": 0,
        "top1": top1,
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main_eval"
    extreme = tmp_path / "extreme_eval"
    base = {"T2": 0.8, "amber_full": 0.7, "rmbp_mm": 0.5}
    slope = {"T2": 0.10, "amber_full": 0.15, "rmbp_mm": 0.30}
    for method in METHODS:
        for seed in (1, 2, 3):
            checkpoint = tmp_path / "runs" / method / f"seed{seed}" / "checkpoints" / "last.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            main_rows = []
            extreme_rows = []
            seed_offset = (seed - 2) * 0.01
            for domain in range(15):
                clean = base[method] + seed_offset + domain * 0.001
                main_rows.append(_row(method, seed, domain, checkpoint, rate=0.0, mask_type="clean", top1=clean, family="whole_modality"))
                main_rows.append(_row(method, seed, domain, checkpoint, rate=0.0, mask_type="clean", top1=clean))
                for rate in (0.2, 0.4, 0.6, 0.8):
                    for mask_type in ("frame_level", "block", "modality_frame"):
                        count = module_counts(rate, mask_type)
                        for mask_index in range(count):
                            main_rows.append(
                                _row(
                                    method,
                                    seed,
                                    domain,
                                    checkpoint,
                                    rate=rate,
                                    mask_type=mask_type,
                                    top1=clean - slope[method] * rate,
                                    mask_index=mask_index,
                                )
                            )
                for rate in (0.85, 0.9, 0.95):
                    for mask_index in range(16):
                        row = _row(
                            method,
                            seed,
                            domain,
                            checkpoint,
                            rate=rate,
                            mask_type="modality_frame",
                            top1=clean - slope[method] * rate,
                            mask_index=mask_index,
                        )
                        row["mask_cache_checksum"] = "shared-extreme-cache"
                        extreme_rows.append(row)
            _write_csv(main / method / f"seed{seed}" / "metrics.csv", main_rows)
            _write_csv(extreme / method / f"seed{seed}" / "metrics.csv", extreme_rows)
    return main, extreme


def module_counts(rate: float, mask_type: str) -> int:
    return {
        0.2: {"frame_level": 5, "block": 5, "modality_frame": 16},
        0.4: {"frame_level": 10, "block": 4, "modality_frame": 16},
        0.6: {"frame_level": 10, "block": 3, "modality_frame": 16},
        0.8: {"frame_level": 5, "block": 2, "modality_frame": 16},
    }[rate][mask_type]


def test_multiseed_summary_keeps_seeds_layered_and_bootstraps_paired_domains(tmp_path: Path):
    module = _load_script()
    main, extreme = _build_fixture(tmp_path)
    output = tmp_path / "summary"

    result = module.summarize(
        main,
        output,
        extreme_eval_dir=extreme,
        bootstrap_iterations=64,
        bootstrap_seed=7,
    )

    assert len(result["per_seed"]) == 9
    assert len(result["domain_summary"]) == 135
    assert len(result["weather_summary"]) == 9
    assert len(result["scene_summary"]) == 135
    assert len(result["worst_domains"]) == 9
    t2 = next(row for row in result["multiseed"] if row["method"] == "T2")
    assert t2["aggregation_status"] == "complete"
    assert math.isclose(t2["clean_top1_mean"], 0.807)
    assert math.isclose(t2["clean_top1_std"], 0.01)
    assert math.isclose(t2["auc_top1_0_80_mean"], 0.767)
    assert math.isclose(t2["drop80_top1_mean"], 0.727)

    amber = next(row for row in result["comparisons"] if row["baseline"] == "amber_full")
    assert amber["status"] == "supported"
    assert amber["auc_positive_seed_count"] == 3
    assert math.isclose(amber["clean_top1_delta_mean"], 0.1)
    assert math.isclose(amber["auc_top1_0_80_delta_mean"], 0.12)
    assert math.isclose(amber["drop80_top1_delta_mean"], 0.14)
    assert math.isclose(amber["auc_top1_0_80_delta_ci_low"], 0.12)
    assert math.isclose(amber["auc_top1_0_80_delta_ci_high"], 0.12)
    assert amber["baseline_reproduction_scope"] == "amber_full_local_adaptation"
    assert len([row for row in result["paired_domains"] if row["baseline"] == "amber_full"]) == 45

    extreme_90 = next(
        row
        for row in result["extreme_curves"]
        if row["method"] == "T2" and row["seed"] == "all" and row["missing_rate"] == 0.9
    )
    assert extreme_90["aggregation_status"] == "complete"
    assert math.isclose(extreme_90["top1_mean"], 0.717)
    extreme_80 = next(
        row
        for row in result["extreme_curves"]
        if row["method"] == "T2" and row["seed"] == "all" and row["missing_rate"] == 0.8
    )
    assert math.isclose(extreme_80["top1_mean"], 0.727)
    assert (output / "decision.json").exists()
    assert (output / "top1_robustness_curves.png").stat().st_size > 0
    assert (output / "top1_robustness_curves.pdf").stat().st_size > 0
    assert _read_csv(output / "per_seed_summary.csv")
    assert _read_csv(output / "per_seed_domain_summary.csv")
    assert _read_csv(output / "per_seed_weather_summary.csv")
    assert _read_csv(output / "per_seed_scene_summary.csv")
    assert _read_csv(output / "per_seed_worst_domain_summary.csv")


def test_cross_seed_mask_identity_drift_marks_only_bad_unit_unavailable(tmp_path: Path):
    module = _load_script()
    main, _ = _build_fixture(tmp_path)
    path = main / "amber_full" / "seed3" / "metrics.csv"
    rows = _read_csv(path)
    for row in rows:
        if row["missing_rate"] == "0.4" and row["mask_type"] == "block" and row["mask_index"] == "0":
            row["mask_digest"] = "drifted-mask-0"
    _write_csv(path, rows)

    result = module.summarize(main, tmp_path / "summary", bootstrap_iterations=16)

    availability = next(
        row for row in result["availability"] if row["method"] == "amber_full" and row["seed"] == 3
    )
    assert availability["main_status"] == "unavailable"
    assert availability["main_reason"] == "cross_method_seed_mask_identity_mismatch"
    aggregate = next(row for row in result["multiseed"] if row["method"] == "amber_full")
    assert aggregate["aggregation_status"] == "partial"
    assert aggregate["clean_top1_mean"] == ""
    comparison = next(row for row in result["comparisons"] if row["baseline"] == "amber_full")
    assert comparison["status"] == "partial"
    assert comparison["gate_complete_seed_set"] is False


def test_missing_checkpoint_is_unavailable_and_not_used_as_three_seed_mean(tmp_path: Path):
    module = _load_script()
    main, _ = _build_fixture(tmp_path)
    checkpoint = tmp_path / "runs" / "rmbp_mm" / "seed2" / "checkpoints" / "last.pth"
    checkpoint.unlink()

    result = module.summarize(main, tmp_path / "summary", bootstrap_iterations=16)

    availability = next(
        row for row in result["availability"] if row["method"] == "rmbp_mm" and row["seed"] == 2
    )
    assert availability["main_status"] == "unavailable"
    assert availability["main_reason"] == "checkpoint_missing_or_inconsistent"
    aggregate = next(row for row in result["multiseed"] if row["method"] == "rmbp_mm")
    assert aggregate["available_seed_count"] == 2
    assert aggregate["aggregation_status"] == "partial"
    assert aggregate["auc_top1_0_80_mean"] == ""
    comparison = next(row for row in result["comparisons"] if row["baseline"] == "rmbp_mm")
    assert comparison["status"] == "partial"
    assert comparison["paired_seed_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("scope", "baseline_scope_mismatch"),
        ("best_checkpoint", "checkpoint_not_method_seed_last_pth"),
        ("duplicate_mask", "mask_row_count_mismatch_0.2_modality_frame_241"),
    ],
)
def test_invalid_scope_checkpoint_or_duplicate_mask_is_unavailable(tmp_path: Path, mutation: str, reason: str):
    module = _load_script()
    main, _ = _build_fixture(tmp_path)
    path = main / "amber_full" / "seed1" / "metrics.csv"
    rows = _read_csv(path)
    if mutation == "scope":
        for row in rows:
            row["reproduction_scope"] = "project_mainline"
    elif mutation == "best_checkpoint":
        best = tmp_path / "runs/amber_full/seed1/checkpoints/best.pth"
        best.touch()
        for row in rows:
            row["checkpoint"] = str(best)
    else:
        duplicate = next(
            row.copy()
            for row in rows
            if row["eval_family"] == "temporal_missing"
            and row["missing_rate"] == "0.2"
            and row["mask_type"] == "modality_frame"
        )
        duplicate["mask_index"] = "999"
        rows.append(duplicate)
    _write_csv(path, rows)

    result = module.summarize(main, tmp_path / "summary", bootstrap_iterations=16)

    availability = next(
        row for row in result["availability"] if row["method"] == "amber_full" and row["seed"] == 1
    )
    assert availability["main_status"] == "unavailable"
    assert availability["main_reason"] == reason
