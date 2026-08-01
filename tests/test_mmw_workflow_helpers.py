import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_only_plans_u0_and_retained_baselines(tmp_path: Path) -> None:
    launcher = _script_module("launch_mmw_all_weather_matrix")

    assert launcher.METHODS == ("U0", "amber_full", "rmbp_mm")
    jobs = launcher.build_job_matrix(("U0", "amber_full"), (1,), (0, 1), tmp_path)

    assert [(job["method"], job["train_seed"], job["gpu"]) for job in jobs] == [
        ("U0", 1, 0),
        ("amber_full", 1, 1),
    ]
    with pytest.raises(ValueError, match="members"):
        launcher.build_job_matrix(("T2",), (1,), (0,), tmp_path)


def test_evaluator_temporal_cache_is_small_and_self_validating(tmp_path: Path) -> None:
    evaluator = _script_module("eval_mmw_all_weather_matrix")

    cache = evaluator._load_or_create_temporal_cache(
        tmp_path,
        history_window=5,
        modality_frame_masks=1,
        rates=(0.0, 0.2),
        mask_types=("modality_frame",),
    )

    assert set(cache) == {0.0, 0.2}
    assert all(payload["version"] == evaluator.MASK_CACHE_VERSION for payload in cache.values())
    assert (tmp_path / "rate_0.2.json").is_file()


def test_summary_compares_u0_only_against_retained_baselines() -> None:
    summary = _script_module("summarize_mmw_all_weather_matrix")
    protocol = {
        "split_protocol": "mmw_id_stratified_block_v1",
        "protocol_version": "1",
        "split_seed": "0",
        "block_size": "128",
        "split_manifest_hash": "a" * 64,
        "data_source_hash": "b" * 64,
        "window_config_hash": "c" * 64,
        "weather_binding": "True",
    }
    rows = [
        {"method": "U0", "train_seed": "0", "domain_id": "d", "pattern": "full", "top1": "0.8", **protocol},
        {"method": "amber_full", "train_seed": "0", "domain_id": "d", "pattern": "full", "top1": "0.7", **protocol},
        {"method": "rmbp_mm", "train_seed": "0", "domain_id": "d", "pattern": "full", "top1": "0.6", **protocol},
    ]

    summary._validate_rows(rows)
    deltas = summary._paired_deltas(rows)

    assert {row["baseline"] for row in deltas} == {"amber_full", "rmbp_mm"}
    assert {row["baseline"]: row["delta_top1"] for row in deltas} == pytest.approx(
        {"amber_full": 0.1, "rmbp_mm": 0.2}
    )

    rows[-1]["split_manifest_hash"] = "d" * 64
    with pytest.raises(ValueError, match="different MMW protocol"):
        summary._validate_rows(rows)
