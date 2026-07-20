from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load("run_pgcd_quick_search")
evaluator = _load("eval_pgcd_quick_search")


def test_launcher_builds_fixed_channel_free_c0_c7_configs(tmp_path):
    domains = [{
        "id": "sunny/scene",
        "condition": "sunny",
        "scene": "scene",
        "data_root": "dataset/MMW/sunny",
        "train_csv_name": "train.csv",
        "val_csv_name": "val.csv",
        "test_csv_name": "test.csv",
    }]
    for index, (name, ablation, gpu) in enumerate(launcher.EXPERIMENTS):
        config = launcher.build_experiment_config(tmp_path, domains, name=name, ablation=ablation)
        assert gpu == 4 + index % 2
        assert config["model"]["primary"]["pgcd"]["variant"] == f"c{index}"
        assert config["loss"]["u_mask_beam_jepa"]["pgcd"]["variant"] == f"c{index}"
        assert config["data"]["dataset"]["include_router_utility_targets"] is False
        assert config["mmw_pgcd_protocol"]["use_channel"] is False
        assert config["mmw_pgcd_protocol"]["use_path_features"] is False
        assert config["training"]["epochs"] == 16
        assert config["data"]["dataloader"]["train_batch_size"] == 32


def test_evaluator_has_complete_fixed_e0_e5_inventory():
    conditions = evaluator._conditions()
    protocols = {condition.protocol for condition in conditions}
    assert protocols == {"E0", "E1", "E2", "E3", "E4", "E5"}
    assert len([item for item in conditions if item.protocol == "E1"]) == 20
    assert len([item for item in conditions if item.protocol == "E2"]) == 3
    assert len([item for item in conditions if item.protocol == "E3"]) == 4
    assert len([item for item in conditions if item.protocol == "E4"]) == 4
    assert len([item for item in conditions if item.protocol == "E5"]) == 6
    assert all(item.fixed is not None or item.mask_type is not None for item in conditions)


def test_dynamic_summary_reports_d0_minus_d1_and_d3():
    rows = []
    for replacement, top1 in zip(evaluator.REPLACEMENTS, (0.6, 0.5, 0.55, 0.4), strict=True):
        rows.append({
            "condition": "E1_image_blur_L3",
            "protocol": "E1",
            "severity": 3,
            "replacement": replacement,
            "top1": top1,
        })
    result = evaluator._dynamic_summary(rows)
    severe = next(item for item in result if item["category"] == "Severe corruption")
    assert severe["dynamic_gain"] == pytest.approx(0.1)
    assert severe["quality_gain_over_prior"] == pytest.approx(0.2)


def test_evaluator_source_does_not_load_power_or_channel_arrays():
    source = (SCRIPTS / "eval_pgcd_quick_search.py").read_text(encoding="utf-8").lower()
    assert "future_beam_power" not in source
    assert "future_beam_path" not in source
    assert "np.loadtxt" not in source
    assert "channel_gain" not in source
