import importlib.util
from pathlib import Path

import pytest
import torch

from kd_sensing.data.temporal_block_mask import TemporalBlockMaskGenerator
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _domains() -> list[dict[str, str]]:
    return [
        {
            "id": f"weather/scene-{index}",
            "condition": "weather",
            "scene": f"scene-{index}",
            "data_root": "dataset/MMW/sunny",
            "train_csv_name": "/tmp/train.csv",
            "val_csv_name": "/tmp/validation.csv",
            "test_csv_name": "/tmp/test.csv",
        }
        for index in range(15)
    ]


def test_four_configs_share_protocol_and_only_enable_declared_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load("run_quick_pcer_validation.py", monkeypatch)
    configs = {}
    for name, ablation, mode, fusion, oracle in launcher.EXPERIMENTS:
        config = launcher.build_experiment_config(
            tmp_path,
            _domains(),
            name=name,
            ablation=ablation,
            pcer_mode=mode,
            fusion_type=fusion,
            oracle_weight=oracle,
        )
        configs[ablation] = config
        assert config["training"]["epochs"] == 16
        assert config["data"]["dataloader"]["num_workers"] == 12
        assert config["data"]["dataloader"]["prefetch_factor"] == 1
        assert config["training"]["checkpoint_selection"] == "best_validation_loss"
        assert config["training"]["amp"] == {"enabled": True, "dtype": "bfloat16", "grad_scaler": False}
        assert config["temporal_missing"]["mode"] == "pcer_curriculum"
        assert len(config["data"]["dataset"]["domains"]) == 15
        u_mask_beam_jepa_config(config)

    assert configs["A0"]["model"]["primary"]["fusion_type"] == "uniform_mean"
    assert "pcer" not in configs["A0"]["model"]["primary"]
    assert configs["A1"]["model"]["primary"]["fusion_type"] == "supervised_router"
    assert configs["A1"]["loss"]["u_mask_beam_jepa"]["router_oracle_weight"] == pytest.approx(0.1)
    assert configs["A2"]["model"]["primary"]["pcer"]["mode"] == "evidence_static"
    assert configs["A2"]["loss"]["u_mask_beam_jepa"]["pcer"]["lambda_route"] == 0.0
    assert configs["A3"]["model"]["primary"]["pcer"]["mode"] == "counterfactual_router"
    assert configs["A3"]["loss"]["u_mask_beam_jepa"]["pcer"]["lambda_route"] == pytest.approx(0.2)


def test_fixed_evaluation_masks_are_deterministic_balanced_and_delete_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load("eval_quick_pcer_validation.py", monkeypatch)
    ids = [f"sample-{index}" for index in range(24)]
    generator = TemporalBlockMaskGenerator(evaluator.EVAL_SEED)
    first = evaluator._conditions(ids, 0, generator)
    second = evaluator._conditions(ids, 0, generator)
    assert [item["name"] for item in first] == evaluator._condition_names()
    assert all(torch.equal(left["mask"], right["mask"]) for left, right in zip(first, second, strict=True))
    s4 = next(item for item in first if item["family"] == "S4")
    assert (~s4["mask"][:, :, -1]).all()
    s5 = next(item for item in first if item["family"] == "S5")
    assert len(set(s5["groups"])) == 6
    assert all(s5["groups"].count(pair) == 4 for pair in set(s5["groups"]))


def test_metric_accumulator_reports_requested_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load("eval_quick_pcer_validation.py", monkeypatch)
    accumulator = evaluator.MetricAccumulator()
    logits = torch.tensor([[5.0, 1.0, 0.0, -1.0], [0.0, 3.0, 2.0, 1.0]])
    labels = torch.tensor([0, 2])
    powers = torch.tensor([[4.0, 1.0, 1.0, 1.0], [1.0, 2.0, 4.0, 1.0]])
    accumulator.update(logits, labels, powers)
    result = accumulator.result()
    assert result["top1"] == pytest.approx(0.5)
    assert result["top3"] == pytest.approx(1.0)
    assert result["top5"] == pytest.approx(1.0)
    assert result["within3"] == pytest.approx(1.0)
    assert result["normalized_gain"] == pytest.approx(0.75)
