import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset, WeightedRandomSampler

from kd_sensing.engine.data_factory import build_domain_balanced_sampler
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_weather_launcher_uses_the_four_tracked_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)

    assert launcher.METHODS == ("S1", "T2", "amber_full", "rmbp_mm")
    assert all(path.startswith("configs/mmw/") for path in launcher.METHOD_BASES.values())
    cfg = launcher.build_config("T2", Path("outputs/test"), smoke=False, epochs=40, batch_size=32)

    assert cfg["data"]["dataset"]["type"] == "mmw"
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert cfg["model"]["primary"]["temporal_pooling"] == {"enabled": True, "type": "masked_mean"}


def test_t2_bpa_cma_ablation_configs_are_explicit_and_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}
    configs = {method: launcher.build_config(method, Path("outputs/ablation"), **kwargs) for method in launcher.T2_ABLATION_METHODS}

    no_bpa = u_mask_beam_jepa_config(configs["T2-NoBPA"])
    cma = u_mask_beam_jepa_config(configs["T2-BPA2CMA"])
    linear = u_mask_beam_jepa_config(configs["T2-Linear"])
    classifier = u_mask_beam_jepa_config(configs["T2-CLS-CMA"])

    assert no_bpa["use_beam_prototype_alignment"] is False
    assert cma["use_amber_cma_analogue"] is True
    assert linear["prototype_target_circular"] is False
    assert classifier["head_type"] == "classifier"
    assert classifier["use_amber_cma_analogue"] is True


def test_multiseed_launcher_changes_only_seed_bound_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}
    seed1 = launcher.build_config("T2", Path("outputs/matrix"), seed=1, **kwargs)
    seed2 = launcher.build_config("T2", Path("outputs/matrix"), seed=2, **kwargs)

    normalized = deepcopy(seed2)
    for path in (("experiment", "seed"), ("data", "domain_balanced_sampling", "seed"), ("temporal_missing", "seed"), ("mmw_all_weather_protocol", "seed")):
        target = normalized
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = 1
    normalized["output"]["run_name"] = "seed1"
    assert normalized == seed1


def test_temporal_matrix_cache_has_unique_masks_for_each_retained_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_script("eval_mmw_all_weather_matrix.py", monkeypatch)
    cache = evaluator._load_or_create_temporal_cache(tmp_path, modality_frame_masks=4)

    for rate in (0.2, 0.4, 0.6, 0.8):
        masks = cache[(rate, 0)]["masks"]
        assert masks
        for mask_type in ("modality_frame", "frame_level", "block"):
            selected = [item for item in masks if item["mask_type"] == mask_type]
            digests = {evaluator._matrix_digest(item["modality_temporal_mask"]) for item in selected}
            assert selected and len(selected) == len(digests)


def test_domain_balanced_sampler_equalizes_mmw_domains() -> None:
    pooled = ConcatDataset([TensorDataset(torch.arange(2)), TensorDataset(torch.arange(8))])
    pooled.domain_inventory = [{"id": "small"}, {"id": "large"}]
    sampler = build_domain_balanced_sampler(pooled, {"enabled": True, "seed": 19, "replacement": True, "num_samples": 20})

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.weights[:2].tolist() == [0.5, 0.5]
    assert sampler.weights[2:].tolist() == [0.125] * 8
