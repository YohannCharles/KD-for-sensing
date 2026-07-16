import csv
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
    cfg = launcher.build_config(
        "T2",
        Path("outputs/test"),
        smoke=False,
        epochs=40,
        batch_size=32,
        umask_training_profile="umask_h4_v1",
    )

    assert cfg["data"]["dataset"]["type"] == "mmw"
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert cfg["model"]["primary"]["temporal_pooling"] == {"enabled": True, "type": "masked_mean"}
    assert cfg["training"]["optimizer"] == {"type": "adamw"}
    assert cfg["training"]["lr"] == 5.0e-4
    assert cfg["training"]["weight_decay"] == 3.0e-4
    assert cfg["scheduler"] == {"type": "cosine_warm_restarts", "T_0": 40, "T_mult": 1, "eta_min": 1.0e-6}
    profile = cfg["mmw_all_weather_protocol"]["training_profile"]
    assert profile["id"] == "umask_h4_v1"
    assert profile["canonical_values"] == launcher.UMASK_TRAINING_PROFILES["umask_h4_v1"]
    assert profile["sha256"] == launcher._profile_sha256(profile["id"], profile["canonical_values"])


def test_umask_profiles_are_explicit_and_do_not_change_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}

    assert launcher.default_umask_training_profile("T2") == "umask_h4_v1"
    assert launcher.default_umask_training_profile("S1") == "umask_h4_v1"
    assert launcher.default_umask_training_profile("T2-NoBPA") == "legacy_h0_v1"
    assert launcher.default_umask_training_profile("amber_full") is None

    s1 = launcher.build_config("S1", Path("outputs/profile"), umask_training_profile="umask_h4_v1", **kwargs)
    ablation = launcher.build_config(
        "T2-NoBPA",
        Path("outputs/profile"),
        umask_training_profile="legacy_h0_v1",
        **kwargs,
    )
    amber = launcher.build_config("amber_full", Path("outputs/profile"), umask_training_profile=None, **kwargs)

    assert s1["mmw_all_weather_protocol"]["training_profile"]["id"] == "umask_h4_v1"
    assert s1["training"]["optimizer"] == {"type": "adamw"}
    assert ablation["mmw_all_weather_protocol"]["training_profile"]["id"] == "legacy_h0_v1"
    assert ablation["training"]["optimizer"] == {"type": "adam"}
    assert ablation["training"]["weight_decay"] == 1.0e-4
    assert ablation["scheduler"] == {"type": "none"}
    assert "training_profile" not in amber["mmw_all_weather_protocol"]
    assert amber["training"].get("optimizer", {"type": "adam"}) == {"type": "adam"}
    assert amber["training"]["weight_decay"] == 1.0e-4
    assert amber["scheduler"] == {"type": "none"}

    with pytest.raises(ValueError, match="requires an explicit U-Mask training profile"):
        launcher.build_config("T2", Path("outputs/profile"), umask_training_profile=None, **kwargs)
    with pytest.raises(ValueError, match="does not accept a U-Mask training profile"):
        launcher.build_config("amber_full", Path("outputs/profile"), umask_training_profile="umask_h4_v1", **kwargs)
    with pytest.raises(ValueError, match="must use the legacy_h0_v1"):
        launcher.build_config("T2-NoBPA", Path("outputs/profile"), umask_training_profile="umask_h4_v1", **kwargs)
    with pytest.raises(ValueError, match="fixed to a 40-epoch budget"):
        launcher.build_config("T2", Path("outputs/profile"), epochs=39, batch_size=32, smoke=False, umask_training_profile="umask_h4_v1")


def test_t2_bpa_cma_ablation_configs_are_explicit_and_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32, "umask_training_profile": "legacy_h0_v1"}
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
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32, "umask_training_profile": "umask_h4_v1"}
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


def test_all_weather_preflight_checks_future_label_bs_gps_and_derived_radar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    root = tmp_path / "dataset"
    root.mkdir()
    csv_path = root / "split.csv"
    fields = ["camera1", "radar1", "gps1", "bs_gps1", "lidar1", "future_beam_label1"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "camera1": "camera.png",
                "radar1": "radar_RA.npy",
                "gps1": "gps.yaml",
                "bs_gps1": "",
                "lidar1": "lidar.npy",
                "future_beam_label1": 64,
            }
        )
    for name in ("camera.png", "radar_RA.npy", "gps.yaml", "lidar.npy"):
        (root / name).touch()

    report = launcher._inspect_csv(csv_path, root, set(fields))
    failures = "\n".join(report["failures"])

    assert "bs_gps1" in failures
    assert "radar1 (_DA)" in failures
    assert "future_beam_label1" in failures
