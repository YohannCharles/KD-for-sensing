import csv
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset, WeightedRandomSampler

import kd_sensing.engine.data_factory as data_factory
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_domain_balanced_sampler,
    build_split_dataset,
    has_validation_csv,
)
from kd_sensing.engine.run_metadata import dataloaders_run_metadata
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


def _domain_config(tmp_path: Path) -> dict:
    domains = []
    for index, (condition, count) in enumerate((("sunny", 2), ("rainy", 5))):
        root = tmp_path / condition
        root.mkdir(parents=True)
        train = root / "train.csv"
        test = root / "test.csv"
        train.write_text("sample_id\n" + "\n".join(f"train-{item}" for item in range(count)) + "\n", encoding="utf-8")
        test.write_text("sample_id\ntest\n", encoding="utf-8")
        domains.append(
            {
                "id": f"{condition}/scene{index}",
                "condition": condition,
                "scene": f"scene{index}",
                "data_root": str(root),
                "train_csv_name": train.name,
                "test_csv_name": test.name,
            }
        )
    return {
        "experiment": {"seed": 11},
        "data": {"dataset": {"type": "mmw", "domains": domains}, "dataloader": {}},
        "model": {"modalities": ["gps"]},
    }


def test_mmw_domains_build_independent_leaves_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _domain_config(tmp_path)
    assert has_validation_csv(cfg) is False
    calls = []

    def fake_build_dataset(leaf_cfg, split, **_kwargs):
        domain_cfg = leaf_cfg["data"]["dataset"]
        calls.append((domain_cfg["condition"], domain_cfg["scene"], domain_cfg.get("csv_name"), split))
        count = 2 if domain_cfg["condition"] == "sunny" else 5
        return TensorDataset(torch.arange(count))

    monkeypatch.setattr(data_factory, "build_dataset", fake_build_dataset)
    pooled = build_split_dataset(cfg, "train")

    assert isinstance(pooled, ConcatDataset)
    assert len(pooled.datasets) == 2
    assert [item["id"] for item in pooled.domain_inventory] == ["sunny/scene0", "rainy/scene1"]
    assert [item["sample_count"] for item in pooled.domain_inventory] == [2, 5]
    assert calls == [("sunny", "scene0", None, "train"), ("rainy", "scene1", None, "train")]


def test_mmw_domains_require_explicit_validation_csv(tmp_path: Path):
    cfg = _domain_config(tmp_path)

    with pytest.raises(ValueError, match="missing validation CSV field"):
        build_split_dataset(cfg, "validation")

    for domain in cfg["data"]["dataset"]["domains"]:
        validation = Path(domain["data_root"]) / "validation.csv"
        validation.write_text("sample_id\nvalidation\n", encoding="utf-8")
        domain["val_csv_name"] = validation.name

    assert has_validation_csv(cfg) is True


def test_mmw_domains_reject_duplicate_id_and_missing_split(tmp_path: Path):
    cfg = _domain_config(tmp_path)
    cfg["data"]["dataset"]["domains"][1]["id"] = cfg["data"]["dataset"]["domains"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate MMW domain id"):
        build_split_dataset(cfg, "train")

    cfg = _domain_config(tmp_path / "other")
    cfg["data"]["dataset"]["domains"][0]["train_csv_name"] = "missing.csv"
    with pytest.raises(FileNotFoundError, match="sunny/scene0"):
        build_split_dataset(cfg, "train")


def test_all_weather_launcher_world_local_configs_are_matched(monkeypatch: pytest.MonkeyPatch):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}
    world = launcher.build_config("T2", Path("outputs/matched"), gps_feature_mode="relative_polar", **kwargs)
    local = launcher.build_config("T2", Path("outputs/matched"), gps_feature_mode="rsu_local_relative_polar", **kwargs)

    assert world["data"]["dataset"]["sample_cache"] == {"enabled": False}
    assert local["data"]["dataset"]["input_profiles"]["gps"] == "rsu_local_relative_polar_history"
    assert local["mmw_all_weather_protocol"]["gps_yaw_source"] == "bs_yaml:sensors.rsu_pose.rotation.yaw"
    normalized_local = deepcopy(local)
    normalized_local["data"]["dataset"]["gps_feature_mode"] = "relative_polar"
    normalized_local["data"]["dataset"]["input_profiles"]["gps"] = "relative_polar_history"
    normalized_local["mmw_all_weather_protocol"]["gps_feature_mode"] = "relative_polar"
    normalized_local["mmw_all_weather_protocol"]["gps_angle_frame"] = "world"
    normalized_local["mmw_all_weather_protocol"]["gps_yaw_source"] = None
    assert normalized_local == world


def test_all_weather_launcher_multiseed_changes_only_behavior_seeds(monkeypatch: pytest.MonkeyPatch):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32, "gps_feature_mode": "relative_polar"}
    seed1 = launcher.build_config("T2", Path("outputs/matched"), seed=1, **kwargs)
    seed2 = launcher.build_config("T2", Path("outputs/matched"), seed=2, **kwargs)

    assert seed1["data"]["dataset"]["domains"] == seed2["data"]["dataset"]["domains"]
    assert seed1["data"]["dataset"].get("portion_seed") == seed2["data"]["dataset"].get("portion_seed")
    assert seed1["experiment"]["seed"] == seed1["data"]["domain_balanced_sampling"]["seed"] == 1
    assert seed2["experiment"]["seed"] == seed2["data"]["domain_balanced_sampling"]["seed"] == 2
    assert seed1["temporal_missing"]["seed"] == 1
    assert seed2["temporal_missing"]["seed"] == 2
    assert seed2["output"]["run_name"] == "seed2"
    assert seed2["mmw_all_weather_protocol"]["seed"] == 2
    normalized = deepcopy(seed2)
    normalized["experiment"]["seed"] = 1
    normalized["data"]["domain_balanced_sampling"]["seed"] = 1
    normalized["temporal_missing"]["seed"] = 1
    normalized["output"]["run_name"] = "seed1"
    normalized["mmw_all_weather_protocol"]["seed"] = 1
    assert normalized == seed1


def test_all_weather_launcher_t2_ablation_methods_are_explicit_without_changing_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)

    assert launcher.METHODS == ("S1", "T2", "amber_full", "rmbp_mm")
    assert launcher.T2_ABLATION_METHODS == (
        "T2-NoBPA",
        "T2-BPA2CMA",
        "T2-Linear",
        "T2-CLS",
        "T2-CLS-CMA",
    )
    assert all(not path.startswith("outputs/") for path in launcher.METHOD_BASES.values())
    for method in launcher.T2_ABLATION_METHODS:
        assert launcher.METHOD_BASES[method] == launcher.METHOD_BASES["T2"]
        cfg = launcher.build_config(
            method,
            Path("outputs/mmw_t2_ablation"),
            seed=3,
            smoke=False,
            epochs=40,
            batch_size=32,
        )
        assert cfg["experiment"]["name"] == method
        assert cfg["experiment"]["ablation_id"] == method
        assert cfg["output"] == {
            "dir": f"outputs/mmw_t2_ablation/{method}",
            "run_name": "seed3",
            "group_by_scene": False,
            "overwrite": False,
            "progress": {"enabled": False},
            "tensorboard": {"enabled": False},
        }
        provenance = cfg["mmw_all_weather_protocol"]["t2_ablation"]
        assert provenance["protocol"] == "mmw_t2_bpa_cma_ablation_v1"
        assert provenance["paper_equivalent"] is False
        assert provenance["cma_scope"] == "pooled_feature_objective_analogue_not_full_amber_class_former"


def test_all_weather_launcher_t2_objective_ablation_pairs_are_strictly_matched(
    monkeypatch: pytest.MonkeyPatch,
):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}
    configs = {
        method: launcher.build_config(method, Path("outputs/mmw_t2_ablation"), **kwargs)
        for method in ("T2-NoBPA", "T2-BPA2CMA", "T2-CLS", "T2-CLS-CMA")
    }

    for method in ("T2-NoBPA", "T2-BPA2CMA"):
        cfg = configs[method]
        assert cfg["model"]["primary"]["head_type"] == "prototype"
        assert cfg["model"]["primary"]["router_use_prototype_margin"] is True
        assert cfg["training"]["use_beam_prototype_alignment"] is False
        assert cfg["training"]["beam_proto_align_weight"] == 0.0
        assert cfg["training"]["lambda_proto"] == 0.0
        assert cfg["training"]["use_modality_prototype_loss"] is False
        assert cfg["training"]["lambda_modality_proto"] == 0.0

    for method in ("T2-CLS", "T2-CLS-CMA"):
        cfg = configs[method]
        assert cfg["model"]["primary"]["head_type"] == "classifier"
        assert cfg["model"]["primary"]["use_beam_prototype_alignment"] is False
        assert cfg["model"]["primary"]["router_use_prototype_margin"] is False
        assert cfg["training"]["router_use_prototype_margin"] is False
        assert cfg["loss"]["router_use_prototype_margin"] is False
        assert cfg["loss"]["pcpg_radar_balance"]["router_use_prototype_margin"] is False
        assert cfg["training"]["use_beam_prototype_alignment"] is False
        assert cfg["training"]["beam_proto_align_weight"] == 0.0
        assert cfg["training"]["use_modality_prototype_loss"] is False

    for disabled, enabled in (("T2-NoBPA", "T2-BPA2CMA"), ("T2-CLS", "T2-CLS-CMA")):
        control = configs[disabled]
        candidate = deepcopy(configs[enabled])
        candidate["experiment"]["name"] = disabled
        candidate["experiment"]["ablation_id"] = disabled
        candidate["output"]["dir"] = control["output"]["dir"]
        candidate["mmw_all_weather_protocol"]["t2_ablation"] = deepcopy(
            control["mmw_all_weather_protocol"]["t2_ablation"]
        )
        for section in (candidate["training"], candidate["loss"]["u_mask_beam_jepa"]):
            section["use_amber_cma_analogue"] = False
            section["lambda_amber_cma"] = 0.0
        assert candidate == control

        assert configs[enabled]["training"]["use_amber_cma_analogue"] is True
        assert configs[enabled]["training"]["lambda_amber_cma"] == 0.2
        assert configs[enabled]["training"]["amber_cma_temperature"] == 0.2
        assert configs[enabled]["loss"]["u_mask_beam_jepa"]["use_amber_cma_analogue"] is True
        assert configs[enabled]["mmw_all_weather_protocol"]["t2_ablation"]["cma_weight"] == 0.2
        resolved = u_mask_beam_jepa_config(configs[enabled])
        assert resolved["use_beam_prototype_alignment"] is False
        assert resolved["use_amber_cma_analogue"] is True


def test_all_weather_launcher_t2_linear_changes_only_prototype_target_wrap(
    monkeypatch: pytest.MonkeyPatch,
):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    kwargs = {"smoke": False, "epochs": 40, "batch_size": 32}
    full = launcher.build_config("T2", Path("outputs/mmw_t2_ablation"), **kwargs)
    linear = launcher.build_config("T2-Linear", Path("outputs/mmw_t2_ablation"), **kwargs)

    assert linear["model"]["primary"] == full["model"]["primary"]
    assert linear["training"]["prototype_target_circular"] is False
    assert linear["training"]["beam_label_circular"] is True
    assert linear["training"]["use_circular_soft_targets"] is True
    assert linear["training"]["circular_beam_distance"] is True
    assert linear["loss"]["u_mask_beam_jepa"]["circular_beam_distance"] is True
    assert linear["loss"]["pcpg_radar_balance"]["circular_beam_distance"] is True
    assert linear["evaluation"]["beam_distance_circular"] is True
    assert linear["evaluation"]["circular_beam_distance"] is True
    assert linear["evaluation"]["dba_distance_mode"] == "circular"
    resolved = u_mask_beam_jepa_config(linear)
    assert resolved["use_beam_prototype_alignment"] is True
    assert resolved["prototype_target_circular"] is False
    assert resolved["beam_label_circular"] is True
    assert resolved["circular_beam_distance"] is True
    provenance = linear["mmw_all_weather_protocol"]["t2_ablation"]
    assert provenance["prototype_target_geometry"] == "linear"
    assert provenance["router_oracle_geometry"] == "circular"
    assert provenance["evaluation_geometry"] == "circular"
    assert provenance["intervention"] == "remove_prototype_target_wrap_prior_only"
    assert linear["training"]["superset_consistency"] == full["training"]["superset_consistency"]


def test_all_weather_launcher_builds_unique_six_job_matrix_and_rejects_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    jobs = launcher.build_job_matrix(
        ("T2", "amber_full", "rmbp_mm"),
        (2, 3),
        (0, 1, 2, 3, 4, 5),
        tmp_path,
    )

    assert [(job["method"], job["seed"], job["gpu"]) for job in jobs] == [
        ("T2", 2, 0),
        ("T2", 3, 1),
        ("amber_full", 2, 2),
        ("amber_full", 3, 3),
        ("rmbp_mm", 2, 4),
        ("rmbp_mm", 3, 5),
    ]
    assert len({job["config_path"] for job in jobs}) == 6
    assert len({job["run_dir"] for job in jobs}) == 6

    with pytest.raises(ValueError, match="unique"):
        launcher.build_job_matrix(("T2",), (2, 3), (0, 0), tmp_path)
    with pytest.raises(ValueError, match="Expected 2 GPUs"):
        launcher.build_job_matrix(("T2",), (2, 3), (0,), tmp_path)

    manifest = tmp_path / launcher._manifest_name((2, 3))
    launcher.validate_job_targets(jobs, manifest)
    jobs[0]["run_dir"].mkdir(parents=True)
    with pytest.raises(FileExistsError, match="seed2"):
        launcher.validate_job_targets(jobs, manifest)


def test_all_weather_evaluator_checks_checkpoint_gps_provenance_before_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluator = _load_script("eval_mmw_all_weather_matrix.py", monkeypatch)
    root = tmp_path / "run"
    config = root / "generated_configs" / "T2_seed1.yaml"
    checkpoint = root / "T2" / "seed1" / "checkpoints" / "last.pth"
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    checkpoint.touch()
    monkeypatch.setattr(evaluator, "load_checkpoint_metadata", lambda _path: {"gps_feature_mode": "relative_polar"})

    def reject(_cfg, _metadata):
        raise RuntimeError("gps-provenance-guard-called")

    monkeypatch.setattr(evaluator, "validate_evaluation_gps_checkpoint_provenance", reject)

    with pytest.raises(RuntimeError, match="gps-provenance-guard-called"):
        evaluator.evaluate_method("T2", root, tmp_path / "eval", {}, None)


def test_all_weather_evaluator_isolates_seed_artifacts_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluator = _load_script("eval_mmw_all_weather_matrix.py", monkeypatch)

    config, checkpoint = evaluator._seed_artifact_paths(tmp_path / "run", "T2", 3)
    target = evaluator._seed_evaluation_target(tmp_path / "eval", "T2", 3, seed_subdir=True)
    legacy = evaluator._seed_evaluation_target(tmp_path / "eval", "T2", 1, seed_subdir=False)

    assert config == tmp_path / "run/generated_configs/T2_seed3.yaml"
    assert checkpoint == tmp_path / "run/T2/seed3/checkpoints/last.pth"
    assert target == tmp_path / "eval/T2/seed3/metrics.csv"
    assert legacy == tmp_path / "eval/T2/metrics.csv"


@pytest.mark.parametrize(
    ("bad_reference", "bad_contents", "expected"),
    [
        ("", None, "reference is empty"),
        ("-99", None, "missing-value sentinel is not allowed"),
        ("missing.yaml", None, "file does not exist"),
        (
            "not_yaml.txt",
            "sensors:\n  rsu_pose:\n    rotation:\n      yaw: 45.0\n",
            "expected a YAML file",
        ),
        (
            "nonfinite.yaml",
            "sensors:\n  rsu_pose:\n    rotation:\n      yaw: .nan\n",
            "non-finite",
        ),
    ],
)
def test_all_weather_local_preflight_rejects_four_of_five_valid_bs_gps_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_reference: str,
    bad_contents: str | None,
    expected: str,
):
    launcher = _load_script("launch_mmw_all_weather_matrix.py", monkeypatch)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    columns = [f"bs_gps{index}" for index in range(1, 6)]
    row = {}
    for index, column in enumerate(columns[:4], start=1):
        path = tmp_path / f"bs_{index}.yaml"
        yaw = 46.0 if index == 2 else 45.0
        path.write_text(
            f"sensors:\n  rsu_pose:\n    rotation:\n      yaw: {yaw}\n",
            encoding="utf-8",
        )
        row[column] = path.name
    row["bs_gps5"] = bad_reference
    if bad_contents is not None:
        (tmp_path / bad_reference).write_text(bad_contents, encoding="utf-8")
    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    report = launcher._inspect_csv(
        csv_path,
        tmp_path,
        set(columns),
        validate_rsu_yaw=True,
        yaw_cache={},
    )

    failure = "\n".join(report["failures"])
    expected_path = (
        bad_reference
        if bad_reference in {"", "-99"}
        else str(tmp_path / bad_reference)
    )
    assert "row 2 column bs_gps5" in failure
    assert f"path '{expected_path or '<empty>'}'" in failure
    assert expected in failure
    assert "inconsistent" not in failure
    assert report["rsu_yaw_degrees"] == []


def test_domain_balanced_sampler_is_equal_weight_and_reproducible():
    pooled = ConcatDataset([TensorDataset(torch.arange(2)), TensorDataset(torch.arange(8))])
    pooled.domain_inventory = [{"id": "small"}, {"id": "large"}]
    config = {"enabled": True, "seed": 19, "replacement": True, "num_samples": 20}

    first = build_domain_balanced_sampler(pooled, config)
    second = build_domain_balanced_sampler(pooled, config)

    assert isinstance(first, WeightedRandomSampler)
    assert first.weights[:2].tolist() == [0.5, 0.5]
    assert first.weights[2:].tolist() == [0.125] * 8
    assert list(first) == list(second)
    assert first.domain_balanced_metadata["domains"][0]["total_weight"] == 1.0
    assert first.domain_balanced_metadata["domains"][1]["total_weight"] == 1.0


def test_domain_balanced_sampler_applies_only_to_train_and_records_metadata():
    pooled = ConcatDataset([TensorDataset(torch.arange(2)), TensorDataset(torch.arange(3))])
    pooled.domain_inventory = [{"id": "a"}, {"id": "b"}]
    config = {"enabled": True, "seed": 5}

    train = build_dataloader(
        pooled,
        {"batch_size": 2, "num_workers": 0},
        split="train",
        experiment_seed=5,
        domain_balanced_sampling_cfg=config,
    )
    validation = build_dataloader(
        pooled,
        {"batch_size": 2, "num_workers": 0},
        split="validation",
        experiment_seed=5,
        domain_balanced_sampling_cfg=config,
    )

    assert isinstance(train.sampler, WeightedRandomSampler)
    assert not isinstance(validation.sampler, WeightedRandomSampler)
    metadata = dataloaders_run_metadata({"train": train, "validation": validation})
    assert metadata["train"]["domain_balanced_sampling"]["seed"] == 5
    assert "domain_balanced_sampling" not in metadata["validation"]


def test_mmw_temporal_v2_cache_covers_unique_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_script("eval_mmw_all_weather_matrix.py", monkeypatch)

    cache = evaluator._load_or_create_temporal_cache(tmp_path, modality_frame_masks=4)

    assert len(cache[(0.0, 0)]["masks"]) == 1
    expected_counts = {
        0.2: {"modality_frame": 4, "frame_level": 5, "block": 5},
        0.4: {"modality_frame": 4, "frame_level": 10, "block": 4},
        0.6: {"modality_frame": 4, "frame_level": 10, "block": 3},
        0.8: {"modality_frame": 4, "frame_level": 5, "block": 2},
    }
    for rate, expected in expected_counts.items():
        payload = cache[(rate, 0)]
        for mask_type, count in expected.items():
            masks = [item for item in payload["masks"] if item["mask_type"] == mask_type]
            digests = {evaluator._matrix_digest(item["modality_temporal_mask"]) for item in masks}
            assert len(masks) == count
            assert len(digests) == count
        evaluator._validate_temporal_cache_payload(payload, rate=rate, modality_frame_masks=4)

    with pytest.raises(ValueError, match="cache contract mismatch"):
        evaluator._load_or_create_temporal_cache(tmp_path, modality_frame_masks=5)


def test_mmw_extreme_temporal_rates_keep_exactly_three_two_one_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evaluator = _load_script("eval_mmw_all_weather_matrix.py", monkeypatch)
    rates = (0.85, 0.9, 0.95)

    cache = evaluator._load_or_create_temporal_cache(
        tmp_path,
        modality_frame_masks=4,
        rates=rates,
        mask_types=("modality_frame",),
    )

    for rate, remaining in ((0.85, 3), (0.9, 2), (0.95, 1)):
        payload = cache[(rate, 0)]
        assert len(payload["masks"]) == 4
        assert payload["mask_types"] == ["modality_frame"]
        for item in payload["masks"]:
            mask = torch.as_tensor(item["modality_temporal_mask"], dtype=torch.bool)
            assert int(mask.sum().item()) == remaining
            assert float((~mask).float().mean().item()) == pytest.approx(rate)

    with pytest.raises(ValueError, match="use modality_frame only"):
        evaluator._load_or_create_temporal_cache(
            tmp_path / "invalid",
            modality_frame_masks=4,
            rates=(0.85,),
            mask_types=("frame_level",),
        )


def test_mmw_temporal_summary_reports_mask_variance_and_terminal_strata(monkeypatch: pytest.MonkeyPatch):
    summary = _load_script("summarize_mmw_all_weather_matrix.py", monkeypatch)
    rows = []
    masks = (
        ("modality_frame", "a", True, 0.60),
        ("frame_level", "b", True, 0.50),
        ("block", "c", False, 0.20),
    )
    for mask_type, digest, terminal, top1 in masks:
        for domain_index in range(15):
            rows.append(
                {
                    "method": "rmbp_mm",
                    "domain_id": f"domain-{domain_index}",
                    "eval_family": "temporal_missing",
                    "missing_rate": "0.4",
                    "mask_type": mask_type,
                    "mask_digest": digest,
                    "sample_count": "10",
                    "observed_missing_rate": "0.4",
                    "last_frame_available": str(terminal),
                    "last_frame_available_modalities": "4" if terminal else "0",
                    "trailing_fully_missing_frames": "0" if terminal else "2",
                    "reproduction_scope": "rmbp_mm_channel_attention_local",
                    "paper_equivalent": "False",
                    "temporal_result_scope": "out_of_paper_scope_diagnostic",
                    "top1": str(top1),
                    "top3": str(top1),
                    "top5": str(top1),
                    "within_3": str(top1),
                    "adba": str(top1),
                    "mae": str(1.0 - top1),
                    "gate_entropy": "0.1",
                }
            )

    result = summary._temporal_rate_summary(rows)

    combined = next(item for item in result if item["mask_type"] == "type_equal_all")
    missing = next(item for item in result if item["mask_type"] == "terminal_missing")
    modality_frame = next(item for item in result if item["mask_type"] == "modality_frame")
    assert combined["top1"] == pytest.approx((0.6 + 0.5 + 0.2) / 3)
    assert combined["claim_status"] == "diagnostic_only_not_paper_equivalent"
    assert missing["top1"] == pytest.approx(0.2)
    assert missing["domain_count_min"] == 15
    assert modality_frame["top1_std_across_masks"] == pytest.approx(0.0)


def test_mmw_coordinate_pair_summary_reports_paired_and_gps_branch_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    summary = _load_script("summarize_mmw_all_weather_matrix.py", monkeypatch)
    world_dir = tmp_path / "world"
    local_dir = tmp_path / "local"
    output_dir = tmp_path / "summary"
    base = {
        "method": "T2",
        "domain_id": "sunny/scene",
        "condition": "sunny",
        "scene": "scene",
        "sample_count": "10",
        "sample_csv_sha256": "samples",
        "mask_cache_checksum": "cache",
        "available_modalities": "image,radar,lidar,gps",
        "top3": "0.5",
        "top5": "0.6",
        "within_3": "0.7",
        "adba": "0.8",
        "mae": "2.0",
        "gate_entropy": "0.4",
    }
    patterns = (
        ("whole_modality", "full", "0.0", "full", 0.40, 0.25),
        ("whole_modality", "available_gps", "0.0", "gps", 0.30, 1.00),
        ("whole_modality", "available_image_radar_lidar", "0.0", "no-gps", 0.35, 0.00),
        ("temporal_missing", "modality_frame", "0.2", "temporal", 0.32, 0.20),
    )
    world = []
    local = []
    for family, pattern, rate, digest, top1, gate in patterns:
        row = {
            **base,
            "eval_family": family,
            "pattern": pattern,
            "missing_rate": rate,
            "mask_type": pattern,
            "mask_digest": digest,
            "top1": str(top1),
            "mean_gate_gps": str(gate),
            "observed_missing_rate": rate,
            "last_frame_available": "true",
            "last_frame_available_modalities": "4",
            "trailing_fully_missing_frames": "0",
            "reproduction_scope": "project_mainline",
            "paper_equivalent": "false",
            "temporal_result_scope": "mainline_local_validation",
        }
        world.append(row)
        local.append({**row, "top1": str(top1 + 0.1), "mean_gate_gps": str(min(gate + 0.05, 1.0))})
    (world_dir / "T2").mkdir(parents=True)
    (local_dir / "T2").mkdir(parents=True)
    summary._write_csv(world_dir / "T2" / "metrics.csv", world)
    summary._write_csv(local_dir / "T2" / "metrics.csv", local)

    assert summary._write_coordinate_pair_summary(world_dir, local_dir, output_dir) == 0

    paired = summary._read_csv(output_dir / "coordinate_paired_deltas.csv")
    branches = {row["branch"]: row for row in summary._read_csv(output_dir / "gps_branch_summary.csv")}
    temporal = summary._read_csv(output_dir / "coordinate_temporal_deltas.csv")
    assert len(paired) == 4
    assert float(branches["full"]["delta_top1"]) == pytest.approx(0.1)
    assert float(branches["gps_only"]["delta_mean_gate_gps"]) == pytest.approx(0.0)
    assert float(next(row for row in temporal if row["mask_type"] == "modality_frame")["delta_top1"]) == pytest.approx(0.1)


def test_mmw_coordinate_pair_summary_rejects_unpaired_masks(monkeypatch: pytest.MonkeyPatch):
    summary = _load_script("summarize_mmw_all_weather_matrix.py", monkeypatch)
    world = [{"domain_id": "a", "mask_digest": "world"}]
    local = [{"domain_id": "a", "mask_digest": "local"}]

    with pytest.raises(ValueError, match="not one-to-one paired"):
        summary._validate_coordinate_pair_rows(world, local)
