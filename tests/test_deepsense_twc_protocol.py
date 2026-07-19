import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from kd_sensing.config.io import load_config
from kd_sensing.data.deepsense_twc import PROTOCOL_ID, SCENES, load_protocol, prepare_protocol


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("t2", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")


def test_deepsense_secondary_configs_share_fixed_balanced_schedule() -> None:
    for method in METHODS:
        cfg = load_config(ROOT / f"configs/deepsense6g/{method}.yaml")
        assert cfg["data"]["dataset"]["type"] == "deepsense6g"
        assert cfg["temporal_missing"]["schedule_id"] == "deepsense6g_fair_pattern_v1"
        assert cfg["temporal_missing"]["panel_size"] == 600
        assert cfg["training"]["optimizer"]["type"] == "adamw"
    assert load_config(ROOT / "configs/deepsense6g/t2.yaml")["model"]["primary"]["router_use_pattern_features"] is False


def test_deepsense_protocol_freezes_four_scene_csvs_and_masks(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for scene in SCENES:
        root = dataset / f"scenario{scene}"
        root.mkdir(parents=True)
        np.savetxt(root / "power.txt", np.ones(64))
        for name in ("train_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR.csv"):
            with (root / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "future_beam1"])
                writer.writerow([f"{scene}-1", "power.txt"])

    path = prepare_protocol(dataset, tmp_path / "cache")
    protocol = load_protocol(path)

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert len(protocol["scenes"]) == 4
    assert protocol["pooled_dataset"]["scene_ids"] == [31, 32, 33, 34]
    assert protocol["pooled_dataset"]["train_row_count"] == 4
    assert protocol["pooled_dataset"]["test_row_count"] == 4
    assert protocol["fixed_mask_cache"]["condition_count"] > 100

    launcher_path = ROOT / "scripts/launch_deepsense_twc_evidence.py"
    spec = importlib.util.spec_from_file_location(launcher_path.stem, launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(launcher)
    plan_path = launcher.prepare_plan(
        tmp_path / "outputs",
        path,
        methods=launcher.METHODS,
        seeds=launcher.SEEDS,
        batch_size=64,
        epochs=40,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["jobs"]) == 15
    assert {(job["method"], job["seed"]) for job in plan["jobs"]} == {
        (method, seed) for method in launcher.METHODS for seed in launcher.SEEDS
    }
    assert all(job["scope"] == "DeepSense6G-Scene31-34合并" for job in plan["jobs"])
    generated = load_config(Path(plan["jobs"][0]["config_path"]))
    assert [domain["scene"] for domain in generated["data"]["dataset"]["domains"]] == [31, 32, 33, 34]


def test_deepsense_launcher_builds_one_pooled_config_per_method_seed(tmp_path: Path) -> None:
    path = ROOT / "scripts/launch_deepsense_twc_evidence.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    protocol = {
        "manifest_sha256": "protocol-sha",
        "fixed_mask_cache": {"sha256": "mask-sha", "checksum": "mask-checksum"},
        "pooled_dataset": {
            "id": "deepsense6g_scene31_34_pooled_v1",
            "scene_ids": [31, 32, 33, 34],
            "train_row_count": 4,
            "test_row_count": 4,
            "component_inventory_sha256": "inventory-sha",
        },
    }
    scenes = [
        {
            "scene": scene,
            "data_root": f"/data/scenario{scene}",
            "train": {"path": f"/cache/{scene}_train.csv", "sha256": f"train-{scene}"},
            "test": {"path": f"/cache/{scene}_test.csv", "sha256": f"test-{scene}"},
        }
        for scene in SCENES
    ]

    cfg = module.build_config("T2", scenes, protocol, output_root=tmp_path, seed=2, batch_size=64, epochs=40)

    assert [item["scene"] for item in cfg["data"]["dataset"]["domains"]] == [31, 32, 33, 34]
    assert "scene" not in cfg["data"]["dataset"]
    assert cfg["experiment"]["seed"] == 2
    assert cfg["deepsense6g_twc_evidence"]["dataset_scope"] == "deepsense6g_scene31_34_pooled_v1"
    assert cfg["training"]["final_test"]["enabled"] is False


def test_data_factory_pools_deepsense_scenes_into_one_dataset(tmp_path: Path, monkeypatch) -> None:
    from kd_sensing.engine import data_factory

    domains = []
    for scene in SCENES:
        csv_path = tmp_path / f"scene{scene}.csv"
        csv_path.write_text("sample\n1\n", encoding="utf-8")
        domains.append(
            {
                "id": f"scenario{scene}",
                "scene": scene,
                "data_root": str(tmp_path),
                "train_csv_name": str(csv_path),
            }
        )
    cfg = {"data": {"dataset": {"type": "deepsense6g", "domains": domains}}}
    monkeypatch.setattr(data_factory, "build_dataset", lambda _cfg, _split, **_kwargs: TensorDataset(torch.arange(2)))

    pooled = data_factory.build_split_dataset(cfg, "train")

    assert isinstance(pooled, ConcatDataset)
    assert len(pooled.datasets) == 4
    assert len(pooled) == 8
    assert [item["id"] for item in pooled.domain_inventory] == [f"scenario{scene}" for scene in SCENES]


def test_deepsense_summary_requires_fifteen_pooled_checkpoints(tmp_path: Path) -> None:
    path = ROOT / "scripts/summarize_deepsense_twc_evidence.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    eval_root = tmp_path / "eval_fixed"
    for method in module.METHODS:
        for seed in module.SEEDS:
            target = eval_root / method / f"seed{seed}" / "metrics.csv"
            target.parent.mkdir(parents=True)
            row = {
                "method": method,
                "seed": seed,
                "dataset_scope": "deepsense6g_scene31_34_pooled_v1",
                "coverage_status": "complete",
                "eval_family": "whole_modality",
                "pattern": "full",
                "mask_type": "whole_modality",
                "mask_digest": "fixed-mask",
                **{metric: 0.5 for metric in module.METRICS},
            }
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

    result = module.summarize(eval_root, tmp_path / "summary")

    assert result["complete_unit_count"] == 15
    assert result["dataset_scope"] == "deepsense6g_scene31_34_pooled_v1"
    assert (tmp_path / "summary/pooled_method_summary.csv").is_file()
