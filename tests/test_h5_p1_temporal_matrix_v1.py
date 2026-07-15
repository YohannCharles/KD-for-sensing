import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kd_sensing.data.temporal_missing import (
    apply_modality_temporal_mask_to_batch,
    generate_fixed_eval_mask_cache,
    sample_stratified_modality_temporal_mask,
)
from kd_sensing.engine.pcpg_radar_balance import pcpg_radar_balance_config
from kd_sensing.engine.data_factory_groups import audit_temporal_split_identities
from kd_sensing.engine.data_factory_protocols import stratified_2604_split_cfg
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.registries import REPRESENTATION_CORES, import_default_components


ROOT = Path(__file__).resolve().parents[1]


class _TemporalIdentityDataset:
    def __init__(self, root: Path, csv_path: Path) -> None:
        self.data_root = root
        self.root_csv = csv_path
        self.scene_id = 31
        self.seq_len = 2
        self.gps_source_seq_len = 2
        self.num_pred = 1
        self.samples = SimpleNamespace(
            input_beam_paths=[[f"h{index}_0.txt", f"h{index}_1.txt"] for index in range(3)],
            future_beam_paths=[[f"t{index}.txt"] for index in range(3)],
        )

    def __len__(self) -> int:
        return len(self.samples.input_beam_paths)

    def _target_beam_paths(self, _input_paths, future_paths):
        return list(future_paths[: self.num_pred])


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stratified_modality_temporal_sampler_shape_drop_and_fallback() -> None:
    item = sample_stratified_modality_temporal_mask(
        history_window=5,
        fixed_drop_modalities=["image", "radar", "lidar"],
        fixed_rate=0.8,
        fixed_mask_type="modality_frame",
    )
    mask = item["modality_temporal_mask"]
    assert mask.shape == (5, 4)
    assert item["drop_count"] == 3
    assert item["dropped_modalities"] == ["image", "radar", "lidar"]
    assert bool(mask.any())
    assert not bool(mask[:, :3].any())
    assert bool(mask[:, 3].any())


def test_fixed_eval_mask_cache_is_reused_and_balanced(tmp_path: Path) -> None:
    first = generate_fixed_eval_mask_cache(tmp_path, rates=(0.2,), drop_counts=(1, 2, 3), num_masks_per_cell=16, seed=20260708)
    second = generate_fixed_eval_mask_cache(tmp_path, rates=(0.2,), drop_counts=(1, 2, 3), num_masks_per_cell=16, seed=999)
    assert first[(0.2, 1)] == second[(0.2, 1)]
    payload = first[(0.2, 2)]
    assert payload["checksum"]
    combos = [tuple(item["dropped_modalities"]) for item in payload["masks"]]
    assert len(set(combos)) == 6
    counts = {combo: combos.count(combo) for combo in set(combos)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_apply_modality_temporal_mask_zero_fills_batch() -> None:
    batch = {
        "image": torch.ones(2, 5, 1),
        "radar_ra": torch.ones(2, 5, 1) * 2,
        "radar_da": torch.ones(2, 5, 1) * 3,
        "lidar": torch.ones(2, 5, 1) * 4,
        "gps": torch.ones(2, 5, 1) * 5,
    }
    mask = torch.ones(5, 4, dtype=torch.bool)
    mask[0, 0] = False
    mask[:, 3] = False
    out = apply_modality_temporal_mask_to_batch(batch, mask)
    assert out["modality_temporal_mask"].shape == (2, 5, 4)
    assert out["image"][:, 0].abs().sum().item() == 0.0
    assert out["gps"].abs().sum().item() == 0.0
    assert out["radar_ra"].abs().sum().item() > 0.0


def test_h5_p1_launcher_dry_run_writes_manifest(tmp_path: Path) -> None:
    launcher = _load_script("launch_h5_p1_temporal_models_v1.py")
    code = launcher.main([
        "--output_root",
        str(tmp_path),
        "--seeds",
        "1",
        "--methods",
        "ours_c2_main,ours_b4_nonrouter_soft_jepa,ours_e5_low_lr_pcpg,amber_full,rmbp_mm",
        "--gpus",
        "0,1,2,3,4,5,6,7",
        "--max_jobs",
        "8",
        "--per_gpu",
        "1",
        "--dry_run",
    ])
    assert code == 0
    rows = _read_csv(tmp_path / "job_manifest.csv")
    assert [row["method"] for row in rows] == [
        "ours_c2_main",
        "ours_b4_nonrouter_soft_jepa",
        "ours_e5_low_lr_pcpg",
        "amber_full",
        "rmbp_mm",
    ]
    assert {row["history_window"] for row in rows} == {"5"}
    assert {row["prediction_window"] for row in rows} == {"1"}
    assert {row["torch_num_threads"] for row in rows} == {"1"}
    assert {row["persistent_workers"] for row in rows} == {"False"}
    assert max(int(row["gpu"]) for row in rows if row["gpu"]) <= 7
    assert next(row for row in rows if row["method"] == "rmbp_mm")["status"] == "planned"
    assert "rmbp_mm" in {row["method"] for row in rows}
    amber_cfg = launcher.yaml.safe_load((tmp_path / "generated_configs" / "amber_full_seed1.yaml").read_text())
    amber_encoders = amber_cfg["model"]["primary"]["encoders"]
    assert amber_encoders["image"]["freeze_backbone"] is False
    assert amber_encoders["radar"]["freeze_backbone"] is False
    assert amber_encoders["lidar"]["freeze_backbone"] is False
    rmbp_cfg = launcher.yaml.safe_load((tmp_path / "generated_configs" / "rmbp_mm_seed1.yaml").read_text())
    assert rmbp_cfg["model"]["primary"]["encoders"]["image"]["freeze_backbone"] is False
    ours_cfg = launcher.yaml.safe_load((tmp_path / "generated_configs" / "ours_c2_main_seed1.yaml").read_text())
    split_keys = (
        "scenes",
        "train_scenes",
        "validation_scenes",
        "test_scenes",
        "split_protocol",
        "split_strategy",
        "split_group_identity_policy",
        "split_seed",
        "split_source_splits",
        "split_fractions",
    )
    expected_split = {key: ours_cfg["data"]["dataset"][key] for key in split_keys}
    assert expected_split["scenes"] == [31, 32, 33, 34]
    assert expected_split["split_strategy"] == "stratified_by_target_beam_per_scene_sequence_group"
    assert expected_split["split_group_identity_policy"] == "scene_id:seq_index"
    assert {key: amber_cfg["data"]["dataset"][key] for key in split_keys} == expected_split
    assert {key: rmbp_cfg["data"]["dataset"][key] for key in split_keys} == expected_split


def test_temporal_sample_level_split_is_rejected() -> None:
    cfg = {
        "data": {
            "dataset": {
                "split_protocol": "stratified_80_10_10",
                "split_strategy": "stratified_by_target_beam_per_scene",
                "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "seq_len": 5,
            }
        }
    }

    with pytest.raises(ValueError, match="group-safe sequence strategy"):
        stratified_2604_split_cfg(cfg)


def test_temporal_split_identity_audit_records_digests_and_rejects_frame_overlap(tmp_path: Path) -> None:
    csv_path = tmp_path / "sequences.csv"
    csv_path.write_text("seq_index\nseq0\nseq1\nseq2\n", encoding="utf-8")
    dataset = _TemporalIdentityDataset(tmp_path, csv_path)
    splits = {"train": [0], "validation": [1], "test": [2]}

    audit = audit_temporal_split_identities(dataset, splits, max_conflict_examples=2)

    assert audit["status"] == "passed"
    assert audit["roles"]["train"]["sample_count"] == 1
    assert audit["roles"]["validation"]["identities"]["history_frame"]["count"] == 2
    assert len(audit["roles"]["test"]["identities"]["referenced_frame"]["digest"]) == 64
    assert json.loads(json.dumps(audit, sort_keys=True))["status"] == "passed"
    assert all(
        count == 0
        for pair in audit["pairwise"]
        for count in pair["overlap_counts"].values()
    )

    dataset.samples.input_beam_paths[1][0] = dataset.samples.input_beam_paths[0][0]
    with pytest.raises(ValueError, match=r"train/validation history_frame count=1"):
        audit_temporal_split_identities(dataset, splits, max_conflict_examples=2)

    dataset.samples.input_beam_paths[1][0] = "h1_0.txt"
    dataset.samples.future_beam_paths[0][0] = dataset.samples.input_beam_paths[1][0]
    with pytest.raises(ValueError, match=r"train/validation referenced_frame count=1"):
        audit_temporal_split_identities(dataset, splits, max_conflict_examples=2)


def test_s1_lightweight_launcher_maps_seed1_to_gpu0_7(tmp_path: Path) -> None:
    launcher = _load_script("launch_h5_p1_temporal_models_v1.py")
    assert launcher.main([
        "--profile",
        "s1_lightweight",
        "--output_root",
        str(tmp_path),
        "--gpus",
        "0,1,2,3,4,5,6,7",
        "--max_jobs",
        "8",
        "--per_gpu",
        "1",
        "--max_epochs",
        "1",
        "--dry_run",
    ]) == 0

    rows = _read_csv(tmp_path / "job_manifest.csv")
    methods = ["S1", "T2", "T1", "A1", "A2", "A3", "T1+T2", "J1"]
    assert [row["method"] for row in rows] == methods
    assert [row["gpu"] for row in rows] == [str(index) for index in range(8)]
    assert {row["profile"] for row in rows} == {"s1_lightweight"}
    assert {row["max_jobs"] for row in rows} == {"8"}
    assert {row["per_gpu"] for row in rows} == {"1"}
    assert {row["torch_num_threads"] for row in rows} == {"12"}
    assert {row["persistent_workers"] for row in rows} == {"True"}
    assert len({row["output_dir"] for row in rows}) == 8
    assert len({row["log_path"] for row in rows}) == 8

    configs = {
        method: launcher.yaml.safe_load((tmp_path / "generated_configs" / f"{method}_seed1.yaml").read_text())
        for method in methods
    }
    split_keys = (
        "scenes",
        "train_scenes",
        "validation_scenes",
        "test_scenes",
        "split_protocol",
        "split_strategy",
        "split_group_identity_policy",
        "split_seed",
        "split_source_splits",
        "split_fractions",
    )
    shared_split = {key: configs["S1"]["data"]["dataset"][key] for key in split_keys}
    shared_optimizer = configs["S1"]["training"].get("optimizer")
    for method, cfg in configs.items():
        primary = cfg["model"]["primary"]
        assert primary["fusion_type"] == "supervised_router"
        assert primary["consume_missing_modality_metadata"] is True
        assert "temporal_router_type" not in primary
        assert {key: cfg["data"]["dataset"][key] for key in split_keys} == shared_split
        assert cfg["training"].get("optimizer") == shared_optimizer
        assert cfg["training"]["cpu_threads"] == {"enabled": True, "intra_op": 12, "inter_op": 1}
        assert cfg["training"]["max_epochs"] == 1
        assert cfg["data"]["dataloader"]["persistent_workers"] is True
        assert cfg["data"]["dataloader"]["train_persistent_workers"] is True
        assert cfg["data"]["dataloader"]["test_persistent_workers"] is True
        assert cfg["temporal_missing"]["mask_sampler"] == "stratified_modality_temporal"
        assert cfg["loss"]["u_mask_beam_jepa"]["enabled"] is True
        assert cfg["loss"]["u_mask_beam_jepa"]["superset_consistency"] == cfg["training"]["superset_consistency"]
        assert cfg["output"]["dir"] == str(tmp_path / method)
        assert cfg["output"]["run_name"] == "seed1"

    assert configs["S1"]["model"]["primary"]["temporal_pooling"]["type"] == "masked_mean"
    assert configs["A1"]["model"]["primary"]["use_mask_statistics"] is True
    assert configs["A2"]["model"]["primary"]["temporal_pooling"]["type"] == "fixed_recency"
    assert configs["A3"]["model"]["primary"]["temporal_pooling"]["type"] == "gap_aware_residual"
    assert configs["J1"]["model"]["primary"]["temporal_pooling"]["type"] == "gap_aware_residual"
    for method, kl_enabled, rank_enabled in (
        ("T2", True, False),
        ("T1", False, True),
        ("T1+T2", True, True),
        ("J1", True, True),
    ):
        superset = configs[method]["training"]["superset_consistency"]
        assert superset["confidence_gated_kl"] is kl_enabled
        assert superset["beam_monotonic_rank"] is rank_enabled
        assert superset["feature_l2_weight"] == 0.0
        assert superset["rank_tolerance"] == 0.0
        assert configs[method]["temporal_missing"]["preserve_unmasked_for_superset"] is True


def test_t2_geometry_and_classifier_candidates_are_explicit_only(tmp_path: Path) -> None:
    launcher = _load_script("launch_h5_p1_temporal_models_v1.py")
    methods = ["S1-LG", "T2-LG", "S1-CLS", "T2-CLS"]
    assert launcher.main([
        "--profile",
        "s1_lightweight",
        "--methods",
        ",".join(methods),
        "--seeds",
        "1",
        "--gpus",
        "4,5,6,7",
        "--output_root",
        str(tmp_path),
        "--dry_run",
    ]) == 0

    rows = _read_csv(tmp_path / "job_manifest.csv")
    assert [row["method"] for row in rows] == methods
    assert [row["gpu"] for row in rows] == ["4", "5", "6", "7"]
    assert launcher.S1_LIGHTWEIGHT_METHODS == "S1,T2,T1,A1,A2,A3,T1+T2,J1"
    configs = {
        method: launcher.yaml.safe_load((tmp_path / "generated_configs" / f"{method}_seed1.yaml").read_text())
        for method in methods
    }
    for method in ("S1-LG", "T2-LG"):
        cfg = configs[method]
        assert cfg["training"]["proto_target_type"] == "gaussian"
        assert cfg["training"]["use_gaussian_beam_targets"] is True
        assert cfg["training"]["use_circular_soft_targets"] is False
        assert cfg["training"]["beam_label_circular"] is False
        assert cfg["training"]["circular_beam_distance"] is False
        assert cfg["evaluation"]["dba_distance_mode"] == "linear"
        resolved = u_mask_beam_jepa_config(cfg)
        assert resolved["proto_target_type"] == "gaussian"
        assert resolved["beam_label_circular"] is False
        assert resolved["circular_beam_distance"] is False
        assert pcpg_radar_balance_config(cfg)["circular_beam_distance"] is False
    for method in ("S1-CLS", "T2-CLS"):
        cfg = configs[method]
        primary = cfg["model"]["primary"]
        assert primary["head_type"] == "classifier"
        assert primary["use_beam_prototype_alignment"] is False
        assert primary["router_use_prototype_margin"] is False
        assert cfg["training"]["use_modality_prototype_loss"] is False
        assert cfg["training"]["beam_proto_align_weight"] == 0.0
        assert cfg["training"]["beam_label_circular"] is False
        assert cfg["training"]["circular_beam_distance"] is False
        resolved = u_mask_beam_jepa_config(cfg)
        assert resolved["use_beam_prototype_alignment"] is False
        assert resolved["lambda_proto"] == 0.0
        assert resolved["lambda_modality_proto"] == 0.0
        assert pcpg_radar_balance_config(cfg)["circular_beam_distance"] is False
    for method in ("T2-LG", "T2-CLS"):
        assert configs[method]["training"]["superset_consistency"]["confidence_gated_kl"] is True
    for method in ("S1-LG", "S1-CLS"):
        assert configs[method]["training"]["superset_consistency"]["enabled"] is False


def test_rmbp_channel_attention_core_is_registered() -> None:
    import_default_components()
    assert "rmbp_channel_attention_fusion" in REPRESENTATION_CORES.list()


def test_summary_generates_five_method_matrices(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    for method_index, method in enumerate(("ours_c2_main", "ours_b4_nonrouter_soft_jepa", "ours_e5_low_lr_pcpg", "amber_full", "rmbp_mm")):
        seed_dir = eval_dir / method / "seed1"
        seed_dir.mkdir(parents=True)
        rows = [
            {"missing_rate": "0.0", "full": 0.8 + method_index * 0.01, "drop1": 0.7, "drop2": 0.6, "drop3": 0.5},
            {"missing_rate": "0.8", "full": 0.6, "drop1": 0.5, "drop2": 0.4, "drop3": 0.3 + method_index * 0.01},
        ]
        for filename in ("top1_matrix.csv", "within3_matrix.csv", "mae_matrix.csv"):
            _write_csv(seed_dir / filename, rows, ["missing_rate", "full", "drop1", "drop2", "drop3"])
        _write_csv(seed_dir / "pattern_metrics.csv", [{"pattern": "missing_image", "top1": 0.5}], ["pattern", "top1"])
    summary = _load_script("summarize_h5_p1_temporal_matrix_v1.py")
    out_dir = tmp_path / "summary"
    assert summary.main(["--eval_dir", str(eval_dir), "--output_dir", str(out_dir)]) == 0
    for method in ("ours_c2_main", "ours_b4_nonrouter_soft_jepa", "ours_e5_low_lr_pcpg", "amber_full", "rmbp_mm"):
        assert (out_dir / f"{method}_top1_matrix.csv").exists()
        assert (out_dir / f"{method}_within3_matrix.csv").exists()
        assert (out_dir / f"{method}_mae_matrix.csv").exists()
    text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "H5/P1 Temporal Matrix v1 Summary" in text
    assert "time-aware router" in text
    assert "Drop0 Guardrail" not in text
    assert "drop0_guardrail_status" not in _read_csv(out_dir / "summary.csv")[0]


def test_s1_summary_keeps_extra_metrics_diagnostics_and_guardrail(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    clean_top1 = {"S1": 0.8, "T2": 0.796, "T1": 0.794}
    for method in clean_top1:
        seed_dir = eval_dir / method / "seed1"
        seed_dir.mkdir(parents=True)
        rows = [
            {"missing_rate": "0.0", "full": clean_top1[method], "drop1": 0.7, "drop2": 0.6, "drop3": 0.5},
            {"missing_rate": "0.8", "full": 0.6, "drop1": 0.5, "drop2": 0.4, "drop3": 0.3},
        ]
        for filename in ("top1_matrix.csv", "top3_matrix.csv", "within3_matrix.csv", "adba_matrix.csv", "mae_matrix.csv"):
            _write_csv(seed_dir / filename, rows, ["missing_rate", "full", "drop1", "drop2", "drop3"])
        _write_csv(
            seed_dir / "training_diagnostics.csv",
            [{
                "method": method,
                "seed": 1,
                "temporal_pooling/type": "masked_mean",
                "superset_consistency/gate_mean": 0.75,
                "dba_distance_mode": "circular",
            }],
            ["method", "seed", "temporal_pooling/type", "superset_consistency/gate_mean", "dba_distance_mode"],
        )
        _write_csv(
            seed_dir / "diagnostics.csv",
            [{"method": method, "seed": 1, "missing_rate": 0.8, "drop_count": 0, "pattern": "full", "gate_entropy": 0.25}],
            ["method", "seed", "missing_rate", "drop_count", "pattern", "gate_entropy"],
        )

    summary = _load_script("summarize_h5_p1_temporal_matrix_v1.py")
    out_dir = tmp_path / "summary"
    assert summary.main([
        "--profile",
        "s1_lightweight",
        "--eval_dir",
        str(eval_dir),
        "--output_dir",
        str(out_dir),
        "--methods",
        "S1,T2,T1",
    ]) == 0
    rows = {row["method"]: row for row in _read_csv(out_dir / "summary.csv")}
    assert rows["S1"]["drop0_guardrail_status"] == "baseline"
    assert rows["T2"]["drop0_guardrail_status"] == "pass"
    assert rows["T1"]["drop0_guardrail_status"] == "fail"
    assert rows["T2"]["mean_top3_five_rates"]
    assert rows["T2"]["mean_adba_five_rates"]
    diagnostics = {row["method"]: row for row in _read_csv(out_dir / "diagnostics.csv")}
    assert diagnostics["T2"]["temporal_pooling/type"] == "masked_mean"
    assert float(diagnostics["T2"]["superset_consistency/gate_mean"]) == 0.75
    text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "Top3 Matrices" in text
    assert "ADBA Matrices" in text
    assert "Drop0 Guardrail" in text
    assert (out_dir / "seed_summary.csv").exists()
    assert (out_dir / "seed_deltas.csv").exists()
    assert (out_dir / "paired_mask_deltas.csv").exists()
    assert (out_dir / "gate_decisions.csv").exists()


def test_eval_mask_cache_cli_inputs(tmp_path: Path) -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    cache = generate_fixed_eval_mask_cache(
        tmp_path / "masks",
        rates=(0.0, 0.2, 0.4, 0.6, 0.8),
        drop_counts=(0, 1, 2, 3),
        num_masks_per_cell=4,
        seed=20260708,
    )
    assert len(cache) == 20
    assert eval_script.MATRIX_COLUMNS == ["missing_rate", "full", "drop1", "drop2", "drop3"]
    payload = json.loads((tmp_path / "masks" / "rate_0.8_drop3.json").read_text(encoding="utf-8"))
    assert payload["history_window"] == 5
    assert payload["num_modalities"] == 4
    eval_script._validate_eval_mask_cache_contract(
        cache,
        rates=(0.0, 0.2, 0.4, 0.6, 0.8),
        drop_counts=(0, 1, 2, 3),
        mask_types=("modality_frame", "frame_level", "block"),
        num_masks_per_cell=4,
        seed=20260708,
        history_window=5,
        modalities=("image", "radar", "lidar", "gps"),
    )
    cache[(0.0, 0)]["seed"] = 1
    with pytest.raises(ValueError, match="cache contract mismatch"):
        eval_script._validate_eval_mask_cache_contract(
            cache,
            rates=(0.0, 0.2, 0.4, 0.6, 0.8),
            drop_counts=(0, 1, 2, 3),
            mask_types=("modality_frame", "frame_level", "block"),
            num_masks_per_cell=4,
            seed=20260708,
            history_window=5,
            modalities=("image", "radar", "lidar", "gps"),
        )


def test_eval_mask_identity_hashes_actual_mask_not_type_or_index() -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    base = {
        "mask_type": "frame_level",
        "modality_temporal_mask": [[1, 1, 1, 1], [0, 0, 0, 0]],
    }
    same_mask = {**base, "mask_type": "block"}
    changed_mask = {
        **base,
        "modality_temporal_mask": [[1, 1, 1, 1], [0, 0, 0, 1]],
    }
    kwargs = {
        "modalities": ["image", "radar", "lidar", "gps"],
        "cache_checksum": "cache-a",
        "cache_seed": 7,
    }

    first = eval_script._mask_identity(base, mask_index=0, **kwargs)
    duplicate = eval_script._mask_identity(same_mask, mask_index=3, **kwargs)
    changed = eval_script._mask_identity(changed_mask, mask_index=0, **kwargs)

    assert first["mask_digest"] == duplicate["mask_digest"]
    assert first["mask_digest"] != changed["mask_digest"]
    assert first["mask_index"] == 0
    assert duplicate["mask_index"] == 3
    assert duplicate["mask_type"] == "block"
    assert first["mask_cache_checksum"] == "cache-a"
    assert first["mask_cache_seed"] == "7"
    assert first["observed_missing_rate"] == pytest.approx(0.5)
    assert first["last_frame_available"] is False
    assert first["last_frame_available_modalities"] == 0
    assert first["trailing_fully_missing_frames"] == 1
    assert changed["last_frame_available"] is True
    assert changed["last_frame_available_modalities"] == 1


def test_s1_eval_prefers_best_top1_checkpoint(tmp_path: Path) -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    checkpoint_dir = tmp_path / "S1" / "seed1" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_avg_missing_top1.pth").touch()
    (checkpoint_dir / "best_top1.pth").touch()

    default_path, default_policy = eval_script._find_checkpoint(tmp_path, "S1", 1)
    s1_path, s1_policy = eval_script._find_checkpoint(tmp_path, "S1", 1, profile="s1_lightweight")

    assert default_path.name == "best_avg_missing_top1.pth"
    assert default_policy == "best_avg_missing_top1"
    assert s1_path.name == "best_top1.pth"
    assert s1_policy == "best_top1"


def test_eval_mask_is_reordered_from_cache_to_model_modalities() -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    model = type("Model", (), {"modalities": ("image", "radar", "gps", "lidar")})()
    mask_item = {
        "modality_temporal_mask": [
            [1, 1, 1, 0],
            [1, 1, 0, 1],
        ]
    }

    mask, modalities = eval_script._mask_in_model_order(
        model,
        mask_item,
        ("image", "radar", "lidar", "gps"),
    )

    assert modalities == ("image", "radar", "gps", "lidar")
    assert mask.tolist() == [[True, True, False, True], [True, True, True, False]]
    batch = {
        "image": torch.ones(1, 2, 1),
        "radar_ra": torch.ones(1, 2, 1),
        "radar_da": torch.ones(1, 2, 1),
        "gps": torch.ones(1, 2, 1),
        "lidar": torch.ones(1, 2, 1),
    }
    out = apply_modality_temporal_mask_to_batch(batch, mask, modalities=modalities)
    assert out["gps"][0, 0].item() == 0.0
    assert out["gps"][0, 1].item() == 1.0
    assert out["lidar"][0, 0].item() == 1.0
    assert out["lidar"][0, 1].item() == 0.0


def test_eval_batch_size_override_updates_split_loaders() -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    cfg = {"data": {"dataloader": {"test_batch_size": 128, "validation": {"batch_size": 64}}}}

    eval_script._override_eval_batch_size(cfg, 8)

    assert cfg["data"]["dataloader"]["test_batch_size"] == 8
    assert cfg["data"]["dataloader"]["validation_batch_size"] == 8
    assert cfg["data"]["dataloader"]["validation"]["batch_size"] == 8


def test_eval_distance_override_is_explicit_and_records_provenance() -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    cfg = {
        "model": {"primary": {"head_type": "prototype", "use_beam_prototype_alignment": True}},
        "training": {"beam_label_circular": True},
        "evaluation": {"metric_profile": "configured"},
    }
    assert eval_script._apply_dba_distance_mode(cfg, "linear") == "linear"
    assert cfg["evaluation"]["beam_distance_circular"] is False
    assert cfg["eval"]["beam_distance_circular"] is False
    provenance = eval_script._evaluation_provenance(cfg)
    assert provenance == {
        "ablation_id": "",
        "training_beam_geometry": "circular",
        "prototype_target_geometry": "circular",
        "router_oracle_geometry": "not_applicable",
        "head_type": "prototype",
        "prototype_enabled": True,
        "prototype_head_enabled": True,
        "bpa_auxiliary_enabled": True,
        "use_amber_cma_analogue": False,
        "lambda_amber_cma": 0.0,
        "amber_cma_temperature": 0.2,
        "metric_profile": "64_beam_linear_topk",
        "dba_distance_mode": "linear",
    }

    logits = torch.full((1, 64), -20.0)
    logits[0, 63] = 20.0
    target = torch.tensor([0])
    linear = eval_script._beam_classification_metrics(logits, target, cfg)
    assert linear["within_3"] == 0.0
    assert linear["mae"] == 63.0
    eval_script._apply_dba_distance_mode(cfg, "circular")
    circular = eval_script._beam_classification_metrics(logits, target, cfg)
    assert circular["within_3"] == 1.0
    assert circular["mae"] == 1.0


def test_eval_matrix_uses_explicit_final_test_loader() -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    validation = object()
    test = object()

    assert eval_script._final_test_loader({"validation": validation, "test": test}) is test
    with pytest.raises(KeyError, match="explicit test dataloader"):
        eval_script._final_test_loader({"validation": validation})


def test_eval_collects_available_pooling_router_and_training_diagnostics(tmp_path: Path) -> None:
    eval_script = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    diagnostics = eval_script._router_metrics(
        {
            "reliability_fusion_weights": torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
            "temporal_pooling_weights": torch.full((1, 5, 4), 0.2),
            "temporal_mask_statistics": torch.tensor([[[0.4, 0.5]] * 4]),
            "temporal_mask_statistic_names": ("coverage", "last_age"),
            "coverage_shrinkage_rho": torch.tensor([0.25]),
        },
        ("image", "radar", "gps", "lidar"),
    )
    assert abs(diagnostics["mean_gate_image"] - 0.7) < 1e-6
    assert abs(diagnostics["mean_temporal_pooling_weight_t4"] - 0.2) < 1e-6
    assert abs(diagnostics["mean_mask_statistic_coverage"] - 0.4) < 1e-6
    assert abs(diagnostics["coverage_shrinkage_rho"] - 0.25) < 1e-6

    run_dir = tmp_path / "T2" / "seed1"
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "metrics.csv",
        [
            {
                "epoch": 1,
                "loss/superset_consistency": 0.2,
                "superset_consistency/gate_mean": 0.75,
                "beam_monotonic_rank/partial_excess_violation_rate": 0.25,
                "beam_monotonic_rank/superset_worse_rate": 0.1,
            }
        ],
        [
            "epoch",
            "loss/superset_consistency",
            "superset_consistency/gate_mean",
            "beam_monotonic_rank/partial_excess_violation_rate",
            "beam_monotonic_rank/superset_worse_rate",
        ],
    )
    cfg = {
        "model": {"primary": {"temporal_pooling": {"enabled": True, "type": "masked_mean"}}},
        "training": {"superset_consistency": {"enabled": True, "confidence_gated_kl": True}},
    }
    row = eval_script._training_diagnostics(tmp_path, "T2", 1, cfg)
    assert row["temporal_pooling/type"] == "masked_mean"
    assert row["loss/superset_consistency"] == 0.2
    assert row["superset_consistency/gate_mean"] == 0.75
    assert row["beam_monotonic_rank/partial_excess_violation_rate"] == 0.25
    assert row["beam_monotonic_rank/superset_worse_rate"] == 0.1


def test_paired_mask_evidence_strictly_pairs_then_deduplicates_by_digest() -> None:
    summary = _load_script("summarize_h5_p1_temporal_matrix_v1.py")
    rows = _synthetic_paired_pattern_rows()

    paired, statuses = summary._paired_mask_rows(rows, [("T2", "S1")])

    assert statuses[("T2", "S1", 1)]["status"] == "complete"
    assert statuses[("T2", "S1", 1)]["entry_count"] == 20
    assert statuses[("T2", "S1", 1)]["unique_mask_count"] == 16
    assert len(paired) == 16
    assert max(int(row["duplicate_entry_count"]) for row in paired) == 4
    assert any(int(row["duplicate_entry_count"]) == 2 for row in paired)
    assert summary._paired_rate_equal_mean(paired, "top1_delta") == pytest.approx(0.03)

    for row in rows:
        if row["method"] == "T2":
            row["dba_distance_mode"] = "linear"
    incompatible, incompatible_status = summary._paired_mask_rows(rows, [("T2", "S1")])
    assert incompatible == []
    assert incompatible_status[("T2", "S1", 1)]["status"] == "unavailable"
    assert "dba_distance_mode mismatch" in incompatible_status[("T2", "S1", 1)]["reason"]

    sparse_rows = [row for row in _synthetic_paired_pattern_rows() if row["mask_index"] == 0]
    _, sparse_status = summary._paired_mask_rows(sparse_rows, [("T2", "S1")])
    assert sparse_status[("T2", "S1", 1)]["status"] == "unavailable"
    assert "frozen protocol mismatch" in sparse_status[("T2", "S1", 1)]["reason"]

    mixed_rows = _synthetic_paired_pattern_rows()
    for row in mixed_rows:
        if row["method"] == "T2" and row["missing_rate"] in {0.4, 0.6, 0.8}:
            row["dba_distance_mode"] = "linear"
    _, mixed_status = summary._paired_mask_rows(mixed_rows, [("T2", "S1")])
    assert mixed_status[("T2", "S1", 1)]["status"] == "unavailable"
    assert "mixed or missing dba_distance_mode" in mixed_status[("T2", "S1", 1)]["reason"]

    head_mismatch = _synthetic_paired_pattern_rows()
    for row in head_mismatch:
        if row["method"] == "T2":
            row["head_type"] = "classifier"
            row["prototype_enabled"] = False
    _, head_status = summary._paired_mask_rows(head_mismatch, [("T2", "S1")])
    assert head_status[("T2", "S1", 1)]["status"] == "unavailable"
    assert "matched provenance mismatch for head_type" in head_status[("T2", "S1", 1)]["reason"]

    missing_aux = _synthetic_paired_pattern_rows()
    missing_aux[0]["top3"] = ""
    _, missing_aux_status = summary._paired_mask_rows(missing_aux, [("T2", "S1")])
    assert missing_aux_status[("T2", "S1", 1)]["status"] == "unavailable"
    assert "missing or non-finite top3 metric" in missing_aux_status[("T2", "S1", 1)]["reason"]


def test_candidate_and_final_gate_boundaries_and_missing_seed_are_explicit() -> None:
    summary = _load_script("summarize_h5_p1_temporal_matrix_v1.py")
    cache = {
        "mask_cache_checksums": "cache-0,cache-2,cache-4,cache-6,cache-8",
        "mask_cache_seeds": "20260708",
        "mask_cache_entry_count": 20,
    }
    provenance = {
        "training_beam_geometry": "circular",
        "prototype_target_geometry": "circular",
        "router_oracle_geometry": "circular",
        "head_type": "prototype",
        "prototype_enabled": True,
        "metric_profile": "64_beam_circular_topk",
        "dba_distance_mode": "circular",
    }
    seed_summaries = [
        {"method": "T2", "seed": 1, "status": "complete", "mean_top1_five_rates": 0.5, **cache},
        {"method": "T2-LG", "seed": 1, "status": "complete", "mean_top1_five_rates": 0.495, **cache},
    ]
    candidate_delta = {
        "candidate_method": "T2-LG",
        "baseline_method": "S1-LG",
        "seed": 1,
        "status": "complete",
        "mean_top1_five_rates_delta": 0.0,
        "top1_drop80_delta": 0.01,
        "top1_drop0_delta": -0.005,
    }

    decisions = summary._gate_decision_rows(
        seed_summaries,
        [candidate_delta],
        [("T2-LG", "S1-LG")],
        current_t2_method="T2",
        max_drop=0.005,
    )
    screen = next(row for row in decisions if row["stage"] == "candidate_screen")
    assert screen["status"] == "fail"
    assert screen["criterion_mean5_positive"] == "fail"
    assert screen["criterion_drop0_guardrail"] == "pass"
    assert screen["criterion_vs_current_t2_guardrail"] == "pass"

    candidate_delta["mean_top1_five_rates_delta"] = 1e-6
    decisions = summary._gate_decision_rows(
        seed_summaries,
        [candidate_delta],
        [("T2-LG", "S1-LG")],
        current_t2_method="T2",
        max_drop=0.005,
    )
    assert next(row for row in decisions if row["stage"] == "candidate_screen")["status"] == "pass"

    mismatched = [dict(row) for row in seed_summaries]
    mismatched[0]["mask_cache_checksums"] = "different-cache"
    mismatch_decisions = summary._gate_decision_rows(
        mismatched,
        [candidate_delta],
        [("T2-LG", "S1-LG")],
        current_t2_method="T2",
        max_drop=0.005,
    )
    mismatch_screen = next(row for row in mismatch_decisions if row["stage"] == "candidate_screen")
    assert mismatch_screen["status"] == "unavailable"
    assert "cache provenance mismatch" in mismatch_screen["reason"]

    final_deltas = []
    for seed, delta in ((1, 0.01), (2, 0.01), (3, -0.005)):
        final_deltas.append(
            {
                "candidate_method": "T2",
                "baseline_method": "S1",
                "seed": seed,
                "status": "complete",
                "mean_top1_five_rates_delta": delta,
                "mean_top1_drop0_60_delta": delta,
                "top1_drop80_delta": delta,
                "top1_drop0_delta": -0.005,
            }
        )
    final_summaries = [
        {"method": method, "seed": seed, "status": "complete", **cache, **provenance}
        for method in ("S1", "T2")
        for seed in (1, 2, 3)
    ]
    final = summary._gate_decision_rows(
        final_summaries,
        final_deltas,
        [("T2", "S1")],
        current_t2_method="T2",
        max_drop=0.005,
    )
    assert next(row for row in final if row["stage"] == "final_multiseed")["status"] == "pass"

    missing = summary._gate_decision_rows(
        final_summaries,
        final_deltas[:2],
        [("T2", "S1")],
        current_t2_method="T2",
        max_drop=0.005,
    )
    missing_final = next(row for row in missing if row["stage"] == "final_multiseed")
    assert missing_final["status"] == "unavailable"
    assert "required seeds mismatch" in missing_final["reason"]


def test_seed_summary_checks_matrix_against_raw_four_entry_protocol(tmp_path: Path) -> None:
    summary = _load_script("summarize_h5_p1_temporal_matrix_v1.py")
    seed_dir = tmp_path / "T2" / "seed1"
    seed_dir.mkdir(parents=True)
    patterns = [row for row in _synthetic_paired_pattern_rows() if row["method"] == "T2"]
    metric_keys = {
        "top1": "top1",
        "top3": "top3",
        "within3": "within_3",
        "adba": "adba",
        "mae": "mae",
    }
    matrix_rows = {}
    for metric, key in metric_keys.items():
        rows = []
        for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
            values = [float(row[key]) for row in patterns if row["missing_rate"] == rate]
            rows.append({"missing_rate": rate, "full": sum(values) / len(values)})
        matrix_rows[metric] = rows
        _write_csv(seed_dir / summary.S1_MATRIX_FILES[metric], rows, ["missing_rate", "full"])
    _write_csv(seed_dir / "pattern_metrics.csv", patterns, list(patterns[0]))
    provenance = {
        key: patterns[0][key]
        for key in summary.MATCHED_PROVENANCE_FIELDS
    }
    _write_csv(seed_dir / "training_diagnostics.csv", [provenance], list(provenance))
    mask_stats = []
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
        rate_rows = [row for row in patterns if row["missing_rate"] == rate]
        mask_stats.append(
            {
                "missing_rate": rate,
                "drop_count": 0,
                "num_masks": 4,
                "num_unique_masks": len({row["mask_digest"] for row in rate_rows}),
            }
        )
    _write_csv(
        seed_dir / "mask_stats.csv",
        mask_stats,
        ["missing_rate", "drop_count", "num_masks", "num_unique_masks"],
    )

    seed_rows = summary._seed_summary_rows(tmp_path, ["T2"], summary.S1_MATRIX_FILES)
    assert seed_rows[0]["status"] == "complete"
    assert seed_rows[0]["mask_cache_entry_count"] == 20
    assert seed_rows[0]["mask_cache_unique_count"] == 16

    matrix_rows["top1"][-1]["full"] += 0.1
    _write_csv(seed_dir / "top1_matrix.csv", matrix_rows["top1"], ["missing_rate", "full"])
    stale = summary._seed_summary_rows(tmp_path, ["T2"], summary.S1_MATRIX_FILES)
    assert stale[0]["status"] == "unavailable"
    assert "matrix/raw 4-entry mismatch" in stale[0]["reason"]

    _write_csv(seed_dir / "top1_matrix.csv", matrix_rows["top1"][:-1], ["missing_rate", "full"])
    missing_rate = summary._seed_summary_rows(tmp_path, ["T2"], summary.S1_MATRIX_FILES)
    assert missing_rate[0]["status"] == "unavailable"
    assert "top1 matrix rate rows mismatch" in missing_rate[0]["reason"]

    valid_top1 = []
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
        values = [float(row["top1"]) for row in patterns if row["missing_rate"] == rate]
        valid_top1.append({"missing_rate": rate, "full": sum(values) / len(values)})
    expected_mean = sum(float(row["full"]) for row in valid_top1) / len(valid_top1)
    top1_with_invalid_rate = [*valid_top1, {"missing_rate": "", "full": 99.0}]
    _write_csv(seed_dir / "top1_matrix.csv", top1_with_invalid_rate, ["missing_rate", "full"])
    invalid_rate = summary._seed_summary_rows(tmp_path, ["T2"], summary.S1_MATRIX_FILES)
    assert invalid_rate[0]["status"] == "unavailable"
    assert "invalid_rate_rows=1" in invalid_rate[0]["reason"]
    assert invalid_rate[0]["mean_top1_five_rates"] == pytest.approx(expected_mean)

    _write_csv(seed_dir / "top1_matrix.csv", valid_top1, ["missing_rate", "full"])
    _write_csv(seed_dir / "top3_matrix.csv", matrix_rows["top3"][:-1], ["missing_rate", "full"])
    missing_aux = summary._seed_summary_rows(tmp_path, ["T2"], summary.S1_MATRIX_FILES)
    assert missing_aux[0]["status"] == "unavailable"
    assert "top3 matrix rate rows mismatch" in missing_aux[0]["reason"]

    seed2_dir = tmp_path / "T2" / "seed2"
    seed2_dir.mkdir(parents=True)
    _write_csv(
        seed2_dir / "within3_matrix.csv",
        matrix_rows["within3"],
        ["missing_rate", "full"],
    )
    _write_csv(
        seed2_dir / "training_diagnostics.csv",
        [{**provenance, "dba_distance_mode": "linear"}],
        list(provenance),
    )
    assert summary._aggregate_method(
        tmp_path / "T2",
        "within3_matrix.csv",
        distance_sensitive=True,
    ) == []


def _synthetic_paired_pattern_rows() -> list[dict]:
    rows = []
    rate_digests = {
        0.0: ["r0", "r0", "r0", "r0"],
        0.2: ["r2a", "r2a", "r2b", "r2c"],
        0.4: ["r4a", "r4b", "r4c", "r4d"],
        0.6: ["r6a", "r6b", "r6c", "r6d"],
        0.8: ["r8a", "r8b", "r8c", "r8d"],
    }
    rate_deltas = {0.0: 0.01, 0.2: 0.02, 0.4: 0.03, 0.6: 0.04, 0.8: 0.05}
    mask_types = ["modality_frame", "frame_level", "block", "modality_frame"]
    for method in ("S1", "T2"):
        for rate, digests in rate_digests.items():
            for index, digest in enumerate(digests):
                baseline = 0.5 + index * 0.001
                value = baseline if method == "S1" else baseline + rate_deltas[rate]
                rows.append(
                    {
                        "method": method,
                        "seed": 1,
                        "missing_rate": rate,
                        "drop_count": 0,
                        "mask_index": index,
                        "mask_type": mask_types[index],
                        "mask_digest": digest,
                        "mask_cache_checksum": f"cache-{rate}",
                        "mask_cache_seed": "20260708",
                        "dba_distance_mode": "circular",
                        "training_beam_geometry": "circular",
                        "prototype_target_geometry": "circular",
                        "router_oracle_geometry": "circular",
                        "head_type": "prototype",
                        "prototype_enabled": True,
                        "metric_profile": "64_beam_circular_topk",
                        "top1": value,
                        "top3": value + 0.1,
                        "within_3": value + 0.2,
                        "adba": value + 0.3,
                        "mae": 1.0 - value,
                    }
                )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
