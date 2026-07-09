import csv
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from kd_sensing.config import load_config
from kd_sensing.data.temporal_missing import generate_fixed_eval_mask_cache, sample_stratified_modality_temporal_mask
from kd_sensing.losses.u_mask_beam_jepa import _per_time_modality_oracle_ce
from kd_sensing.models.u_mask_beam_jepa import (
    global_oracle_targets,
    modality_oracle_targets,
    per_time_modality_oracle_targets,
    temporal_oracle_targets,
)
from kd_sensing.registries import ENCODERS, MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("temporal_router_test_encoder", force=True)
class TemporalRouterTestEncoder(nn.Module):
    def __init__(self, output_dim: int = 12, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size, steps = batch.shape[:2]
        pooled = batch.float().reshape(batch_size, steps, -1).mean(dim=-1, keepdim=True)
        return pooled.expand(batch_size, steps, self.output_dim)


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model(router_type: str):
    import_default_components()
    return MODELS.build(
        {
            "type": "u_mask_beam_jepa",
            "modalities": ["image", "radar", "lidar", "gps"],
            "d_model": 12,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "fusion_type": "supervised_router",
            "head_type": "prototype",
            "router_supervision": "oracle",
            "router_distill_weight": 0.1,
            "temporal_router_distill_weight": 0.1,
            "temporal_router_type": router_type,
            "encoders": {
                "image": {"type": "temporal_router_test_encoder", "output_dim": 12},
                "radar": {"type": "temporal_router_test_encoder", "output_dim": 12},
                "lidar": {"type": "temporal_router_test_encoder", "output_dim": 12},
                "gps": {"type": "temporal_router_test_encoder", "output_dim": 12},
            },
        }
    )


def _batch(batch_size: int = 2, steps: int = 5):
    return {
        "image_batch": torch.randn(batch_size, steps, 3, 4, 4),
        "radar_batch": torch.randn(batch_size, steps, 2, 4, 4),
        "lidar_batch": torch.randn(batch_size, steps, 3, 4, 4),
        "gps_batch": torch.randn(batch_size, steps, 3),
    }


def _mask() -> torch.Tensor:
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    mask[0, :, 1] = False
    mask[0, 2, :] = False
    mask[1, :, :] = False
    mask[1, 4, 3] = True
    return mask


def test_mask_sampler_shape_rate_drop3_fallback() -> None:
    item = sample_stratified_modality_temporal_mask(
        history_window=5,
        fixed_drop_modalities=["image", "radar", "lidar"],
        fixed_rate=0.8,
        fixed_mask_type="modality_frame",
    )
    mask = item["modality_temporal_mask"]
    assert mask.shape == (5, 4)
    assert item["drop_count"] == 3
    assert bool(mask.any())
    assert not bool(mask[:, :3].any())


def test_s1_forward_gate_shape_mask_and_single_modality() -> None:
    out = _model("s1_temporalagg_modality")(**_batch(), modality_temporal_mask=_mask())
    gate = out["temporal_router_modality_gate"]
    assert gate.shape == (2, 4)
    assert torch.all(gate[0, 1] == 0)
    assert gate[1, 3].item() == 1.0
    assert torch.isfinite(out["logits"]).all()


def test_s2_forward_per_time_gate_and_empty_time() -> None:
    out = _model("s2_pertime_modality")(**_batch(), modality_temporal_mask=_mask())
    gate = out["temporal_router_modality_gate"]
    assert gate.shape == (2, 5, 4)
    assert torch.all(gate[0, 2] == 0)
    available = _mask().any(dim=2)
    sums = gate.sum(dim=2)
    assert torch.allclose(sums[available], torch.ones_like(sums[available]))
    assert torch.isfinite(out["logits"]).all()


def test_s3_forward_temporal_gate_shape_and_sum() -> None:
    out = _model("s3_two_level")(**_batch(), modality_temporal_mask=_mask())
    assert out["temporal_router_modality_gate"].shape == (2, 5, 4)
    temporal_gate = out["temporal_gate"]
    assert temporal_gate.shape == (2, 5)
    assert torch.all(temporal_gate[0, 2] == 0)
    assert torch.allclose(temporal_gate.sum(dim=1), torch.ones(2))
    assert torch.isfinite(out["logits"]).all()


def test_s4_forward_global_gate_shape_and_single_cell() -> None:
    out = _model("s4_global")(**_batch(), modality_temporal_mask=_mask())
    gate = out["global_gate"]
    assert gate.shape == (2, 5, 4)
    assert torch.all(gate[~_mask()] == 0)
    assert torch.allclose(gate.sum(dim=(1, 2)), torch.ones(2))
    assert gate[1, 4, 3].item() == 1.0
    assert torch.isfinite(out["logits"]).all()


def test_oracle_targets_choose_only_available_and_tie_first() -> None:
    logits = torch.zeros(1, 4, 8)
    logits[0, :, 0] = 1.0
    mask = torch.tensor([[False, True, True, False]])
    assert modality_oracle_targets(logits, torch.tensor([0]), mask).item() == 1
    per_time = logits.view(1, 1, 4, 8).expand(1, 5, 4, 8)
    per_mask = mask.view(1, 1, 4).expand(1, 5, 4)
    assert torch.all(per_time_modality_oracle_targets(per_time, torch.tensor([0]), per_mask) == 1)
    time_logits = torch.zeros(1, 5, 8)
    time_mask = torch.tensor([[False, False, True, True, False]])
    assert temporal_oracle_targets(time_logits, torch.tensor([0]), time_mask).item() == 2
    assert global_oracle_targets(per_time, torch.tensor([0]), per_mask).item() == 1


def test_router_oracle_loss_accepts_s1_gate_and_half_masks() -> None:
    labels = torch.tensor([0, 1])
    gate = torch.zeros(2, 4)
    logits = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [False, True, True, False]])
    loss, diag = _per_time_modality_oracle_ce(gate, logits, labels, mask)
    assert torch.isfinite(loss)
    assert diag["router_oracle_modality_oracle_active_ratio"] > 0

    gate_half = torch.zeros(2, 5, 4, dtype=torch.float16)
    logits_time = torch.randn(2, 5, 4, 8, dtype=torch.float16)
    mask_time = mask.unsqueeze(1).expand(-1, 5, -1)
    loss_half, _ = _per_time_modality_oracle_ce(gate_half, logits_time, labels, mask_time)
    assert torch.isfinite(loss_half)


def test_eval_mask_cache_reproducible_balanced_checksum(tmp_path: Path) -> None:
    first = generate_fixed_eval_mask_cache(tmp_path, rates=(0.2,), drop_counts=(1, 2, 3), num_masks_per_cell=16, seed=20260708)
    second = generate_fixed_eval_mask_cache(tmp_path, rates=(0.2,), drop_counts=(1, 2, 3), num_masks_per_cell=16, seed=1)
    assert first == second
    combos = [tuple(item["dropped_modalities"]) for item in first[(0.2, 2)]["masks"]]
    assert len(set(combos)) == 6
    assert first[(0.2, 3)]["checksum"]


def test_launcher_dry_run_aliases_and_manifest(tmp_path: Path) -> None:
    launcher = _load_script("launch_temporal_router_s1_s4_v1.py")
    code = launcher.main([
        "--output_root",
        str(tmp_path),
        "--seeds",
        "1",
        "--methods",
        "s1,s2,s3,s4,amber_full,rmbp_mm",
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
    assert [row["method"] for row in rows] == list(launcher.DEFAULT_METHODS.split(","))
    assert {row["history_window"] for row in rows} == {"5"}
    assert max(int(row["gpu"]) for row in rows if row["gpu"]) <= 7


def test_launcher_defaults_to_gpu2_7_and_thread_caps(tmp_path: Path) -> None:
    launcher = _load_script("launch_temporal_router_s1_s4_v1.py")
    assert launcher.main(["--output_root", str(tmp_path), "--seeds", "1", "--dry_run"]) == 0
    rows = _read_csv(tmp_path / "job_manifest.csv")
    assert [row["gpu"] for row in rows] == ["2", "3", "4", "5", "6", "7"]
    config = yaml.safe_load((tmp_path / "generated_configs" / "s1_temporalagg_modality_router_seed1.yaml").read_text(encoding="utf-8"))
    assert config["data"]["dataloader"]["train_batch_size"] == 64
    assert config["data"]["dataloader"]["num_workers"] == 4
    assert config["data"]["dataloader"]["prefetch_factor"] == 2
    assert config["training"]["cpu_threads"]["intra_op"] == 4
    assert config["training"]["cpu_threads"]["inter_op"] == 2
    assert config["output"]["progress"]["enabled"] is False


def test_launcher_generated_config_uses_only_temporal_missing_sampler(tmp_path: Path) -> None:
    launcher = _load_script("launch_temporal_router_s1_s4_v1.py")
    assert launcher.main(["--output_root", str(tmp_path), "--seeds", "1", "--methods", "s1,rmbp_mm", "--dry_run"]) == 0
    for config_path in (
        tmp_path / "generated_configs" / "s1_temporalagg_modality_router_seed1.yaml",
        tmp_path / "generated_configs" / "rmbp_mm_seed1.yaml",
    ):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["training"]["random_modality_dropout"]["enabled"] is False
        assert raw["training"]["mask_sampler"] == "default"
        assert raw["loss"]["u_mask_beam_jepa"]["missing_pattern"]["available_modalities"] == ["image", "radar", "gps", "lidar"]
        resolved = load_config(config_path)
        operators = [
            op["type"]
            for profile in resolved["difficulty"]["profiles"]
            for op in profile["operators"]
        ]
        assert operators == ["temporal_missing"]


def test_summary_parser_generates_six_method_matrices(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    methods = _load_script("summarize_temporal_router_s1_s4_v1.py").DEFAULT_METHODS
    for index, method in enumerate(methods):
        seed_dir = eval_dir / method / "seed1"
        seed_dir.mkdir(parents=True)
        rows = [{"missing_rate": "0.0", "full": 0.8 + index * 0.01, "drop1": 0.7, "drop2": 0.6, "drop3": 0.5}]
        for filename in ("top1_matrix.csv", "within3_matrix.csv", "mae_matrix.csv"):
            _write_csv(seed_dir / filename, rows, ["missing_rate", "full", "drop1", "drop2", "drop3"])
        _write_csv(seed_dir / "pattern_metrics.csv", [{"pattern": "missing_image", "top1": 0.5}], ["pattern", "top1"])
        _write_csv(seed_dir / "router_diagnostics.csv", [{"mean_gate_image": 0.25}], ["mean_gate_image"])
    summary = _load_script("summarize_temporal_router_s1_s4_v1.py")
    out_dir = tmp_path / "summary"
    assert summary.main(["--eval_dir", str(eval_dir), "--output_dir", str(out_dir)]) == 0
    for method in methods:
        assert (out_dir / f"{method}_top1_matrix.csv").exists()
        assert (out_dir / f"{method}_within3_matrix.csv").exists()
        assert (out_dir / f"{method}_mae_matrix.csv").exists()
    assert "8.6 自动分析" in (out_dir / "summary.md").read_text(encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
