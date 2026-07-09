import csv
import importlib.util
import json
from pathlib import Path

import torch

from kd_sensing.data.temporal_missing import (
    apply_modality_temporal_mask_to_batch,
    generate_fixed_eval_mask_cache,
    sample_stratified_modality_temporal_mask,
)
from kd_sensing.registries import REPRESENTATION_CORES, import_default_components


ROOT = Path(__file__).resolve().parents[1]


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
