import csv
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from kd_sensing.data.difficulty import DifficultyContext, apply_configured_difficulty
from kd_sensing.models.modular import AmrLiteMaskedGateCore, FeatureModLiteCore, ModularSequenceModel
from kd_sensing.registries import ENCODERS

import scripts.generate_scene31_baseline_pack as generate_baseline_pack
import scripts.generate_scene31_subset_reliability as generate_subset_reliability
import scripts.summarize_scene31_baseline_pack as summarize_baseline_pack
import scripts.summarize_scene31_subset_reference as summarize_subset_reference


@ENCODERS.register("baseline_pack_identity", force=True)
class BaselinePackIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[..., : self.output_dim]


def test_random_modality_dropout_subset_preserves_targets_and_logs_distribution() -> None:
    batch = {
        "image": torch.ones(256, 2, 3),
        "radar_ra": torch.ones(256, 2, 3),
        "gps": torch.ones(256, 2, 3),
        "lidar": torch.ones(256, 2, 3),
        "target_beam": torch.arange(256).reshape(256, 1),
        "sample_id": [str(index) for index in range(256)],
    }
    cfg = {
        "experiment": {"seed": 9},
        "training": {
            "random_modality_dropout": {
                "enabled": True,
                "mode": "random_nonempty_subset",
                "modalities": ["image", "radar", "gps", "lidar"],
                "ensure_at_least_one_modality": True,
            }
        },
    }

    result = apply_configured_difficulty(
        batch,
        cfg,
        DifficultyContext(stage="train", split="train", seed=9, epoch=0, step=0),
    ).batch

    assert torch.equal(result["target_beam"], batch["target_beam"])
    assert result["sample_id"] == batch["sample_id"]
    kept = torch.stack([result[f"{modality}_valid_mask"].any(dim=1) for modality in ("image", "radar", "gps", "lidar")], dim=1)
    assert kept.any(dim=1).all()
    missing_counts = {int(row["missing_count"]) for row in result["random_dropout_pattern_stats"]}
    assert {1, 2, 3} <= missing_counts


def test_pattern_balanced_dropout_uses_named_patterns_not_random_subset() -> None:
    batch = {
        "image": torch.ones(32, 2, 3),
        "radar_ra": torch.ones(32, 2, 3),
        "gps": torch.ones(32, 2, 3),
        "lidar": torch.ones(32, 2, 3),
        "target_beam": torch.zeros(32, 1, dtype=torch.long),
    }
    cfg = {
        "experiment": {"seed": 3},
        "training": {
            "random_modality_dropout": {
                "enabled": True,
                "mode": "pattern_balanced",
                "modalities": ["image", "radar", "lidar", "gps"],
                "patterns": ["full", "missing_gps", "radar_only"],
            }
        },
    }

    result = apply_configured_difficulty(
        batch,
        cfg,
        DifficultyContext(stage="train", split="train", seed=3, epoch=1, step=0),
    ).batch

    names = {row["pattern_or_available_set"] for row in result["random_dropout_pattern_stats"]}
    assert names <= {"full", "missing_gps", "radar_only"}
    assert not any(name.startswith("available:") for name in names)


def test_amr_lite_and_featuremod_forward_use_missing_masks_without_leaking_features() -> None:
    amr = _model({"type": "amr_lite", "d_model": 8, "hidden_dim": 8, "dropout": 0.0})
    batch = _modalities()
    altered = dict(batch)
    altered["image_batch"] = batch["image_batch"] + 1000.0
    missing_image = {"image_valid_mask": torch.zeros(2, 3, dtype=torch.bool)}

    out_a = amr(**batch, **missing_image)
    out_b = amr(**altered, **missing_image)

    assert isinstance(amr.representation_core, AmrLiteMaskedGateCore)
    assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-5)
    assert out_a["amr_lite_gate_stats"]
    assert amr.training_strategy_metadata()["consumes_missing_modality_metadata"] is True

    featuremod = _model({"type": "featuremod_lite", "d_model": 8, "adapter_dim": 4})
    out = featuremod(**batch, gps_valid_mask=torch.zeros(2, 3, dtype=torch.bool))
    assert isinstance(featuremod.representation_core, FeatureModLiteCore)
    assert out["logits"].shape == (2, 3, 6)
    assert featuremod.training_strategy_metadata()["representation_core"]["condition"] == "missing_modalities"


def test_modular_sequence_forward_accepts_fresh_eval_missing_mask() -> None:
    model = _model({"type": "amr_lite", "d_model": 8, "hidden_dim": 8, "dropout": 0.0})
    batch = _modalities()
    altered = dict(batch)
    altered["image_batch"] = batch["image_batch"] + 1000.0
    missing_image = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)

    out_a = model(**batch, missing_mask=missing_image)
    out_b = model(**altered, missing_mask=missing_image)

    assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-5)
    metadata = out_a["missing_modality_metadata"]
    assert metadata["missing_counts"]["image"] == 6
    assert metadata["missing_counts"]["radar"] == 0


def test_baseline_pack_generator_writes_expected_groups_and_configs(tmp_path: Path) -> None:
    out_dir = tmp_path / "configs"
    output_dir = tmp_path / "outputs"

    generate_baseline_pack.main(["--out_dir", str(out_dir), "--output_dir", str(output_dir), "--overwrite", "true"])

    rows = list(csv.DictReader((out_dir / "experiment_manifest.csv").open(newline="", encoding="utf-8")))
    run_names = {row["run_name"] for row in rows}
    assert "proto_randomdrop_subset_es40_seed1" in run_names
    assert "amr_lite_uniform_es40_seed1" in run_names
    assert "amber_lite_uniform_es40_seed1" in run_names
    assert "featuremod_lite_uniform_es40_seed1" in run_names

    amber_cfg = yaml.safe_load((out_dir / "amber_lite_uniform_es40_seed1.yaml").read_text(encoding="utf-8"))
    assert amber_cfg["model"]["primary"]["representation_core"]["num_layers"] == 1
    assert amber_cfg["training"]["random_modality_dropout"]["mode"] == "pattern_balanced"
    assert amber_cfg["loss"]["u_mask_beam_jepa"]["enabled"] is False


def test_subset_reliability_generator_writes_reference_combos(tmp_path: Path) -> None:
    out_dir = tmp_path / "configs"
    output_dir = tmp_path / "outputs"

    generate_subset_reliability.main(["--out_dir", str(out_dir), "--output_dir", str(output_dir), "--overwrite", "true"])

    rows = list(csv.DictReader((out_dir / "experiment_manifest.csv").open(newline="", encoding="utf-8")))
    run_names = {row["run_name"] for row in rows}
    assert "proto_randomdrop_subset_reliability_fusion_es40_seed1" in run_names
    assert "proto_randomdrop_subset_pattern_film_d8_es40_seed3" in run_names

    reliability_cfg = yaml.safe_load((out_dir / "proto_randomdrop_subset_reliability_fusion_es40_seed1.yaml").read_text(encoding="utf-8"))
    assert reliability_cfg["training"]["random_modality_dropout"]["mode"] == "random_nonempty_subset"
    assert reliability_cfg["model"]["primary"]["reliability_fusion"]["enabled"] is True
    assert reliability_cfg["model"]["primary"]["fusion_type"] == "weighted_sum"

    film_cfg = yaml.safe_load((out_dir / "proto_randomdrop_subset_pattern_film_d8_es40_seed3.yaml").read_text(encoding="utf-8"))
    assert film_cfg["model"]["primary"]["pattern_film"]["dim"] == 8
    assert film_cfg["model"]["primary"]["pattern_film"]["init_identity"] is True


def test_baseline_pack_runner_help() -> None:
    result = subprocess.run(
        ["bash", "scripts/run_scene31_baseline_pack.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--overwrite-eval" in result.stdout
    assert "all_core" in result.stdout


def test_subset_reliability_runner_help() -> None:
    result = subprocess.run(
        ["bash", "scripts/run_scene31_subset_reliability.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "eval_modular_lite_maskfix" in result.stdout
    assert "--max-parallel" in result.stdout
    assert "subset_film" in result.stdout


def test_baseline_pack_summary_fixture(tmp_path: Path) -> None:
    root = tmp_path / "root"
    eval_dir = root / "fresh_eval" / "proto_randomdrop_subset_es40_seed1"
    eval_dir.mkdir(parents=True)
    _write_metrics(eval_dir / "apples_to_apples_metrics.csv", "proto_randomdrop_subset_es40_seed1", 0.2)
    (eval_dir / "checkpoint_manifest.json").write_text('{"max_batches": null}', encoding="utf-8")

    uniform_eval = root / "fresh_eval" / "proto_sampler_uniform_es40_seed1"
    uniform_eval.mkdir(parents=True)
    _write_metrics(uniform_eval / "apples_to_apples_metrics.csv", "proto_sampler_uniform_es40_seed1", 0.3)
    (uniform_eval / "checkpoint_manifest.json").write_text('{"max_batches": null}', encoding="utf-8")

    out = tmp_path / "summary"
    summarize_baseline_pack.main(["--root", str(root), "--uniform-root", "", "--out", str(out)])

    method_rows = list(csv.DictReader((out / "baseline_method_mean_std.csv").open(newline="", encoding="utf-8")))
    by_method = {row["method"]: row for row in method_rows}
    assert by_method["proto_randomdrop_subset_es40"]["training_strategy"] == "randomdrop_subset"
    assert by_method["proto_sampler_uniform_es40"]["training_strategy"] == "pattern_balanced_uniform"
    assert (out / "baseline_conclusion.txt").exists()
    assert (out / "backbone_training_comparison.csv").exists()


def test_subset_reference_summary_uses_subset_and_excludes_suspect(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for seed in (1, 2, 3):
        eval_dir = root / "fresh_eval" / f"proto_randomdrop_subset_es40_seed{seed}"
        eval_dir.mkdir(parents=True)
        _write_metrics(eval_dir / "apples_to_apples_metrics.csv", f"proto_randomdrop_subset_es40_seed{seed}", 0.30)
        (eval_dir / "checkpoint_manifest.json").write_text('{"max_batches": null}', encoding="utf-8")

    uniform_eval = root / "fresh_eval" / "proto_sampler_uniform_es40_seed1"
    uniform_eval.mkdir(parents=True)
    _write_metrics(uniform_eval / "apples_to_apples_metrics.csv", "proto_sampler_uniform_es40_seed1", 0.20)
    (uniform_eval / "checkpoint_manifest.json").write_text('{"max_batches": null}', encoding="utf-8")

    amber_eval = root / "fresh_eval_maskfix" / "amber_lite_natural_es40_seed1"
    amber_eval.mkdir(parents=True)
    metrics_path = amber_eval / "apples_to_apples_metrics.csv"
    _write_metrics(metrics_path, "amber_lite_natural_es40_seed1", 0.50)
    _append_mask_suspect(metrics_path)

    out = tmp_path / "subset_reference"
    summarize_subset_reference.main(["--baseline-root", str(root), "--out", str(out)])

    delta_rows = list(csv.DictReader((out / "delta_vs_randomdrop_subset.csv").open(newline="", encoding="utf-8")))
    uniform = next(row for row in delta_rows if row["method"] == "proto_sampler_uniform_es40")
    assert float(uniform["delta_avg_missing_top1_vs_subset"]) < 0
    method_rows = list(csv.DictReader((out / "subset_reference_method_mean_std.csv").open(newline="", encoding="utf-8")))
    assert "amber_lite_natural_es40" not in {row["method"] for row in method_rows}
    assert "amber_lite_natural_es40_seed1" in (out / "suspect_modular_results.md").read_text(encoding="utf-8")


def _model(core: dict[str, object]) -> ModularSequenceModel:
    modalities = ["image", "radar", "gps", "lidar"]
    return ModularSequenceModel(
        modalities=modalities,
        encoders={modality: {"type": "baseline_pack_identity", "output_dim": 8} for modality in modalities},
        projectors={modality: {"type": "identity"} for modality in modalities},
        representation_core=core,
        heads={"beam": {"type": "beam_head"}},
        feature_size=8,
        d_model=8,
        num_classes=6,
        num_pred=1,
    )


def _modalities() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.randn(2, 3, 8),
        "radar_batch": torch.randn(2, 3, 8),
        "gps_batch": torch.randn(2, 3, 8),
        "lidar_batch": torch.randn(2, 3, 8),
    }


def _write_metrics(path: Path, run_name: str, base: float) -> None:
    rows = [
        ("full", base + 0.10, base + 0.20, base + 0.30, 0.6, 4.0),
        ("missing_gps", base, base + 0.10, base + 0.20, 0.5, 5.0),
        ("missing_radar", base + 0.01, base + 0.11, base + 0.21, 0.51, 4.9),
        ("radar_only", base - 0.02, base + 0.08, base + 0.18, 0.48, 5.2),
        ("lidar_only", base - 0.03, base + 0.07, base + 0.17, 0.47, 5.3),
        ("missing_gps_image", base - 0.01, base + 0.09, base + 0.19, 0.49, 5.1),
        ("gps_only", base - 0.04, base + 0.06, base + 0.16, 0.46, 5.4),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_name", "pattern", "top1", "top3", "top5", "within_3", "mae", "status"])
        writer.writeheader()
        for pattern, top1, top3, top5, within, mae in rows:
            writer.writerow(
                {
                    "run_name": run_name,
                    "pattern": pattern,
                    "top1": top1,
                    "top3": top3,
                    "top5": top5,
                    "within_3": within,
                    "mae": mae,
                    "status": "ok",
                }
            )


def _append_mask_suspect(path: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    fields = list(rows[0])
    if "mask_suspect" not in fields:
        fields.append("mask_suspect")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["mask_suspect"] = "true"
            writer.writerow(row)
