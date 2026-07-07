import csv
import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.engine.pcpg_radar_balance import hard_subset_sample_weight, soft_static_hard_subset_weight
from kd_sensing.losses.u_mask_beam_jepa import u_mask_beam_jepa_config, u_mask_beam_jepa_loss
from kd_sensing.registries import ENCODERS, MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("final_c2_test_encoder", force=True)
class FinalC2TestEncoder(nn.Module):
    def __init__(self, output_dim: int = 16, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = batch.shape[:2]
        pooled = batch.float().reshape(batch_size, seq_len, -1).mean(dim=-1, keepdim=True)
        return pooled.expand(batch_size, seq_len, self.output_dim)


def cfg(**overrides):
    payload = {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar", "lidar", "gps"],
        "d_model": 16,
        "num_classes": 8,
        "num_pred": 1,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "use_jepa_loss": False,
        "encoders": {
            "image": {"type": "final_c2_test_encoder", "output_dim": 16},
            "radar": {"type": "final_c2_test_encoder", "output_dim": 16},
            "lidar": {"type": "final_c2_test_encoder", "output_dim": 16},
            "gps": {"type": "final_c2_test_encoder", "output_dim": 16},
        },
    }
    payload.update(overrides)
    return payload


def batch(batch_size: int = 2):
    return {
        "image_batch": torch.randn(batch_size, 2, 3, 8, 8),
        "radar_batch": torch.randn(batch_size, 2, 2, 6, 6),
        "lidar_batch": torch.randn(batch_size, 2, 3, 6, 6),
        "gps_batch": torch.randn(batch_size, 2, 3),
    }


def build_model(**overrides):
    import_default_components()
    return MODELS.build(cfg(**overrides))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_router_feature_ablation_flags_forward_no_nan_and_diagnostics():
    torch.manual_seed(1)
    model = build_model(
        fusion_type="supervised_router",
        router_supervision="none",
        router_use_pattern_features=False,
        router_use_reliability_features=False,
        router_use_prototype_margin=False,
        head_type="prototype",
    )
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool)

    output = model(**batch(), missing_mask=mask)
    metadata = output["metadata"]

    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["supervised_router_gate_weights"]).all()
    assert torch.all(output["supervised_router_gate_weights"][~mask] == 0)
    assert output["router_use_pattern_features"] is False
    assert output["router_use_reliability_features"] is False
    assert output["router_use_prototype_margin"] is False
    assert output["router_pattern_feature_fallback"] == pytest.approx(1.0)
    assert metadata["router_use_pattern_features"] is False
    assert metadata["router_use_reliability_features"] is False
    assert metadata["router_use_prototype_margin"] is False


def test_prototype_ablation_config_and_classifier_head_fallback():
    torch.manual_seed(2)
    model = build_model(
        fusion_type="supervised_router",
        head_type="classifier",
        router_use_prototype_margin=True,
        use_beam_prototype_alignment=False,
    )
    output = model(**batch(), missing_mask=torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.bool))
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[1], [3]]),
        use_teacher=False,
        use_jepa_loss=False,
        prototype_bank=model.prototype_bank,
        use_beam_prototype_alignment=False,
        lambda_proto=0.2,
        lambda_modality_proto=0.1,
    )
    resolved = u_mask_beam_jepa_config(
        {
            "model": {"primary": {"head_type": "classifier", "use_beam_prototype_alignment": False}},
            "training": {
                "use_beam_prototype_alignment": False,
                "beam_proto_align_weight": 0.0,
                "use_modality_prototype_loss": False,
                "modality_proto_weight": 0.0,
            },
            "loss": {"u_mask_beam_jepa": {"enabled": True}},
        }
    )
    onehot = u_mask_beam_jepa_config(
        {
            "training": {"use_circular_soft_targets": False, "use_gaussian_beam_targets": False},
            "loss": {"u_mask_beam_jepa": {"enabled": True}},
        }
    )

    assert torch.isfinite(output["logits"]).all()
    assert output["head_type"] == "classifier"
    assert output["prototype_margin_enabled"] is False
    assert output["prototype_margin_fallback"] == pytest.approx(1.0)
    assert not any(key.startswith("loss/prototype") for key in result["diagnostics"])
    assert resolved["use_beam_prototype_alignment"] is False
    assert resolved["lambda_proto"] == pytest.approx(0.0)
    assert resolved["lambda_modality_proto"] == pytest.approx(0.0)
    assert onehot["proto_target_type"] == "onehot"
    assert onehot["beam_label_circular"] is False
    assert onehot["circular_beam_distance"] is False


def test_average_fusion_masks_unavailable_modalities_and_single_modality():
    torch.manual_seed(3)
    model = build_model(fusion_type="average", head_type="prototype")
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)

    output = model(**batch(), missing_mask=mask)
    stats = model._unimodal_branch_stats(output["input_features"])
    weights = mask.to(dtype=output["input_features"].dtype) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    expected = (stats["unimodal_logits"] * weights.unsqueeze(-1)).sum(dim=1)

    assert torch.isfinite(output["logits"]).all()
    assert torch.allclose(output["fused_logits"], expected, atol=1e-6)
    assert torch.allclose(output["fused_logits"][1], stats["unimodal_logits"][1, 1], atol=1e-6)
    assert output["average_fusion"] == pytest.approx(1.0)


@pytest.mark.parametrize("fusion", ["weighted_sum", "raw_conf_gate", "bprr", "pcpg", "supervised_router"])
def test_final_fusion_baselines_forward_no_nan(fusion: str):
    torch.manual_seed(4)
    extra = {"bprr_calibration": "temperature"} if fusion == "bprr" else {}
    model = build_model(fusion_type=fusion, head_type="prototype", **extra)
    output = model(**batch(), missing_mask=torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool))

    assert torch.isfinite(output["logits"]).all()
    assert output["logits"].shape == (2, 1, 8)


def test_soft_static_hard_subset_order_unknown_and_static_difference():
    radar_only = soft_static_hard_subset_weight("radar_only")
    drop3 = soft_static_hard_subset_weight("image_only")
    drop2 = soft_static_hard_subset_weight("drop2")
    full = soft_static_hard_subset_weight("full")

    assert radar_only > drop3 > drop2 > full
    assert soft_static_hard_subset_weight("unknown_pattern") == pytest.approx(1.0)
    assert hard_subset_sample_weight("drop2", mode="soft_static") == pytest.approx(1.15)
    assert hard_subset_sample_weight("drop2", mode="static", focus=["radar_only"]) == pytest.approx(1.0)


def test_final_launcher_dry_run_gpu_plan_manifest_and_filter(tmp_path: Path, monkeypatch):
    launcher = load_script("launch_final_c2_ablation_v1.py")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    jobs = launcher.plan_jobs(
        experiments=list(launcher.experiment_specs()),
        main_seeds=[1, 2, 3, 4, 5],
        ablation_seeds=[1, 2, 3],
        negative_seeds=[1, 2, 3],
        gpus=[str(i) for i in range(8)],
        per_gpu=1,
        output_root=str(tmp_path / "out"),
        baseline_roots=str(tmp_path / "baseline"),
        base_config="missing_seed{seed}.yaml",
    )
    launcher.write_generated_configs(jobs, dry_run=True)
    manifest = launcher.write_manifest(jobs, str(tmp_path / "out"))
    filtered = launcher.plan_jobs(
        experiments=[launcher.canonical_experiment(item) for item in ["a0", "b0", "c0"]],
        main_seeds=[1, 2, 3, 4, 5],
        ablation_seeds=[1, 2, 3],
        negative_seeds=[1, 2, 3],
        gpus=[str(i) for i in range(8)],
        per_gpu=1,
        output_root=str(tmp_path / "filter"),
        baseline_roots=str(tmp_path / "baseline"),
        base_config="missing_seed{seed}.yaml",
    )

    assert len(jobs) == 67
    assert {job["gpu"] for job in jobs} <= {str(i) for i in range(8)}
    for start in range(0, len(jobs), 8):
        wave = jobs[start : start + 8]
        assert len(wave) <= 8
        assert max(Counter(job["gpu"] for job in wave).values()) <= 1
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 67
    assert {"experiment", "seed", "gpu", "cmd", "status", "log_path", "output_dir"} <= set(rows[0])
    assert len(filtered) == 11


def test_final_summary_parser_reads_fake_metrics_baselines_and_deltas(tmp_path: Path):
    summary = load_script("summarize_final_c2_ablation_v1.py")
    root = tmp_path / "final"
    baseline = tmp_path / "baseline"
    write_eval(root / "a0_c2_full_main" / "eval" / "a0_c2_full_main_seed1_missing_patterns.csv", "a0_c2_full_main", 1, top1=0.60, router=True)
    write_eval(root / "b0_no_router_supervision" / "eval" / "b0_no_router_supervision_seed1_missing_patterns.csv", "b0_no_router_supervision", 1, top1=0.50, router=True)
    write_config(root / "a0_c2_full_main" / "seed1" / "final_config.yaml", fusion="supervised_router", head_type="prototype")
    write_config(root / "b0_no_router_supervision" / "seed1" / "final_config.yaml", fusion="supervised_router", head_type="prototype")
    write_eval(baseline / "e5_pcpg_low_encoder_lr_seed1" / "eval_matrix.csv", "e5_pcpg_low_encoder_lr", 1, top1=0.45)

    assert summary.main(["--root", str(root), "--baseline_roots", str(baseline)]) == 0

    for name in (
        "summary.csv",
        "summary.md",
        "main_results.csv",
        "ablation_router.csv",
        "ablation_prototype.csv",
        "ablation_fusion.csv",
        "ablation_pattern_weighting.csv",
        "router_diagnostics.csv",
        "pattern_metrics.csv",
    ):
        assert (root / name).exists()
    rows = {row["experiment"]: row for row in csv.DictReader((root / "summary.csv").open("r", encoding="utf-8", newline=""))}
    assert {"a0_c2_full_main", "b0_no_router_supervision", "e5_pcpg_low_encoder_lr"} <= set(rows)
    assert float(rows["a0_c2_full_main"]["delta_avg_missing"]) == pytest.approx(0.0)
    assert float(rows["b0_no_router_supervision"]["delta_avg_missing"]) == pytest.approx(-0.10)
    assert "7.7 最终推荐结论" in (root / "summary.md").read_text(encoding="utf-8")


def write_eval(path: Path, experiment: str, seed: int, *, top1: float, router: bool = False) -> None:
    run_name = f"{experiment}/seed{seed}"
    rows = [
        {"run_name": run_name, "seed": seed, "pattern": "full", "mask": "1,1,1,1", "top1": top1, "within_3": 0.80, "mae": 2.0},
        {"run_name": run_name, "seed": seed, "pattern": "missing_image", "mask": "0,1,1,1", "top1": top1 - 0.10, "within_3": 0.70, "mae": 3.0},
        {"run_name": run_name, "seed": seed, "pattern": "missing_image_lidar", "mask": "0,1,0,1", "top1": top1 - 0.15, "within_3": 0.65, "mae": 3.2},
        {"run_name": run_name, "seed": seed, "pattern": "radar_only", "mask": "0,1,0,0", "top1": top1 - 0.20, "within_3": 0.55, "mae": 4.0},
        {"run_name": run_name, "seed": seed, "pattern": "image_only", "mask": "1,0,0,0", "top1": top1 - 0.18, "within_3": 0.56, "mae": 3.9},
        {"run_name": run_name, "seed": seed, "pattern": "lidar_only", "mask": "0,0,1,0", "top1": top1 - 0.19, "within_3": 0.55, "mae": 4.0},
        {"run_name": run_name, "seed": seed, "pattern": "gps_only", "mask": "0,0,0,1", "top1": top1 - 0.21, "within_3": 0.54, "mae": 4.1},
        {"run_name": run_name, "seed": seed, "pattern": "avg_missing", "mask": "aggregate", "top1": top1 - 0.12, "within_3": 0.66, "mae": 3.3},
    ]
    if router:
        for row in rows:
            row.update(
                {
                    "mean_gate_image": 0.2,
                    "mean_gate_lidar": 0.2,
                    "mean_gate_radar": 0.4,
                    "mean_gate_gps": 0.2,
                    "gate_entropy": 1.1,
                    "router_oracle_acc": 0.7,
                    "router_oracle_acc_missing_image": 0.75 if row["pattern"] == "missing_image" else "",
                    "router_oracle_acc_drop2": 0.6 if row["pattern"] == "missing_image_lidar" else "",
                    "oracle_target_radar_rate": 0.5,
                    "radar_gate_missing_image": 0.45 if row["pattern"] == "missing_image" else "",
                    "radar_gate_drop2": 0.35 if row["pattern"] == "missing_image_lidar" else "",
                }
            )
    write_csv(path, rows)


def write_config(path: Path, *, fusion: str, head_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "model:",
                "  primary:",
                f"    fusion_type: {fusion}",
                f"    head_type: {head_type}",
                "    router_use_pattern_features: true",
                "    router_use_reliability_features: true",
                "    router_use_prototype_margin: true",
                "training:",
                "  use_beam_prototype_alignment: true",
                "  use_modality_prototype_loss: true",
                "  use_circular_soft_targets: true",
                "loss:",
                "  router_supervision: oracle",
                "  hard_subset_weighting:",
                "    enabled: true",
                "    mode: soft_static",
                "  use_jepa: false",
                "  branch_aux_loss: false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
