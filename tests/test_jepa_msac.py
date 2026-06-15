from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd_sensing.baselines.jepa_msac.config import validate_jepa_msac_workflow_config
from kd_sensing.baselines.jepa_msac.data import (
    JepaMsacWindowProtocol,
    assemble_sliding_window_samples,
    build_scenario32_manifest,
    map_rf_history,
)
from kd_sensing.baselines.jepa_msac.fixture import make_synthetic_jepa_msac_batch
from kd_sensing.baselines.jepa_msac.metrics import evaluate_jepa_msac_predictions, representation_quality_summary
from kd_sensing.baselines.jepa_msac.report import build_ablation_row, write_ablation_manifest, write_report
from kd_sensing.baselines.jepa_msac.workflow import run_jepa_msac
from kd_sensing.config import load_config
from kd_sensing.losses.jepa_msac import jepa_msac_stage2_losses, masked_latent_smooth_l1_loss
from kd_sensing.models.jepa_msac import (
    JepaMsacModel,
    TemporalBlockMaskSampler,
    build_frozen_jepa_msac_from_checkpoint,
    stage2_optimizer_parameters,
)
from kd_sensing.registries import MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "pretraining" / "jepa_msac_s32_smoke.yaml"


def _model_cfg(depth: int = 1) -> dict:
    return {
        "type": "jepa_msac",
        "latent_dim": 16,
        "t_hist": 8,
        "t_pred": 5,
        "num_beams": 64,
        "image_channels": 3,
        "radar_channels": 1,
        "lidar_channels": 1,
        "gps_input_size": 2,
        "rf_input_size": 64,
        "image_tokens": 9,
        "radar_tokens": 16,
        "lidar_tokens": 16,
        "max_frames": 13,
        "max_tokens_per_frame": 16,
        "transformer_depth": depth,
        "num_heads": 4,
        "ema_momentum": 0.5,
        "mask_ratio": 0.5,
        "mask_pattern": "random",
        "mask_seed": 11,
        "localization_guidance": True,
    }


def _model_inputs(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key.endswith("_batch") or key == "rf_history"}


def test_jepa_msac_config_load_fixture_and_light_import_boundaries(tmp_path: Path):
    cfg = load_config(SMOKE_CONFIG)
    metadata = validate_jepa_msac_workflow_config(cfg)
    batch = make_synthetic_jepa_msac_batch(batch_size=2, image_size=16)

    assert cfg["workflow"]["family"] == "jepa_msac"
    assert cfg["data"]["dataset"]["scene"] == 32
    assert cfg["training"]["early_stopping_metric"] == "val_jepa_msac_loss"
    assert metadata["t_hist"] == 8
    assert metadata["t_pred"] == 5
    assert metadata["num_beams"] == 64
    assert batch["image_batch"].shape == (2, 13, 3, 16, 16)
    assert batch["radar_batch"].shape == (2, 13, 1, 16, 16)
    assert batch["lidar_batch"].shape == (2, 13, 1, 16, 16)
    assert batch["gps_batch"].shape == (2, 13, 2)
    assert batch["rf_history"].shape == (2, 13, 64)
    assert not (ROOT / "jepa_msac_report.json").exists()
    report = {"stage": "smoke", "claim_status": "mock/smoke", "metrics": {}, "caveats": ["synthetic only"]}
    paths = write_report(report, tmp_path)
    assert all(Path(path).is_file() for path in paths.values())


def test_manifest_audit_sliding_window_and_rf_mapping_do_not_make_rf_canonical(tmp_path: Path):
    missing_manifest = build_scenario32_manifest(csv_path=tmp_path / "missing.csv", dry_run=True)
    assert missing_manifest.status == "blocked"
    assert missing_manifest.blocked_reasons[0]["field"] == "csv_source"
    assert "Provide a local Scenario 32 CSV" in missing_manifest.blocked_reasons[0]["fix_hint"]

    csv_path = tmp_path / "scenario32.csv"
    csv_path.write_text(
        "image,radar,lidar,gps,beam_power,beam_index,location,rssi\n" + "\n".join("i,r,l,g,p,1,xy,-42" for _ in range(14)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = build_scenario32_manifest(csv_path=csv_path, output_path=manifest_path, write=True, dry_run=True)
    assert manifest.status == "dry_run_ready"
    assert manifest.sample_count == 2
    assert manifest_path.is_file()

    rows = [
        {"image": f"i{idx}", "radar": f"r{idx}", "lidar": f"l{idx}", "gps": idx, "beam_power": idx, "beam_index": idx % 64, "location": idx}
        for idx in range(14)
    ]
    samples = assemble_sliding_window_samples(rows, protocol=JepaMsacWindowProtocol())
    assert len(samples) == 2
    assert len(samples[0]["history"]["rf_power_history"]) == 8
    assert len(samples[0]["targets"]["future_beam"]) == 5

    rf, metadata = map_rf_history({"beam_power_history": torch.randn(2, 8, 64)}, source_key="beam_power_history")
    assert rf.shape == (2, 8, 64)
    assert metadata["paper_modality"] == "RF"
    cfg_modalities = ["image", "radar", "gps", "lidar", "mmwave"]
    assert "rf" not in cfg_modalities
    assert "mmwave" in cfg_modalities

    with pytest.raises(ValueError, match="workflow-local beam-power history"):
        load_config(SMOKE_CONFIG, ["model.modalities=[image,rf]"])


def test_tokenizers_forward_mask_loss_backward_and_ema_update():
    import_default_components()
    model = MODELS.build(_model_cfg(depth=1))
    batch = make_synthetic_jepa_msac_batch(batch_size=2, image_size=16)

    output = model(**_model_inputs(batch), jepa_epoch=1, jepa_step=2)

    assert output["predicted_target_latent"].shape == output["target_latent"].shape
    assert output["concat_index_metadata"]["token_counts"] == {"image": 9, "radar": 16, "lidar": 16, "gps": 1, "rf": 1}
    assert output["context_mask"].shape[0] == 13 * (9 + 16 + 16 + 1 + 1)
    assert not torch.any(output["context_mask"] & output["target_mask"])
    assert output["target_latent"].requires_grad is False
    assert model.training_strategy_metadata()["pretraining_metric"] == "val_jepa_msac_loss"

    loss, diagnostics = masked_latent_smooth_l1_loss(
        output["predicted_target_latent"],
        output["target_latent"],
        output["loss_mask"],
        ema_momentum=model.ema_momentum,
    )
    loss.backward()
    assert all(param.grad is None for param in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.5)
    model.update_target_encoder_ema()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)
    assert diagnostics["target_token_count"] == output["predicted_target_latent"].shape[1] * output["predicted_target_latent"].shape[0]


def test_mask_sampler_is_per_modality_contiguous_and_position_errors_are_clear():
    model = JepaMsacModel(**{key: value for key, value in _model_cfg(depth=0).items() if key != "type"})
    batch = make_synthetic_jepa_msac_batch(batch_size=1, image_size=16)
    schema = model.tokenize(**_model_inputs(batch))
    sampler = TemporalBlockMaskSampler(rho=0.5, pattern="random", seed=3)
    sample = sampler.sample(time_index=schema.time_index, modality_index=schema.modality_index, total_frames=schema.total_frames)

    for modality_id in torch.unique(schema.modality_index):
        frames = sorted(torch.unique(schema.time_index[sample.mask & (schema.modality_index == modality_id)]).tolist())
        assert frames == list(range(frames[0], frames[-1] + 1))
    assert sample.diagnostics["target_token_count"] == int(sample.mask.sum())

    too_long = make_synthetic_jepa_msac_batch(batch_size=1, t_hist=8, t_pred=6, image_size=16)
    with pytest.raises(ValueError, match="time positional embedding"):
        model(**_model_inputs(too_long))


def test_stage2_freeze_future_latent_heads_losses_and_checkpoint_metadata(tmp_path: Path):
    source = JepaMsacModel(**{key: value for key, value in _model_cfg(depth=1).items() if key != "type"})
    checkpoint = tmp_path / "stage1.pth"
    torch.save(source.state_dict(), checkpoint)
    model, metadata = build_frozen_jepa_msac_from_checkpoint(
        checkpoint,
        model_config={key: value for key, value in _model_cfg(depth=1).items() if key != "type"},
    )
    batch = make_synthetic_jepa_msac_batch(batch_size=2, image_size=16)

    output = model(stage="heads", **_model_inputs(batch))
    losses = jepa_msac_stage2_losses(output, batch["targets"])

    assert metadata["checkpoint_path"] == str(checkpoint)
    assert metadata["trainable_parameter_count"] > 0
    assert all(param.requires_grad for param in stage2_optimizer_parameters(model))
    assert all("head" in name or "probe" in name for name in metadata["trainable_parameter_names"])
    assert output["S_pred"].shape == (2, 5, 16)
    assert output["predicted_location"].shape == (2, 5, 2)
    assert output["beam_logits"].shape == (2, 5, 64)
    assert output["rssi_profile"].shape == (2, 5, 64)
    assert losses["total"].requires_grad
    assert output["stage2_metadata"]["localization_guidance"] is True


def test_metrics_report_ablation_and_workflow_cli_dispatch(tmp_path: Path):
    model = JepaMsacModel(**{key: value for key, value in _model_cfg(depth=0).items() if key != "type"})
    batch = make_synthetic_jepa_msac_batch(batch_size=2, image_size=16)
    output = model(stage="heads", **_model_inputs(batch))
    metrics = evaluate_jepa_msac_predictions(
        output,
        batch["targets"],
        beam_power_reference=batch["targets"]["future_rssi_profile"],
        representation_latents=output["S_pred"],
    )
    no_ref = evaluate_jepa_msac_predictions(output, batch["targets"], representation_latents=output["S_pred"])
    rlda = representation_quality_summary(output["S_pred"])
    report_paths = write_report({"stage": "evaluate", "claim_status": "mock/smoke", "metrics": metrics["task_metrics"]}, tmp_path)
    ablation = write_ablation_manifest(
        [
            build_ablation_row(config_path=str(SMOKE_CONFIG), checkpoint_provenance=None, run_status="planned", claim_status="unverified"),
            build_ablation_row(config_path=str(SMOKE_CONFIG), checkpoint_provenance="local", run_status="complete", metrics_path="metrics.json"),
        ],
        tmp_path,
    )
    run_report = run_jepa_msac(config_path=SMOKE_CONFIG, stage="report", dry_run=True, output_dir=tmp_path / "runner", write=True)

    assert metrics["task_metrics"]["ADE"]["available"] is True
    assert metrics["task_metrics"]["Top-3"]["available"] is True
    assert metrics["task_metrics"]["L1-RSRP diff"]["available"] is True
    assert no_ref["task_metrics"]["L1-RSRP diff"]["available"] is False
    assert rlda["RLDA"]["available"] is False
    assert all(Path(path).is_file() for path in report_paths.values())
    assert ablation["row_count"] == 2
    assert ablation["result_row_count"] == 1
    assert run_report["stage"] == "report"
    assert run_report["manifest"]["status"] == "blocked"
