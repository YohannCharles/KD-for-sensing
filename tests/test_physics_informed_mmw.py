from pathlib import Path

import pytest
import torch

from kd_sensing.config import load_config
from kd_sensing.data.datasets.mmw_physics_adapter import build_mmw_physics_targets, physics_shape_summary
from kd_sensing.engine.batch import prepare_csi_inputs
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.artifacts import final_config_with_runtime
from kd_sensing.engine.optim import build_model
from kd_sensing.evaluation.physics_metrics import grouped_report, normalized_beamforming_gain
from kd_sensing.losses.physics_informed import PhysicsInformedBeamLoss
from kd_sensing.models.physics.beam_scoring import beam_logits_from_channel
from kd_sensing.models.physics.channel_synthesizer import synthesize_ula_channel
from kd_sensing.registries import MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


def test_mmw_physics_adapter_normalizes_targets_and_missing_fields():
    sample = {
        "sample_id": "s0",
        "csi": torch.randn(1, 8, 4, 2),
        "beamspace_power_label": torch.rand(1, 64),
        "path_params": {
            "AoD": torch.tensor([0.3, 0.1]),
            "AoA": torch.tensor([0.0, 0.2]),
            "tau": torch.tensor([1e-9, 2e-9]),
            "complex_gain": torch.tensor([[0.1, 0.2], [2.0, 0.0]]),
            "valid": torch.tensor([1, 1]),
        },
        "metadata": {"condition": "sunny", "town": "Town10", "scenario": "town10"},
    }

    targets = build_mmw_physics_targets(sample)
    summary = physics_shape_summary({**sample, "physics_targets": targets})

    assert targets["csi_target"].shape == (1, 8, 4, 2)
    assert targets["csi_input_valid"].item() is False
    assert "csi_input" not in targets
    assert targets["beamspace_power"].shape == (1, 64)
    assert targets["path_params"].shape == (2, 6)
    assert targets["path_params"][0, 3].abs() > targets["path_params"][1, 3].abs()
    assert targets["metadata"]["field_mapping"]["AoD"] == "aod"
    assert summary["num_subcarriers"] == 8
    assert summary["num_paths"] == 2

    missing = build_mmw_physics_targets({"target_beam": torch.tensor([1])})
    assert missing["csi_target_valid"].item() is False
    assert "path_params" in missing["unavailable_reasons"]
    with pytest.raises(RuntimeError, match="Required MMW physics field"):
        build_mmw_physics_targets({"sample_id": "bad"}, {"required_fields": ["csi"]})


def test_csi_input_modes_are_leakage_safe_by_default_and_oracle_guarded():
    csi = torch.arange(7 * 8 * 4 * 2, dtype=torch.float32).view(7, 8, 4, 2)
    partial = build_mmw_physics_targets(
        {"csi": csi},
        {
            "use_csi_input": True,
            "csi_input_mode": "partial",
            "partial_subcarrier_ratio": 0.5,
            "partial_antenna_ratio": 0.5,
            "num_pred": 1,
        },
    )
    assert partial["csi_target"].shape == (1, 8, 4, 2)
    assert partial["csi_input"].shape == partial["csi_target"].shape
    assert torch.count_nonzero(partial["csi_input"][..., 4:, :, :]) == 0
    assert torch.count_nonzero(partial["csi_input"][..., :, 2:, :]) == 0

    sparse = build_mmw_physics_targets(
        {"csi": csi},
        {
            "use_csi_input": True,
            "csi_input_mode": "sparse_pilot",
            "pilot_pattern": "grid",
            "pilot_subcarrier_stride": 2,
            "pilot_antenna_stride": 2,
            "num_pred": 1,
        },
    )
    mask = sparse["csi_observation_mask"]
    assert sparse["csi_input"].shape == sparse["csi_target"].shape
    assert mask.shape == sparse["csi_target"].shape[:-1]
    assert torch.equal(sparse["csi_input"], sparse["csi_target"] * mask.unsqueeze(-1))
    assert torch.count_nonzero(sparse["csi_input"][..., 1::2, :, :]) == 0
    assert torch.count_nonzero(sparse["csi_input"][..., :, 1::2, :]) == 0
    assert sparse["metadata"]["csi_input"]["observed_fraction"] == pytest.approx(0.25)

    history = build_mmw_physics_targets(
        {"csi": csi},
        {"use_csi_input": True, "csi_input_mode": "history", "history_len": 3, "num_pred": 1},
    )
    assert history["csi_input"].shape == (3, 8, 4, 2)
    assert torch.equal(history["csi_input"], csi[3:6])
    assert not torch.equal(history["csi_input"][-1], history["csi_target"][-1])

    with pytest.raises(RuntimeError, match="oracle"):
        build_mmw_physics_targets({"csi": csi}, {"use_csi_input": True, "csi_input_mode": "oracle_full"})
    oracle = build_mmw_physics_targets(
        {"csi": csi},
        {"use_csi_input": True, "csi_input_mode": "oracle_full", "allow_oracle_full_csi_input": True},
    )
    assert torch.equal(oracle["csi_input"], oracle["csi_target"])

    with pytest.raises(ValueError, match="csi_target"):
        prepare_csi_inputs(
            {"csi_target": torch.randn(2, 1, 8, 4, 2)},
            seq_length=4,
            num_pred=1,
            device=torch.device("cpu"),
        )
    safe_input = prepare_csi_inputs(
        {"csi_input": torch.randn(2, 3, 8, 4, 2)},
        seq_length=3,
        num_pred=1,
        device=torch.device("cpu"),
    )
    assert safe_input.shape == (2, 3, 8, 4, 2)


def test_complex_physics_helpers_are_finite_and_differentiable():
    path = torch.randn(2, 1, 3, 5, requires_grad=True)
    path_mask = torch.tensor([[[True, False, True]], [[True, True, False]]])
    h_hat = synthesize_ula_channel(path, num_subcarriers=5, num_antennas=4, path_mask=path_mask)
    logits, metadata = beam_logits_from_channel(h_hat, num_beams=8)
    loss = logits.mean() + h_hat.abs().mean()
    loss.backward()

    assert h_hat.shape == (2, 1, 5, 4)
    assert logits.shape == (2, 1, 8)
    assert metadata["codebook_source"] == "ula_dft_fallback"
    assert torch.isfinite(path.grad).all()


def test_pinn_model_registry_forward_adapt_and_metadata():
    import_default_components()
    model = MODELS.build(
        {
            "type": "pinn_multimodal_beam",
            "modalities": ["image"],
            "num_classes": 16,
            "num_pred": 2,
            "num_subcarriers": 5,
            "num_antennas": 4,
            "num_paths": 2,
        }
    )
    output = model(image_batch=torch.randn(3, 4, 3, 8, 8))
    adapted = adapt_model_output(output)
    metadata = model.training_strategy_metadata()

    assert adapted.logits.shape == (3, 2, 16)
    assert adapted.diagnostics["h_hat"].shape == (3, 2, 5, 4)
    assert adapted.diagnostics["path_hat"].shape == (3, 2, 2, 5)
    assert metadata["registry_name"] == "pinn_multimodal_beam"
    assert metadata["architecture_category"] == "whole_model_exception"


def test_pinn_paper_frontend_forward_loss_adapt_and_metadata():
    import_default_components()
    model = MODELS.build(
        {
            "type": "pinn_multimodal_beam",
            "modalities": ["image", "csi"],
            "hidden_dim": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_subcarriers": 1,
            "num_antennas": 4,
            "num_paths": 2,
            "frontend": {
                "type": "paper_modal_tokenizers",
                "num_layers": 1,
                "num_heads": 4,
                "formal_experiment_eligible": False,
            },
            "encoders": {
                "image": {
                    "type": "jepa_context_image",
                    "latent_dim": 16,
                    "pooling": "mean",
                    "visual_encoder": {
                        "type": "patch_vit",
                        "image_size": 8,
                        "patch_size": 4,
                        "depth": 0,
                        "num_heads": 4,
                        "mlp_ratio": 2.0,
                    },
                },
                "csi": {"type": "linear_sequence_tokenizer"},
            },
        }
    )
    csi_input = torch.randn(2, 3, 1, 4, 2)
    output = model(
        image_batch=torch.randn(2, 3, 3, 8, 8),
        csi_input=csi_input,
        csi_observation_mask=torch.ones(2, 3, 1, 4),
    )
    adapted = adapt_model_output(output)
    metadata = model.training_strategy_metadata()
    runtime = final_config_with_runtime(
        {
            "model": {
                "primary": {
                    "type": "pinn_multimodal_beam",
                    "modalities": ["image", "csi"],
                    "hidden_dim": 16,
                    "num_subcarriers": 1,
                    "num_antennas": 4,
                    "num_paths": 2,
                    "frontend": {"type": "paper_modal_tokenizers", "formal_experiment_eligible": False},
                }
            },
            "data": {"csi_input_mode": "sparse_pilot"},
            "loss": {"physics": {"enabled": True}},
        },
        run_dir=ROOT / "outputs" / "unused",
        primary_model=model,
    )["runtime"]["physics_informed"]
    criterion = PhysicsInformedBeamLoss(csi_reconstruction={"enabled": True, "weight": 0.1})
    batch = {"physics_targets": {"csi_target": torch.randn(2, 1, 1, 4, 2)}}
    result = criterion.compute({"logits": adapted.logits, **adapted.diagnostics}, batch, torch.randint(0, 8, (2, 1)))
    result["loss"].backward()

    assert adapted.logits.shape == (2, 1, 8)
    assert adapted.diagnostics["h_hat"].shape == (2, 1, 1, 4)
    assert metadata["frontend_type"] == "paper_modal_tokenizers"
    assert metadata["tokenizers"]["image"]["type"] == "jepa_context_image"
    assert metadata["tokenizers"]["image"]["uses_gps_context"] is False
    assert metadata["tokenizers"]["csi"]["type"] == "linear_sequence_tokenizer"
    assert metadata["channel_target_scope"] == "narrowband_array_channel"
    assert runtime["frontend_type"] == "paper_modal_tokenizers"
    assert runtime["channel_target_scope"] == "narrowband_array_channel"
    assert runtime["formal_experiment_eligible"] is False
    assert any(param.grad is not None for param in model.parameters())


def test_pinn_paper_frontend_requires_formal_image_checkpoint_and_rejects_csi_target():
    import_default_components()
    cfg = {
        "type": "pinn_multimodal_beam",
        "modalities": ["image"],
        "hidden_dim": 16,
        "num_classes": 8,
        "num_pred": 1,
        "num_subcarriers": 1,
        "num_antennas": 4,
        "frontend": {"type": "paper_modal_tokenizers"},
    }
    with pytest.raises(ValueError, match="checkpoint"):
        MODELS.build(cfg)

    cfg["frontend"]["formal_experiment_eligible"] = False
    model = MODELS.build(cfg)
    with pytest.raises(ValueError, match="csi_target"):
        model(image_batch=torch.randn(1, 2, 3, 8, 8), csi_target=torch.randn(1, 1, 1, 4, 2))


def test_physics_loss_backward_missing_targets_metrics_and_config_load():
    cfg = load_config(ROOT / "configs/fusion/physics_informed_mmw_debug.yaml")
    model = build_model(cfg["model"]["primary"])
    output = model(image_batch=torch.randn(2, 4, 3, 8, 8))
    adapted = adapt_model_output(output)
    horizon = adapted.logits.shape[1]
    _, _, num_subcarriers, num_antennas = adapted.diagnostics["h_hat"].shape
    batch = {
        "physics_targets": {
            "csi_target": torch.randn(2, horizon, num_subcarriers, num_antennas, 2),
            "path_params": torch.randn(2, horizon, 3, 5),
            "beamspace_power": torch.rand(2, horizon, 64),
        }
    }
    labels = torch.randint(0, 64, (2, horizon))
    criterion = PhysicsInformedBeamLoss(**cfg["loss"]["physics"])
    result = criterion.compute({"logits": adapted.logits, **adapted.diagnostics}, batch, labels)
    result["loss"].backward()

    assert result["components"]["csi_loss"].item() >= 0.0
    assert result["diagnostics"]["loss/beam_power_available_count"] > 0
    assert any(param.grad is not None for param in model.parameters())
    runtime = final_config_with_runtime(cfg, run_dir=ROOT / "outputs" / "unused", primary_model=model)["runtime"][
        "physics_informed"
    ]
    assert runtime["used_csi_as_input"] is False
    assert runtime["csi_input_mode"] == "none"
    assert runtime["main_conclusion_eligible"] is False
    assert runtime["codebook_source"] == "ula_dft_fallback"

    missing = PhysicsInformedBeamLoss(csi_reconstruction={"enabled": True, "weight": 1.0}).compute(
        {"logits": adapted.logits},
        {},
        labels,
    )
    assert missing["diagnostics"]["loss/csi_available_count"] == 0.0

    gain = normalized_beamforming_gain(torch.tensor([[1]]), torch.tensor([[[0.1, 0.5, 0.2]]]))
    report = grouped_report([{"condition": "sunny", "town": "Town10", "scene": "s", "value": float(gain.item())}])
    assert report["sunny|Town10|s"]["count"] == 1.0


def test_physics_ablation_configs_load_and_toggle_expected_fields():
    no_physics = load_config(ROOT / "configs/fusion/physics_informed_mmw_no_physics.yaml")
    no_csi = load_config(ROOT / "configs/fusion/physics_informed_mmw_no_csi_reconstruction.yaml")
    image_only = load_config(ROOT / "configs/fusion/physics_informed_mmw_image_only.yaml")

    assert no_physics["model"]["primary"]["use_physics_head"] is False
    assert no_physics["loss"]["physics"]["enabled"] is False
    assert no_csi["loss"]["physics"]["csi_reconstruction"]["weight"] == 0.0
    assert image_only["model"]["primary"]["modalities"] == ["image"]
    assert image_only["data"]["dataset"]["use_csi"] is False

    vision = load_config(ROOT / "configs/fusion/physics_informed_mmw_vision_only.yaml")
    partial = load_config(ROOT / "configs/fusion/physics_informed_mmw_partial_csi_multimodal.yaml")
    sparse = load_config(ROOT / "configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml")
    paper_debug = load_config(ROOT / "configs/fusion/physics_informed_mmw_paper_debug.yaml")
    history = load_config(ROOT / "configs/fusion/physics_informed_mmw_history_csi_multimodal.yaml")
    oracle = load_config(ROOT / "configs/fusion/physics_informed_mmw_oracle_full_csi.yaml")

    assert vision["data"]["use_csi_input"] is False
    assert vision["data"]["csi_input_mode"] == "none"
    assert vision["model"]["primary"]["modalities"] == ["image"]
    assert partial["data"]["csi_input_mode"] == "partial"
    assert partial["model"]["primary"]["modalities"] == ["image", "csi"]
    assert sparse["data"]["csi_input_mode"] == "sparse_pilot"
    assert sparse["data"]["dataset"]["physics_supervision"]["pilot_pattern"] == "grid"
    assert sparse["model"]["primary"]["modalities"] == ["image", "csi"]
    assert sparse["model"]["primary"]["frontend"]["type"] == "paper_modal_tokenizers"
    assert sparse["model"]["primary"]["frontend"]["encoders"]["image"]["type"] == "jepa_context_image"
    assert sparse["model"]["primary"]["frontend"]["encoders"]["image"]["pooling"] == "mean"
    assert paper_debug["model"]["primary"]["frontend"]["formal_experiment_eligible"] is False
    assert not paper_debug["model"]["primary"]["frontend"]["encoders"]["image"].get("checkpoint_path")
    assert history["data"]["csi_input_mode"] == "history"
    assert history["data"]["history_len"] == 5
    assert oracle["data"]["csi_input_mode"] == "oracle_full"
    assert oracle["data"]["allow_oracle_full_csi_input"] is True
