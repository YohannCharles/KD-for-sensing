from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.tspc_v2 import TSPCV2LossConfig, TSPCV2Model, TSPCV2ModelConfig, tspc_v2_losses
from tools.run_tspc_v2 import _forbidden_record_keys, _forward_batch, _load_role, _method_spec


def _shared_bank() -> BeamPrototypeBank:
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    bank.prototypes.requires_grad_(False)
    return bank


def _config(**overrides) -> TSPCV2ModelConfig:
    values = {
        "feature_dim": 64,
        "history_length": 5,
        "num_modalities": 4,
        "num_beams": 64,
        "sensing_frame_layers": 1,
        "sensing_temporal_hidden_dim": 32,
        "sensing_temporal_layers": 1,
        "sensing_dropout": 0.0,
        "csi_dim": 16,
        "csi_encoder_layers": 0,
        "csi_heads": 4,
        "csi_temporal_layers": 1,
        "csi_dropout": 0.0,
        "residual_heads": 4,
        "residual_dropout": 0.0,
    }
    values.update(overrides)
    return TSPCV2ModelConfig(**values)


def _inputs(batch: int) -> dict[str, torch.Tensor]:
    return {
        "features": torch.randn(batch, 5, 4, 64),
        "full_probability": torch.softmax(torch.randn(batch, 64), dim=-1),
        "pilot_observations": torch.randn(batch, 5, 2, 2, dtype=torch.complex64),
        "pattern_ids": torch.arange(2).view(1, 1, 2).expand(batch, 5, -1).clone(),
        "frequency_positions": torch.tensor([-1.0, 1.0]),
        "frequency_ids": torch.tensor([0, 15]),
        "pilot_mask": torch.ones(batch, 5, 2, 2, dtype=torch.bool),
        "snr_db": torch.full((batch, 5), 10.0),
    }


def _forward(
    model: TSPCV2Model,
    values: dict[str, torch.Tensor],
    availability: torch.Tensor,
    bank: BeamPrototypeBank | None,
) -> dict[str, torch.Tensor]:
    return model(
        values["features"],
        availability,
        shared_prototype_bank=bank,
        pilot_observations=values["pilot_observations"],
        pattern_ids=values["pattern_ids"],
        frequency_positions=values["frequency_positions"],
        frequency_ids=values["frequency_ids"],
        pilot_mask=values["pilot_mask"],
        snr_db=values["snr_db"],
        full_probability=values["full_probability"],
    )


@pytest.mark.parametrize("batch", (1, 3))
@pytest.mark.parametrize("residual_mode", ("cross_attention", "residual_mlp"))
def test_tspc_v2_shapes_and_backward_across_batch_sizes(batch: int, residual_mode: str):
    model = TSPCV2Model(_config(residual_mode=residual_mode))
    model.configure_stage("stage_c")
    bank = _shared_bank()
    values = _inputs(batch)
    availability = torch.tensor([True, False, True, True]).expand(batch, -1)
    output = _forward(model, values, availability, bank)
    assert output["frame_features"].shape == (batch, 5, 64)
    assert output["z_sensing"].shape == (batch, 64)
    assert output["sensing_evidence"].shape == (batch, 64)
    assert output["csi_frame_features"].shape == (batch, 5, 16)
    assert output["csi_temporal_tokens"].shape == (batch, 5, 16)
    assert output["z_csi"].shape == (batch, 16)
    assert output["prototype_context"].shape == (batch, 64, 64)
    assert output["missing_pattern_token"].shape == (batch, 64)
    assert output["delta_evidence"].shape == (batch, 64)
    assert output["final_probability"].shape == (batch, 64)
    assert torch.isfinite(output["final_evidence"]).all()
    labels = torch.randint(0, 64, (batch,))
    losses = tspc_v2_losses(
        output,
        labels,
        config=TSPCV2LossConfig(prototype_weight=0.1),
        prototype_bank=model.prototype_bank_for_loss(bank),
    )
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 AMP regression")
def test_tspc_v2_supports_bfloat16_autocast():
    device = torch.device("cuda")
    model = TSPCV2Model(_config()).to(device)
    model.configure_stage("stage_c")
    bank = _shared_bank().to(device)
    values = {key: value.to(device) for key, value in _inputs(2).items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward(model, values, torch.tensor([[True, False, True, True]], device=device).expand(2, -1), bank)
    loss = output["final_evidence"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_missing_slots_and_invalid_pilots_cannot_change_observed_path():
    torch.manual_seed(13)
    model = TSPCV2Model(_config())
    model.eval()
    bank = _shared_bank()
    values = _inputs(2)
    temporal_availability = torch.ones(2, 5, 4, dtype=torch.bool)
    temporal_availability[:, :, 1] = False
    values["pilot_mask"].zero_()
    changed = {key: value.clone() for key, value in values.items()}
    changed["features"][:, :, 1] = torch.randn_like(changed["features"][:, :, 1]) * 1000.0
    first = _forward(model, values, temporal_availability, bank)
    second = _forward(model, changed, temporal_availability, bank)
    assert torch.equal(first["sensing_evidence"], second["sensing_evidence"])
    assert torch.equal(first["frame_features"], second["frame_features"])
    assert torch.equal(first["final_evidence"], first["sensing_evidence"])
    assert not bool(first["csi_available"].any())

    valid_values = _inputs(2)
    valid_values["pilot_mask"][:, :, 0, 0] = False
    altered = {key: value.clone() for key, value in valid_values.items()}
    altered["pilot_observations"][:, :, 0, 0] = torch.randn(2, 5, dtype=torch.complex64) * 1000.0
    physical_availability = torch.tensor([True, False, True, True]).expand(2, -1)
    clean_output = _forward(model, valid_values, physical_availability, bank)
    altered_output = _forward(model, altered, physical_availability, bank)
    assert torch.equal(clean_output["csi_frame_features"], altered_output["csi_frame_features"])
    assert torch.equal(clean_output["z_csi"], altered_output["z_csi"])

    last_frame_values = _inputs(2)
    last_frame_values["pilot_mask"][:, :-1] = False
    last_frame_output = _forward(model, last_frame_values, physical_availability, bank)
    assert not bool(last_frame_output["csi_frame_available"][:, :-1].any())
    assert torch.equal(
        last_frame_output["csi_temporal_tokens"][:, :-1],
        torch.zeros_like(last_frame_output["csi_temporal_tokens"][:, :-1]),
    )


def test_one_two_and_three_missing_modalities_are_supported():
    model = TSPCV2Model(_config())
    model.eval()
    bank = _shared_bank()
    values = _inputs(2)
    values["pilot_mask"].zero_()
    for availability in (
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, True, False]),
        torch.tensor([False, False, True, False]),
    ):
        output = _forward(model, values, availability.expand(2, -1), bank)
        assert output["final_evidence"].shape == (2, 64)
        assert torch.equal(output["final_evidence"], output["sensing_evidence"])
    temporal_availability = torch.tensor([True, False, True, False]).view(1, 1, 4).expand(2, 5, -1).clone()
    temporal_availability[:, 0] = False
    temporal_output = _forward(model, values, temporal_availability, bank)
    assert temporal_output["frame_features"].shape == (2, 5, 64)


def test_full_bypass_is_exact_and_mixed_rows_only_encode_missing_csi():
    model = TSPCV2Model(_config())
    model.eval()
    bank = _shared_bank()
    full_values = _inputs(2)
    calls: list[tuple[torch.Tensor, ...]] = []
    hook = model.csi_encoder.register_forward_hook(lambda _, args, __: calls.append(args))
    full = model(
        full_values["features"],
        torch.ones(2, 4, dtype=torch.bool),
        shared_prototype_bank=bank,
        full_probability=full_values["full_probability"],
    )
    assert not calls
    assert torch.equal(full["final_probability"], full_values["full_probability"])
    assert full["pilot_re_window"].sum().item() == 0

    mixed_values = _inputs(2)
    mixed = _forward(model, mixed_values, torch.tensor([[True, True, True, True], [True, False, True, True]]), bank)
    hook.remove()
    assert len(calls) == 1
    assert calls[0][0].shape[0] == 1
    assert torch.equal(mixed["final_probability"][0], mixed_values["full_probability"][0])
    assert mixed["pilot_re_window"][0].item() == 0

    csi_off_values = _inputs(2)
    csi_off_values["pilot_mask"].zero_()
    csi_off_calls = []
    csi_off_hook = model.csi_encoder.register_forward_hook(lambda *_: csi_off_calls.append(True))
    csi_off = _forward(model, csi_off_values, torch.tensor([True, False, True, True]).expand(2, -1), bank)
    csi_off_hook.remove()
    assert not csi_off_calls
    assert torch.equal(csi_off["delta_evidence"], torch.zeros_like(csi_off["delta_evidence"]))
    assert torch.equal(csi_off["final_evidence"], csi_off["sensing_evidence"])
    no_csi_inputs = model(
        csi_off_values["features"],
        torch.tensor([True, False, True, True]).expand(2, -1),
        shared_prototype_bank=bank,
    )
    assert torch.equal(no_csi_inputs["final_evidence"], no_csi_inputs["sensing_evidence"])


def test_prototype_residual_zero_initialization_and_trainable_controls():
    shared = _shared_bank()
    values = _inputs(2)
    availability = torch.tensor([True, False, True, True]).expand(2, -1)
    shared_model = TSPCV2Model(_config())
    shared_output = _forward(shared_model, values, availability, shared)
    assert shared_output["delta_evidence"].count_nonzero().item() == 0
    shared_output["final_evidence"].square().mean().backward()
    assert shared.prototypes.grad is None
    assert shared_model.residual.scalar_output.weight.grad is not None
    assert torch.isfinite(shared_model.residual.scalar_output.weight.grad).all()

    independent_model = TSPCV2Model(_config(prototype_mode="independent"))
    independent_output = _forward(independent_model, values, availability, shared)
    independent_output["sensing_evidence"].square().mean().backward()
    independent_bank = independent_model.sensing.evidence_head.prototype_bank
    assert independent_bank is not None and independent_bank.prototypes.grad is not None

    random_one = TSPCV2Model(_config(prototype_mode="random_frozen", random_seed=29))
    random_two = TSPCV2Model(_config(prototype_mode="random_frozen", random_seed=29))
    first_bank = random_one.sensing.evidence_head.prototype_bank
    second_bank = random_two.sensing.evidence_head.prototype_bank
    assert first_bank is not None and second_bank is not None
    assert not first_bank.prototypes.requires_grad
    assert torch.equal(first_bank.prototypes, second_bank.prototypes)


def test_kl_and_centered_residual_losses_are_explicitly_teacher_bound():
    model = TSPCV2Model(_config())
    bank = _shared_bank()
    values = _inputs(2)
    output = _forward(model, values, torch.tensor([True, False, True, True]).expand(2, -1), bank)
    labels = torch.tensor([2, 7])
    loss_config = TSPCV2LossConfig(compensation_kl_weight=0.1, residual_regression_weight=0.1)
    teacher_probability = values["full_probability"]
    losses = tspc_v2_losses(
        output,
        labels,
        config=loss_config,
        prototype_bank=bank,
        teacher_probability=teacher_probability,
        teacher_evidence=teacher_probability.clamp_min(1e-12).log(),
    )
    assert torch.isfinite(losses["loss_compensation_kl"])
    assert torch.isfinite(losses["loss_residual_regression"])
    with pytest.raises(ValueError, match="teacher_probability"):
        tspc_v2_losses(output, labels, config=loss_config, teacher_evidence=teacher_probability.log())


def test_v2_configs_keep_budget_controls_and_reject_future_or_outer_roles():
    configs = {
        "stage_a_sensing.yaml": ("A0", "A1", "A2", "A3", "A4"),
        "stage_b_compensation.yaml": ("B1", "B2", "B3", "B4", "B5"),
        "stage_c_joint.yaml": ("C0", "C1", "C2"),
    }
    for filename, methods in configs.items():
        config = safe_load_yaml((Path("tools/configs/tspc_v2") / filename).read_text(encoding="utf-8"))
        assert config["protocol"]["outer_test_enabled"] is False
        for method in methods:
            _method_spec(config, method)
        with pytest.raises(ValueError, match="legacy B0"):
            _method_spec(config, "B0")

    c_config = safe_load_yaml(Path("tools/configs/tspc_v2/stage_c_joint.yaml").read_text(encoding="utf-8"))
    c1, c2 = _method_spec(c_config, "C1"), _method_spec(c_config, "C2")
    assert c_config["training"]["sensing_learning_rate_scale"] == 0.1
    assert c_config["training"]["smoke_updates"] == 100
    assert (c1["patterns"] * c1["frequencies"], c1["history_frames"]) == (20, 1)
    assert (c2["patterns"] * c2["frequencies"], c2["history_frames"]) == (4, 5)
    assert c1["patterns"] * c1["frequencies"] * c1["history_frames"] == 20
    assert c2["patterns"] * c2["frequencies"] * c2["history_frames"] == 20
    a_config = safe_load_yaml(Path("tools/configs/tspc_v2/stage_a_sensing.yaml").read_text(encoding="utf-8"))
    assert _method_spec(a_config, "A0")["model_config"].prototype_mode == "shared_frozen"
    b_config = safe_load_yaml(Path("tools/configs/tspc_v2/stage_b_compensation.yaml").read_text(encoding="utf-8"))
    b4 = _method_spec(b_config, "B4")["model_config"]
    assert not b4.use_sensing_context and not b4.use_mask_context
    assert _method_spec(b_config, "B2_CE")["loss_config"].compensation_kl_weight == 0.0
    assert _method_spec(b_config, "B2_KL_DELTA")["loss_config"].residual_regression_weight == 0.1
    assert _forbidden_record_keys({"safe": {"future_csi_payload": torch.tensor(1)}}) == ["safe.future_csi_payload"]
    with pytest.raises(ValueError, match="train or validation"):
        _load_role({}, "outer_test")
    assert all("future" not in name for name in inspect.signature(TSPCV2Model.forward).parameters)


def test_training_label_is_not_read_by_the_forward_data_path():
    class GuardedRecovery(dict):
        def __getitem__(self, key):
            if key == "labels_future":
                raise AssertionError("The t+1 label was read by forward.")
            return super().__getitem__(key)

    model = TSPCV2Model(_config())
    model.eval()
    bank = _shared_bank()
    feature = {
        "token_sequence": torch.randn(3, 5, 4, 64),
        "p_full": torch.softmax(torch.randn(3, 64), dim=-1),
    }
    recovery = GuardedRecovery(candidate_history=torch.randn(3, 5, 32, 16, dtype=torch.complex64))
    output = _forward_batch(
        model,
        SimpleNamespace(prototype_bank=bank),
        feature,
        recovery,
        torch.tensor([0, 2]),
        torch.tensor([True, False, True, False]),
        {"patterns": 2, "frequencies": 2, "history_frames": 5, "csi_enabled": True},
        {"training": {"amp": False}},
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(5),
        snr_db=10.0,
        dropout=0.0,
        frequency_metadata=(torch.tensor([-1.0, 1.0]), torch.tensor([0, 15])),
    )
    assert output["final_evidence"].shape == (2, 64)
