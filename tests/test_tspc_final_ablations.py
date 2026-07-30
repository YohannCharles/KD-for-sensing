from pathlib import Path

import pytest
import torch

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.temporal_radio_encoders import TEMPORAL_RADIO_METHODS, TemporalRadioEncoder
from kd_sensing.models.tspc_ablation_heads import (
    RadioAblationHead,
    SparseRadioAblationModel,
    apply_exact_fallback,
    fuse_expert_probabilities,
    random_frozen_prototype_bank,
)
from tools.run_tspc_final_ablations import MASK_NAMES, pilot_resource_accounting, select_candidate_history


@pytest.mark.parametrize("method", TEMPORAL_RADIO_METHODS)
def test_temporal_radio_encoders_forward_backward(method: str):
    frames = torch.randn(3, 5, 128, requires_grad=True)
    model = TemporalRadioEncoder(method)
    output = model(frames)
    assert output.shape == (3, 128)
    output.square().mean().backward()
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()


def test_temporal_budget_and_history_order_are_exact():
    candidates = torch.arange(2 * 5 * 32 * 16).reshape(2, 5, 32, 16).to(torch.complex64)
    temporal = select_candidate_history(candidates, budget="2x2", history_frames=5)
    expected = candidates[:, :, :2].index_select(-1, torch.tensor([0, 15]))
    assert torch.equal(temporal, expected)
    last = select_candidate_history(candidates, budget="2x2", history_frames=1)
    assert torch.equal(last, expected[:, -1:])
    concentrated = select_candidate_history(candidates, budget="5x4", history_frames=1)
    assert concentrated.shape == (2, 1, 5, 4)
    assert pilot_resource_accounting("2x2", 5)["pilot_re_window"] == 20
    assert pilot_resource_accounting("2x2", 1)["pilot_re_window"] == 4
    assert pilot_resource_accounting("5x4", 1)["pilot_re_window"] == 20


def test_sparse_radio_wrapper_all_temporal_paths_have_one_head_gradient():
    observations = torch.randn(2, 5, 2, 2, dtype=torch.complex64)
    pattern_ids = torch.arange(2).expand(2, 5, 2)
    frequencies = torch.tensor([-1.0, 1.0])
    valid = torch.ones_like(observations, dtype=torch.bool)
    snr = torch.full((2, 5), 10.0)
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    bank.prototypes.requires_grad_(False)
    for method in ("mean", "gru", "lstm", "tcn"):
        model = SparseRadioAblationModel(method, "P0")
        output = model(observations, pattern_ids, frequencies, valid, snr, bank)
        assert output["frame_features"].shape == (2, 5, 128)
        assert output["c_radio"].shape == (2, 128)
        assert output["z_radio"].shape == (2, 64)
        assert output["radio_evidence"].shape == (2, 64)
        output["radio_evidence"].square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.frame_encoder.parameters())
        assert any(parameter.grad is not None for parameter in model.temporal_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 autocast regression")
def test_sparse_radio_wrapper_supports_bfloat16_autocast():
    device = torch.device("cuda")
    model = SparseRadioAblationModel("gru", "P0").to(device)
    bank = BeamPrototypeBank(64, 64, temperature=0.1).to(device)
    bank.prototypes.requires_grad_(False)
    observations = torch.randn(2, 5, 2, 2, dtype=torch.complex64, device=device)
    pattern_ids = torch.arange(2, device=device).expand(2, 5, 2)
    frequencies = torch.tensor([-1.0, 1.0], device=device)
    valid = torch.ones_like(observations, dtype=torch.bool)
    snr = torch.full((2, 5), 10.0, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(observations, pattern_ids, frequencies, valid, snr, bank)
    loss = output["radio_evidence"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_shared_independent_and_random_prototypes_are_isolated():
    shared = BeamPrototypeBank(64, 64, temperature=0.1)
    p0 = RadioAblationHead("P0")
    p1 = RadioAblationHead("P1")
    p4 = RadioAblationHead("P4", seed=19)
    assert p0.decision_bank(shared) is shared
    assert p1.decision_bank(shared) is p1.prototype_bank
    assert p1.prototype_bank is not shared
    assert p1.prototype_bank.prototypes.requires_grad
    assert p4.prototype_bank is not shared
    assert not p4.prototype_bank.prototypes.requires_grad
    first = random_frozen_prototype_bank(seed=19).prototypes
    second = random_frozen_prototype_bank(seed=19).prototypes
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(64), atol=1e-6)


def test_fusion_formulas_use_calibrated_fp32_evidence_once():
    generator = torch.Generator().manual_seed(7)
    sensing = torch.softmax(torch.randn(4, 64, generator=generator), dim=-1).to(torch.float16)
    radio_evidence = torch.randn(4, 64, generator=generator).to(torch.float16)
    temperature = 0.8
    l2 = fuse_expert_probabilities("L2", sensing, radio_evidence, sensing_temperature=temperature)
    l3 = fuse_expert_probabilities("L3", sensing, radio_evidence, sensing_temperature=temperature)
    l4 = fuse_expert_probabilities("L4", sensing, radio_evidence, sensing_temperature=temperature)
    calibrated_sensing = torch.softmax(sensing.float().clamp_min(1e-12).log() / temperature, dim=-1)
    calibrated_radio = torch.softmax(radio_evidence.float(), dim=-1)
    expected_l3 = 0.5 * calibrated_sensing + 0.5 * calibrated_radio
    assert l2.dtype == l3.dtype == l4.dtype == torch.float32
    assert torch.allclose(l3, expected_l3, atol=1e-7)
    assert torch.allclose(l2, l4, atol=2e-7)


def test_feature_sum_queries_shared_bank_and_exact_fallback_bypasses_rows():
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    sensing_probability = torch.softmax(torch.randn(3, 64), dim=-1)
    radio_evidence = torch.randn(3, 64)
    z_sensing = torch.randn(3, 64)
    z_radio = torch.randn(3, 64)
    feature_sum = fuse_expert_probabilities(
        "L1",
        sensing_probability,
        radio_evidence,
        z_sensing=z_sensing,
        z_radio=z_radio,
        shared_bank=bank,
    )
    expected = torch.softmax(bank(0.5 * z_sensing + 0.5 * z_radio), dim=-1)
    assert torch.allclose(feature_sum, expected)
    available = torch.tensor([True, False, True])
    full = torch.tensor([False, False, True])
    final = apply_exact_fallback(sensing_probability, feature_sum, csi_available=available, full=full)
    assert torch.equal(final[1], sensing_probability[1])
    assert torch.equal(final[2], sensing_probability[2])
    assert torch.equal(final[0], feature_sum[0])


def test_config_keeps_all_masks_and_outer_test_sealed():
    config = safe_load_yaml(Path("tools/configs/tspc_final_ablations.yaml").read_text(encoding="utf-8"))
    assert len(MASK_NAMES) == 14
    assert set(MASK_NAMES) == set(ALL_MASKS_WITHOUT_FULL)
    assert config["protocol"]["outer_test_enabled"] is False
    assert config["pilot"]["temporal_budget"] == "2x2"
    assert config["pilot"]["temporal_history_frames"] == 5
    assert config["model"]["fixed_lambda"] == 0.5
    assert config["training"]["max_epochs"] == 60
    assert config["training"]["batch_size"] == 256


ALL_MASKS_WITHOUT_FULL = {
    "missing_image",
    "image_only",
    "missing_lidar",
    "lidar_only",
    "missing_radar",
    "radar_only",
    "missing_gps",
    "gps_only",
    "missing_image_lidar",
    "missing_image_radar",
    "missing_image_gps",
    "missing_lidar_radar",
    "missing_lidar_gps",
    "missing_radar_gps",
}
