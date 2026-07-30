from __future__ import annotations

from itertools import combinations
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from kd_sensing.baselines.csi_anchored_completion import CSIAnchoredCompletionModel
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier
from kd_sensing.models.available_modality_context import AvailableModalityContext
from kd_sensing.models.csi_anchored_completion import (
    CSIAnchoredPrototypeCompletion,
    SparsePilotRadioEncoder,
)
from kd_sensing.config.parsing import safe_load_yaml
from tools.run_csi_anchored_completion import _completion_losses, validate_cache_record


def _all_masks() -> torch.Tensor:
    masks = []
    for count in range(1, 5):
        masks.extend(
            tuple(int(index in available) for index in range(4))
            for available in combinations(range(4), count)
        )
    return torch.tensor(masks, dtype=torch.bool)


def _completion(*, use_radio: bool = True, use_prototypes: bool = True, cross: bool = True):
    return CSIAnchoredPrototypeCompletion(
        feature_dim=64,
        radio_dim=128,
        quality_dim=21,
        hidden_dim=32,
        num_beams=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        top_k=8,
        use_radio=use_radio,
        use_prototype_memory=use_prototypes,
        use_cross_attention=cross,
    )


def _radio_features(batch: int, *, available: bool = True) -> dict[str, torch.Tensor]:
    quality = torch.randn(batch, 21)
    quality[:, -5] = 10.0 / 30.0
    quality[:, -4] = 1.0
    quality[:, -3] = 0.0
    quality[:, -2:] = 1.0
    return {
        "c_radio": torch.randn(batch, 128),
        "csi_quality": quality,
        "csi_available": torch.full((batch,), available, dtype=torch.bool),
    }


def test_available_context_excludes_missing_tokens_from_attention_and_pooling():
    torch.manual_seed(1)
    model = AvailableModalityContext(input_dim=8, hidden_dim=16, num_heads=4, dropout=0.0).eval()
    tokens = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    changed = tokens.clone()
    changed[~mask] = 1e6
    first = model(tokens, mask)
    second = model(changed, mask)
    torch.testing.assert_close(first["z_available"], second["z_available"])
    torch.testing.assert_close(first["available_tokens"], second["available_tokens"])
    assert not bool(first["pool_weights"][~mask].count_nonzero())


def test_radio_encoder_loads_only_validated_encoder_and_gru_checkpoint(tmp_path):
    torch.manual_seed(2)
    source = SparsePilotInformationClassifier(
        history_length=5,
        hidden_dim=32,
        num_candidate_patterns=16,
        encoder_layers=0,
    ).eval()
    checkpoint = tmp_path / "information.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint)
    radio = SparsePilotRadioEncoder(
        history_length=5,
        hidden_dim=32,
        num_candidate_patterns=16,
        encoder_layers=0,
    ).eval()
    audit = radio.load_information_checkpoint(checkpoint)
    assert audit["classifier_loaded"] is False
    assert not any("classifier" in name for name, _ in radio.named_parameters())

    observations = torch.complex(torch.randn(3, 5, 16, 8), torch.randn(3, 5, 16, 8))
    ids = torch.arange(16).expand(3, 5, -1)
    frequencies = torch.linspace(-1.0, 1.0, 8)
    valid = torch.ones_like(observations, dtype=torch.bool)
    snr = torch.full((3, 5), 10.0)
    expected = source(observations, ids, frequencies, valid, snr)["csi_feature"]
    actual = radio(observations, ids, frequencies, valid, snr)
    torch.testing.assert_close(actual["c_radio"], expected)
    assert actual["csi_quality"].shape == (3, 21)


def test_completion_only_queries_missing_slots_preserves_real_slots_and_freezes_prototypes():
    torch.manual_seed(3)
    model = _completion().eval()
    features = torch.randn(3, 4, 64)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    radio = _radio_features(3)
    prototypes = torch.nn.Parameter(torch.randn(64, 64))
    output = model(features, mask, **radio, prototypes=prototypes)
    torch.testing.assert_close(output["completed_tokens"][mask], features[mask])
    assert torch.equal(output["query_active"], ~mask)
    assert output["top_k_indices"].shape == (3, 8)
    assert output["prototype_distribution"].shape == (3, 4, 64)
    assert "target" not in model.forward.__annotations__
    F.cross_entropy(output["completed_tokens"].mean(dim=1), torch.tensor([1, 2, 3])).backward()
    assert prototypes.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_csi_unavailable_is_exact_sensing_only_completion():
    torch.manual_seed(4)
    model = _completion(use_radio=True).eval()
    features = torch.randn(2, 4, 64)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    radio = _radio_features(2, available=False)
    prototypes = torch.randn(64, 64)
    anchored = model(features, mask, **radio, prototypes=prototypes)
    model.use_radio = False
    sensing_only = model(features, mask, **radio, prototypes=prototypes)
    torch.testing.assert_close(anchored["completed_tokens"], sensing_only["completed_tokens"], rtol=0, atol=0)
    assert not bool(anchored["sample_radio_reliability"].count_nonzero())


def test_no_prototype_ablation_does_not_query_or_depend_on_prototype_values():
    torch.manual_seed(41)
    model = _completion(use_prototypes=False).eval()
    assert model.prototype_projection is None
    assert model.sensing_query_projection is None
    assert model.radio_query_projection is None
    features = torch.randn(2, 4, 64)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    radio = _radio_features(2)
    first = model(features, mask, **radio, prototypes=torch.randn(64, 64))
    second = model(features, mask, **radio, prototypes=torch.randn(64, 64) * 1000.0)
    torch.testing.assert_close(first["completed_tokens"], second["completed_tokens"], rtol=0, atol=0)


def test_full_batch_bypasses_radio_and_completion_with_exact_m4_probabilities():
    torch.manual_seed(5)
    base = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    completion = _completion().eval()
    completion.forward = Mock(side_effect=AssertionError("completion must not run"))
    radio = SparsePilotRadioEncoder(encoder_layers=0)
    radio.forward = Mock(side_effect=AssertionError("radio must not run"))
    model = CSIAnchoredCompletionModel(base, completion, radio_encoder=radio).eval()
    sequence = torch.randn(2, 5, 4, 64)
    full = torch.ones(2, 4, dtype=torch.bool)
    expected = base.forward_tokens(model._token_mapping(sequence), availability=full)
    actual = model(sequence, full, radio_inputs={"unused": torch.ones(2)})
    torch.testing.assert_close(actual["probabilities"], expected["logits"].softmax(dim=-1), rtol=0, atol=0)
    assert actual["completion_bypassed"] is True
    assert actual["radio_called"] is False
    assert completion.forward.call_count == 0
    assert radio.forward.call_count == 0


def test_wrapper_supports_all_15_masks_backward_and_keeps_real_sequences_exact():
    torch.manual_seed(6)
    masks = _all_masks()
    base = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    model = CSIAnchoredCompletionModel(base, _completion(), radio_encoder=None).train()
    sequence = torch.randn(len(masks), 5, 4, 64)
    radio = _radio_features(len(masks))
    output = model(sequence, masks, radio_output=radio)
    assert output["logits"].shape == (15, 64)
    assert bool(torch.isfinite(output["logits"]).all())
    expanded = masks[:, None, :, None].expand_as(sequence)
    torch.testing.assert_close(output["reconstructed_token_sequence"][expanded], sequence[expanded])

    full_row = masks.all(dim=1).nonzero(as_tuple=False).squeeze(1)
    expected_full = base.forward_tokens(model._token_mapping(sequence[full_row]))["logits"].softmax(dim=-1)
    torch.testing.assert_close(output["probabilities"][full_row], expected_full, rtol=0, atol=0)
    F.cross_entropy(output["logits"], torch.arange(15) % 64).backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in model.completion.parameters())


def test_radio_encoder_accepts_16x16_and_16x8_and_handles_empty_csi():
    torch.manual_seed(7)
    model = SparsePilotRadioEncoder(
        history_length=5,
        hidden_dim=32,
        num_candidate_patterns=32,
        encoder_layers=0,
        quality_dim=4,
    ).eval()
    for frequency_count in (16, 8):
        observations = torch.complex(
            torch.randn(2, 5, 16, frequency_count),
            torch.randn(2, 5, 16, frequency_count),
        )
        ids = torch.arange(16).expand(2, 5, -1)
        valid = torch.ones_like(observations, dtype=torch.bool)
        valid[1] = False
        output = model(
            observations,
            ids,
            torch.linspace(-1.0, 1.0, frequency_count),
            valid,
            torch.tensor([10.0, -10.0]),
        )
        assert output["c_radio"].shape == (2, 32)
        assert output["csi_quality"].shape == (2, 9)
        assert torch.equal(output["csi_available"], torch.tensor([True, False]))
        assert not bool(output["c_radio"][1].count_nonzero())
        assert bool(torch.isfinite(output["csi_quality"]).all())


def test_feature_cache_contract_rejects_identity_mismatch_and_channel_exposure():
    count = 2
    sample_ids = ["train:a", "train:b"]
    feature = {
        "token_sequence": torch.randn(count, 5, 4, 64),
        "modality_features": torch.randn(count, 4, 64),
        "teacher_prototype_probability": torch.softmax(torch.randn(count, 4, 64), dim=-1),
        "p_full": torch.softmax(torch.randn(count, 64), dim=-1),
        "full_pred": torch.zeros(count, dtype=torch.long),
        "target": torch.tensor([1, 2]),
        "future_beam_power": torch.rand(count, 64),
        "sample_ids": sample_ids,
        "trajectory_ids": ["trajectory_a", "trajectory_b"],
        "csi_cache_keys": ["csi:a", "csi:b"],
    }
    recovery = {"sample_ids": sample_ids, "labels_future": torch.tensor([1, 2])}
    validate_cache_record(feature, recovery, expected_count=count)

    feature["channel_ref"] = ["forbidden"] * count
    try:
        validate_cache_record(feature, recovery, expected_count=count)
    except ValueError as error:
        assert "must not expose channel" in str(error)
    else:
        raise AssertionError("Feature cache accepted a channel input field.")
    del feature["channel_ref"]
    recovery["sample_ids"] = list(reversed(sample_ids))
    try:
        validate_cache_record(feature, recovery, expected_count=count)
    except ValueError as error:
        assert "stable sample IDs" in str(error)
    else:
        raise AssertionError("Feature cache accepted misaligned recovery identities.")


def test_configured_completion_loss_is_finite_and_backpropagates_for_mixed_masks():
    torch.manual_seed(8)
    config_path = Path(__file__).parents[1] / "tools/configs/csi_anchored_completion_trajectory.yaml"
    config = safe_load_yaml(config_path.read_text(encoding="utf-8"))
    base = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    model = CSIAnchoredCompletionModel(base, _completion(), radio_encoder=None).train()
    sequence = torch.randn(5, 5, 4, 64)
    physical = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1]],
        dtype=torch.bool,
    )
    first = model(sequence, physical, radio_output=_radio_features(5))
    second = model(sequence, physical, radio_output=_radio_features(5))
    labels = torch.tensor([1, 2, 3, 4, 5])
    batch = {
        "target": labels,
        "modality_features": sequence.mean(dim=1),
        "teacher_prototype_probability": base.prototype_bank(sequence.mean(dim=1).flatten(0, 1))
        .softmax(dim=-1)
        .view(5, 4, 64)
        .detach(),
        "p_full": base.forward_tokens(model._token_mapping(sequence))["logits"].softmax(dim=-1).detach(),
    }
    total, losses = _completion_losses(model, first, second, batch, physical, config)
    assert set(losses) == {
        "total",
        "task",
        "topology",
        "prototype_semantic",
        "radio_distillation",
        "radio_decision_distillation",
        "slot",
        "consistency",
        "quality",
        "preserve",
        "train_top1",
    }
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    total.backward(retain_graph=True)
    assert any(parameter.grad is not None for parameter in model.completion.parameters())

    model.zero_grad(set_to_none=True)
    batch["radio_teacher_probability"] = torch.softmax(torch.randn(5, 64), dim=-1)
    distilled, distilled_losses = _completion_losses(
        model,
        first,
        second,
        batch,
        physical,
        config,
        radio_distillation_weight=1.0,
        radio_decision_distillation_weight=1.0,
    )
    assert float(distilled_losses["radio_distillation"].detach()) > 0
    assert float(distilled_losses["radio_decision_distillation"].detach()) > 0
    distilled.backward()
    assert any(parameter.grad is not None for parameter in model.completion.parameters())
