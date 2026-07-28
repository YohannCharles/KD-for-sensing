import torch

from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier


def _inputs(batch: int, history: int):
    values = torch.complex(torch.randn(batch, history, 4, 8), torch.randn(batch, history, 4, 8))
    pattern_ids = torch.arange(4).expand(batch, history, -1)
    frequencies = torch.linspace(-1.0, 1.0, 8)
    mask = torch.ones_like(values, dtype=torch.bool)
    snr = torch.full((batch, history), 10.0)
    return values, pattern_ids, frequencies, mask, snr


def test_information_classifier_uses_structured_temporal_encoder_and_gradients():
    model = SparsePilotInformationClassifier(history_length=5, sensing_dim=64)
    values, pattern_ids, frequencies, mask, snr = _inputs(3, 5)
    sensing = torch.randn(3, 64)
    output = model(values, pattern_ids, frequencies, mask, snr, sensing_feature=sensing)
    assert output["logits"].shape == (3, 64)
    assert output["frame_csi_features"].shape == (3, 5, 128)
    output["logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.csi_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.temporal.parameters())


def test_information_classifier_single_frame_does_not_create_temporal_module():
    model = SparsePilotInformationClassifier(history_length=1)
    values, pattern_ids, frequencies, mask, snr = _inputs(2, 1)
    output = model(values, pattern_ids, frequencies, mask, snr)
    assert model.temporal is None
    assert output["logits"].shape == (2, 64)


def test_information_classifier_supports_linear_time_pooling_and_zero_residual():
    model = SparsePilotInformationClassifier(
        history_length=1,
        sensing_dim=64,
        num_candidate_patterns=32,
        encoder_layers=0,
        fusion_mode="residual",
    )
    values = torch.complex(torch.randn(2, 1, 32, 16), torch.randn(2, 1, 32, 16))
    pattern_ids = torch.arange(32).expand(2, 1, -1)
    frequencies = torch.linspace(-1.0, 1.0, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    snr = torch.full((2, 1), 10.0)
    base = torch.softmax(torch.randn(2, 64), dim=-1)
    output = model(
        values,
        pattern_ids,
        frequencies,
        mask,
        snr,
        sensing_feature=torch.randn(2, 64),
        base_probabilities=base,
    )
    torch.testing.assert_close(output["logits"].softmax(dim=-1), base)
    output["logits"].sum().backward()
    assert model.csi_encoder.encoder is None
    assert any(parameter.grad is not None for parameter in model.csi_encoder.parameters())
