import torch

from kd_sensing.models.temporal_transformer import SharedTemporalTransformer


def test_shared_temporal_transformer_shapes_and_missing_zero() -> None:
    module = SharedTemporalTransformer(dropout=0.0)
    features = torch.randn(2, 5, 4, 64)
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    mask[0, :, 2] = False
    mask[1, :3, 1] = False

    output = module(features, mask)

    assert output["temporal_token_features"].shape == (2, 5, 4, 64)
    assert output["temporal_cls_features"].shape == (2, 4, 64)
    assert output["temporal_attention_valid_fraction"].shape == (2, 4)
    assert torch.count_nonzero(output["temporal_cls_features"][0, 2]) == 0
    assert torch.count_nonzero(output["temporal_token_features"][~mask]) == 0
    assert output["available_modalities"].tolist() == [[True, True, False, True], [True] * 4]


def test_temporal_transformer_is_one_shared_encoder_with_expected_embeddings() -> None:
    module = SharedTemporalTransformer()

    assert module.time_embedding.shape == (5, 64)
    assert module.modality_embedding.shape == (4, 64)
    assert module.cls_token.shape == (1, 1, 64)
    assert len(module.encoder.layers) == 2
    assert not hasattr(module, "modality_encoders")


def test_missing_frame_cannot_change_cls_output() -> None:
    torch.manual_seed(3)
    module = SharedTemporalTransformer(dropout=0.0).eval()
    features = torch.randn(1, 5, 4, 64)
    mask = torch.ones(1, 5, 4, dtype=torch.bool)
    mask[:, 2, 0] = False
    changed = features.clone()
    changed[:, 2, 0] = 1e6

    first = module(features, mask)["temporal_cls_features"]
    second = module(changed, mask)["temporal_cls_features"]

    torch.testing.assert_close(first, second)
