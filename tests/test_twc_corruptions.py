import pytest
import torch

from kd_sensing.evaluation.corruptions import CORRUPTION_GRID, CorruptionSpec, apply_inference_corruption


GPS_MEAN = torch.tensor([30.0, 0.0, 0.0])
GPS_SCALE = torch.tensor([10.0, 1.0, 1.0])


def _batch():
    def values(*shape):
        return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 100.0

    return {
        "image": values(2, 5, 3, 16, 16),
        "radar_ra": values(2, 5, 1, 8, 8),
        "radar_da": values(2, 5, 1, 8, 8),
        "gps": values(2, 5, 3),
        "lidar": values(2, 5, 3, 16, 16),
    }


@pytest.mark.parametrize("spec", CORRUPTION_GRID)
def test_inference_corruption_is_deterministic_and_changes_only_its_sensor(spec: CorruptionSpec) -> None:
    kwargs = {"gps_scaler_mean": GPS_MEAN, "gps_scaler_scale": GPS_SCALE}
    first = apply_inference_corruption(
        {key: value.clone() for key, value in _batch().items()}, spec, seed=7, batch_index=2, **kwargs
    )
    second = apply_inference_corruption(
        {key: value.clone() for key, value in _batch().items()}, spec, seed=7, batch_index=2, **kwargs
    )
    explicit_none = apply_inference_corruption(
        {key: value.clone() for key, value in _batch().items()},
        spec,
        seed=7,
        batch_index=2,
        selector=None,
        **kwargs,
    )
    affected = {
        "gps_noise": {"gps"},
        "image_occlusion": {"image"},
        "image_blur": {"image"},
        "radar_noise": {"radar_ra", "radar_da"},
        "lidar_sparsify": {"lidar"},
    }[spec.name]
    for key, original in _batch().items():
        assert torch.equal(first[key], second[key])
        assert torch.equal(first[key], explicit_none[key])
        if key not in affected:
            assert torch.equal(first[key], original)
    assert any(not torch.equal(first[key], _batch()[key]) for key in affected)


@pytest.mark.parametrize("spec", CORRUPTION_GRID)
def test_inference_corruption_selector_changes_only_selected_temporal_cells(
    spec: CorruptionSpec,
) -> None:
    original = _batch()
    selector = torch.zeros((2, 5), dtype=torch.bool)
    selector[0, 1] = True
    selector[1, 3] = True
    result = apply_inference_corruption(
        {key: value.clone() for key, value in original.items()},
        spec,
        seed=7,
        batch_index=2,
        selector=selector,
        gps_scaler_mean=GPS_MEAN,
        gps_scaler_scale=GPS_SCALE,
    )
    affected = {
        "gps_noise": {"gps"},
        "image_occlusion": {"image"},
        "image_blur": {"image"},
        "radar_noise": {"radar_ra", "radar_da"},
        "lidar_sparsify": {"lidar"},
    }[spec.name]

    selected_change = False
    for key, value in result.items():
        if key not in affected:
            assert torch.equal(value, original[key])
            continue
        changed = (value != original[key]).reshape(2, 5, -1).any(dim=-1)
        assert not bool(changed[~selector].any())
        selected_change |= bool(changed[selector].any())
    assert selected_change


def test_inference_corruption_selector_requires_batch_time_shape() -> None:
    with pytest.raises(ValueError, match=r"selector must have shape \(2, 5\)"):
        apply_inference_corruption(
            _batch(),
            CorruptionSpec("image_blur", 1),
            seed=7,
            batch_index=2,
            selector=torch.ones(5, dtype=torch.bool),
        )


def test_gps_noise_is_applied_in_xy_metres_and_round_trips_feature_convention() -> None:
    batch = _batch()
    physical = batch["gps"] * GPS_SCALE + GPS_MEAN
    physical[..., 1] = 0.0
    physical[..., 2] = 1.0
    batch["gps"] = (physical - GPS_MEAN) / GPS_SCALE
    result = apply_inference_corruption(
        batch,
        CorruptionSpec("gps_noise", 2),
        seed=7,
        batch_index=2,
        gps_scaler_mean=GPS_MEAN,
        gps_scaler_scale=GPS_SCALE,
    )
    restored = result["gps"] * GPS_SCALE + GPS_MEAN
    assert torch.allclose(torch.linalg.vector_norm(restored[..., 1:3], dim=-1), torch.ones_like(restored[..., 0]))
    assert bool((restored[..., 0] > 0).all())


def test_lidar_sparsification_uses_one_spatial_mask_across_channels() -> None:
    batch = _batch()
    batch["lidar"] = torch.ones_like(batch["lidar"])
    result = apply_inference_corruption(batch, CorruptionSpec("lidar_sparsify", 2), seed=7, batch_index=2)
    assert torch.equal(result["lidar"][:, :, 0], result["lidar"][:, :, 1])
    assert torch.equal(result["lidar"][:, :, 1], result["lidar"][:, :, 2])


def test_gps_noise_requires_train_fit_scaler() -> None:
    with pytest.raises(ValueError, match="gps_scaler"):
        apply_inference_corruption(_batch(), CorruptionSpec("gps_noise", 1), seed=7, batch_index=2)


def test_gps_noise_accepts_collated_batch_scaler() -> None:
    result = apply_inference_corruption(
        _batch(),
        CorruptionSpec("gps_noise", 1),
        seed=7,
        batch_index=2,
        gps_scaler_mean=GPS_MEAN.repeat(2, 1),
        gps_scaler_scale=GPS_SCALE.repeat(2, 1),
    )
    assert result["gps"].shape == (2, 5, 3)
    assert torch.isfinite(result["gps"]).all()
