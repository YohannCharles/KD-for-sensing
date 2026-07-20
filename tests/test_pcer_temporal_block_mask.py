import torch

from kd_sensing.data.temporal_block_mask import (
    PCER_STABLE_PROBABILITIES,
    PCER_TRANSITION_PROBABILITIES,
    PCER_WARMUP_PROBABILITIES,
    TemporalBlockMaskGenerator,
    pcer_curriculum_probabilities,
)
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch, apply_training_temporal_missing
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY


def _generate(mask_type: str, *, variant: int = 0, source_frame_ids=None) -> torch.Tensor:
    result = TemporalBlockMaskGenerator(17)(
        batch_size=2,
        num_modalities=4,
        num_timesteps=5,
        sample_ids=("sample-a", "sample-b"),
        mask_type=mask_type,
        severity=None,
        seed=19,
        training=False,
        source_frame_ids=source_frame_ids,
        variant_ids=variant,
    )
    return result["availability_mask"]


def test_six_masks_are_deterministic_and_preserve_expected_semantics() -> None:
    for kind in (
        "full",
        "sparse_easy",
        "single_modality_burst2",
        "single_modality_missing",
        "latest_sync_missing",
        "two_modality_recent_async",
    ):
        first = _generate(kind, variant=3)
        second = _generate(kind, variant=3)
        assert first.dtype == torch.bool
        assert first.shape == (2, 4, 5)
        assert torch.equal(first, second)
        assert first.any(dim=(1, 2)).all()

    assert (~_generate("sparse_easy")).sum(dim=(1, 2)).eq(2).all()
    burst = ~_generate("single_modality_burst2", variant=4)
    assert burst.sum(dim=(1, 2)).eq(2).all()
    assert (~_generate("single_modality_missing", variant=2))[:, 2].all()
    latest = _generate("latest_sync_missing")
    assert (~latest[:, :, -1]).all() and latest[:, :, :-1].all()
    asynchronous = ~_generate("two_modality_recent_async", variant=1)
    assert asynchronous.sum(dim=(1, 2)).eq(4).all()


def test_source_frame_replicas_are_group_masked() -> None:
    groups = [[[f"{m}-{t}" for t in range(5)] for m in range(4)] for _ in range(2)]
    groups[0][0][1] = groups[0][0][0]
    mask = _generate("single_modality_burst2", variant=0, source_frame_ids=groups)
    assert not bool(mask[0, 0, 0])
    assert not bool(mask[0, 0, 1])


def test_curriculum_has_frozen_three_phase_probabilities() -> None:
    assert pcer_curriculum_probabilities(0, 40) == PCER_WARMUP_PROBABILITIES
    assert pcer_curriculum_probabilities(4, 40) == PCER_TRANSITION_PROBABILITIES
    assert pcer_curriculum_probabilities(12, 40) == PCER_STABLE_PROBABILITIES


def test_applying_block_mask_zeros_inputs_and_writes_availability() -> None:
    batch = {
        "image": torch.ones(2, 5, 1),
        "radar_ra": torch.ones(2, 5, 1),
        "radar_da": torch.ones(2, 5, 1),
        "gps": torch.ones(2, 5, 1),
        "lidar": torch.ones(2, 5, 1),
    }
    mask = _generate("latest_sync_missing").permute(0, 2, 1)
    apply_modality_temporal_mask_to_batch(batch, mask)
    for key in ("image", "radar_ra", "radar_da", "gps", "lidar"):
        assert not bool(batch[key][:, -1].any())
    assert torch.equal(batch["available_modalities"], mask.any(dim=1))


def test_pcer_curriculum_preserves_full_superset_payload() -> None:
    batch = {
        "sample_id": ["sample-a", "sample-b"],
        "image": torch.ones(2, 5, 1),
        "radar_ra": torch.ones(2, 5, 1),
        "radar_da": torch.ones(2, 5, 1),
        "gps": torch.ones(2, 5, 1),
        "lidar": torch.ones(2, 5, 1),
    }
    cfg = {
        "experiment": {"seed": 1},
        "model": {"primary": {"modalities": ["image", "radar", "gps", "lidar"]}},
        "data": {"dataloader": {"train_batch_size": 2}},
        "training": {"epochs": 16},
        "temporal_missing": {
            "enabled": True,
            "mode": "pcer_curriculum",
            "seed": 1,
            "preserve_unmasked_for_superset": True,
        },
    }
    result = apply_training_temporal_missing(batch, cfg, epoch=8, step=0)
    assert result["modality_temporal_mask"].shape == (2, 5, 4)
    assert result["temporal_missing_metadata"]["condition_ids"]
    assert result[TEMPORAL_SUPERSET_PAYLOAD_KEY]["base_mask"].all()
