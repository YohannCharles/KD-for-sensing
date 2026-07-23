from collections import Counter
from pathlib import Path

import pytest
import torch

from kd_sensing.config import load_config
from kd_sensing.engine.batch_step import _respect_temporal_availability
from kd_sensing.engine.training_extensions import ForwardControls
from kd_sensing.data.temporal_missing import (
    BALANCED_PATTERN_CONDITION_COUNTS,
    BALANCED_PATTERN_PANEL_SIZE,
    BALANCED_PATTERN_SCHEDULE_ID,
    WHOLE_ONLY_PATTERN_CONDITION_COUNTS,
    WHOLE_ONLY_PATTERN_PANEL_SIZE,
    WHOLE_ONLY_PATTERN_SCHEDULE_ID,
    WHOLE_ONLY_PATTERN_SEED_ALGORITHM,
    TEMPORAL_SUPERSET_PAYLOAD_KEY,
    apply_training_temporal_missing,
    masked_temporal_mean,
)


ROOT = Path(__file__).resolve().parents[1]


def _temporal_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "image": torch.ones(batch_size, 5, 1),
        "radar_ra": torch.ones(batch_size, 5, 1),
        "radar_da": torch.ones(batch_size, 5, 1),
        "gps": torch.ones(batch_size, 5, 1),
        "lidar": torch.ones(batch_size, 5, 1),
    }


def test_u0_recipe_declares_the_retained_temporal_protocol() -> None:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")

    assert cfg["temporal_missing"]["enabled"] is True
    assert cfg["model"]["primary"]["temporal_pooling"] == {"enabled": True, "type": "masked_mean"}


def test_masked_temporal_mean_ignores_unavailable_cells() -> None:
    values = torch.tensor([[[1.0], [10.0], [100.0]], [[5.0], [7.0], [9.0]]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    assert torch.allclose(masked_temporal_mean(values, mask), torch.tensor([[50.5], [7.0]]))


def test_u0_temporal_missing_preserves_only_the_same_model_superset_payload() -> None:
    batch = {
        "image": torch.ones(2, 5, 3, 2, 2),
        "radar_ra": torch.ones(2, 5, 1, 2, 2),
        "radar_da": torch.ones(2, 5, 1, 2, 2),
        "gps": torch.ones(2, 5, 3),
        "lidar": torch.ones(2, 5, 3, 2, 2),
    }
    cfg = {
        "experiment": {"seed": 7},
        "model": {"primary": {"modalities": ["image", "radar", "gps", "lidar"]}},
        "temporal_missing": {
            "enabled": True,
            "mode": "stratified_modality_temporal",
            "train_missing_drop_counts": "3",
            "train_temporal_missing_rates": "0.8",
            "train_temporal_missing_types": "modality_frame",
            "preserve_unmasked_for_superset": True,
        },
    }

    apply_training_temporal_missing(batch, cfg, epoch=0, step=0)

    mask = batch["modality_temporal_mask"]
    payload = batch[TEMPORAL_SUPERSET_PAYLOAD_KEY]
    assert mask.shape == (2, 5, 4)
    assert mask.any(dim=(1, 2)).all()
    assert (~mask).any()
    assert torch.equal(payload["inputs"]["image"], torch.ones_like(batch["image"]))
    assert torch.equal(payload["base_mask"], torch.ones_like(mask))

    without_superset = {
        key: value.clone()
        for key, value in payload["inputs"].items()
    }
    cfg["temporal_missing"]["preserve_unmasked_for_superset"] = False
    apply_training_temporal_missing(without_superset, cfg, epoch=0, step=0)
    assert TEMPORAL_SUPERSET_PAYLOAD_KEY not in without_superset


def test_u0_random_modality_mask_keeps_a_temporally_available_modality() -> None:
    controls = ForwardControls(
        model_kwargs={"missing_mask": torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.bool)}
    )
    batch = {"available_modalities": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)}

    result = _respect_temporal_availability(controls, batch)

    assert torch.equal(
        result.model_kwargs["missing_mask"],
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
    )


def test_balanced_pattern_schedule_has_exact_600_entry_inventory_and_replays() -> None:
    cfg = {
        "experiment": {"seed": 7},
        "data": {"dataloader": {"train_batch_size": BALANCED_PATTERN_PANEL_SIZE}},
        "model": {"primary": {"modalities": list(("image", "radar", "gps", "lidar"))}},
        "temporal_missing": {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "history_window": 5,
            "schedule_id": BALANCED_PATTERN_SCHEDULE_ID,
            "panel_size": BALANCED_PATTERN_PANEL_SIZE,
            "condition_counts": dict(BALANCED_PATTERN_CONDITION_COUNTS),
        },
    }
    first = _temporal_batch(BALANCED_PATTERN_PANEL_SIZE)
    second = _temporal_batch(BALANCED_PATTERN_PANEL_SIZE)

    apply_training_temporal_missing(first, cfg, epoch=2, step=0)
    apply_training_temporal_missing(second, cfg, epoch=2, step=0)

    metadata = first["temporal_missing_metadata"]
    condition_ids = metadata["condition_ids"]
    assert Counter(condition_ids) == BALANCED_PATTERN_CONDITION_COUNTS
    assert metadata["schedule_id"] == BALANCED_PATTERN_SCHEDULE_ID
    assert metadata["panel_size"] == BALANCED_PATTERN_PANEL_SIZE
    assert metadata["condition_counts"] == BALANCED_PATTERN_CONDITION_COUNTS
    assert len(metadata["panel_sha256"]) == 64
    assert torch.equal(first["modality_temporal_mask"], second["modality_temporal_mask"])
    assert metadata["panel_sha256"] == second["temporal_missing_metadata"]["panel_sha256"]

    masks = first["modality_temporal_mask"]
    for drop_count, pattern_count in ((1, 4), (2, 6), (3, 4)):
        selected = masks[[condition == f"drop{drop_count}" for condition in condition_ids]].any(dim=1)
        frequencies = Counter(tuple(row.tolist()) for row in selected)
        assert len(frequencies) == pattern_count
        assert set(frequencies.values()) == {60 // pattern_count}
    for rate in (20, 40, 60, 80, 90):
        selected = masks[[condition == f"token{rate}" for condition in condition_ids]]
        retained = 20 - round(rate / 100 * 20)
        assert set(selected.flatten(1).sum(dim=1).tolist()) == {retained}
        assert set(selected.sum(dim=0).flatten().tolist()) == {60 * retained // 20}
        assert len({tuple(row.tolist()) for row in selected.flatten(1)}) == 60


def test_balanced_pattern_schedule_rejects_drifted_contract() -> None:
    cfg = {
        "model": {"primary": {"modalities": list(("image", "radar", "gps", "lidar"))}},
        "temporal_missing": {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "history_window": 5,
            "panel_size": 599,
        },
    }

    with pytest.raises(ValueError, match="panel_size"):
        apply_training_temporal_missing(_temporal_batch(1), cfg, epoch=0, step=0)


def test_whole_only_pattern_schedule_has_exact_balanced_whole_modality_inventory() -> None:
    cfg = {
        "experiment": {"seed": 7},
        "data": {"dataloader": {"train_batch_size": WHOLE_ONLY_PATTERN_PANEL_SIZE}},
        "model": {"primary": {"modalities": list(("image", "radar", "gps", "lidar"))}},
        "temporal_missing": {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "history_window": 5,
            "schedule_id": WHOLE_ONLY_PATTERN_SCHEDULE_ID,
            "panel_size": WHOLE_ONLY_PATTERN_PANEL_SIZE,
            "condition_counts": dict(WHOLE_ONLY_PATTERN_CONDITION_COUNTS),
        },
    }
    first = _temporal_batch(WHOLE_ONLY_PATTERN_PANEL_SIZE)
    second = _temporal_batch(WHOLE_ONLY_PATTERN_PANEL_SIZE)

    apply_training_temporal_missing(first, cfg, epoch=2, step=0)
    apply_training_temporal_missing(second, cfg, epoch=2, step=0)

    metadata = first["temporal_missing_metadata"]
    condition_ids = metadata["condition_ids"]
    masks = first["modality_temporal_mask"]
    assert Counter(condition_ids) == Counter({key: value for key, value in WHOLE_ONLY_PATTERN_CONDITION_COUNTS.items() if value})
    assert metadata["schedule_id"] == WHOLE_ONLY_PATTERN_SCHEDULE_ID
    assert metadata["panel_size"] == WHOLE_ONLY_PATTERN_PANEL_SIZE
    assert metadata["condition_counts"] == WHOLE_ONLY_PATTERN_CONDITION_COUNTS
    assert metadata["seed_algorithm"] == WHOLE_ONLY_PATTERN_SEED_ALGORITHM
    assert torch.equal(masks, masks[:, :1].expand_as(masks))
    assert torch.equal(masks, second["modality_temporal_mask"])
    assert metadata["panel_sha256"] == second["temporal_missing_metadata"]["panel_sha256"]

    for drop_count, pattern_count, frequency in ((1, 4, 30), (2, 6, 20), (3, 4, 30)):
        selected = masks[[condition == f"drop{drop_count}" for condition in condition_ids]].any(dim=1)
        frequencies = Counter(tuple(row.tolist()) for row in selected)
        assert len(frequencies) == pattern_count
        assert set(frequencies.values()) == {frequency}
