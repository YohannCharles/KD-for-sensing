from pathlib import Path

import torch

from kd_sensing.config import load_config
from kd_sensing.engine.batch_step import _respect_temporal_availability
from kd_sensing.engine.training_extensions import ForwardControls
from kd_sensing.data.temporal_missing import (
    TEMPORAL_SUPERSET_PAYLOAD_KEY,
    apply_training_temporal_missing,
    masked_temporal_mean,
)


ROOT = Path(__file__).resolve().parents[1]


def test_t2_recipe_declares_the_retained_temporal_protocol() -> None:
    cfg = load_config(ROOT / "configs/mmw/t2.yaml")

    assert cfg["temporal_missing"]["enabled"] is True
    assert cfg["model"]["primary"]["temporal_pooling"] == {"enabled": True, "type": "masked_mean"}


def test_masked_temporal_mean_ignores_unavailable_cells() -> None:
    values = torch.tensor([[[1.0], [10.0], [100.0]], [[5.0], [7.0], [9.0]]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    assert torch.allclose(masked_temporal_mean(values, mask), torch.tensor([[50.5], [7.0]]))


def test_t2_temporal_missing_preserves_only_the_same_model_superset_payload() -> None:
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

    s1_batch = {
        key: value.clone()
        for key, value in payload["inputs"].items()
    }
    cfg["temporal_missing"]["preserve_unmasked_for_superset"] = False
    apply_training_temporal_missing(s1_batch, cfg, epoch=0, step=0)
    assert TEMPORAL_SUPERSET_PAYLOAD_KEY not in s1_batch


def test_t2_random_modality_mask_keeps_a_temporally_available_modality() -> None:
    controls = ForwardControls(
        model_kwargs={"missing_mask": torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.bool)}
    )
    batch = {"available_modalities": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)}

    result = _respect_temporal_availability(controls, batch)

    assert torch.equal(
        result.model_kwargs["missing_mask"],
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
    )
