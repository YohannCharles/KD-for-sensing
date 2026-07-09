import torch

from kd_sensing.config import load_config
from kd_sensing.data.difficulty.pipeline import apply_configured_difficulty
from kd_sensing.data.difficulty.schema import DifficultyContext, normalize_config_difficulty
from kd_sensing.data.temporal_missing import masked_temporal_mean


def _cfg(mode: str, *, prob: float = 1.0, block_len: int = 1, ensure_frame: bool = True) -> dict:
    cfg = {
        "experiment": {"seed": 7},
        "model": {"primary": {"modalities": ["image", "radar", "lidar", "gps"]}},
        "temporal_missing": {
            "enabled": mode != "none",
            "mode": mode,
            "prob": prob,
            "block_len": block_len,
            "apply": "train",
            "seed": 7,
            "ensure_at_least_one_frame": ensure_frame,
            "ensure_at_least_one_modality_per_frame": False,
        },
    }
    normalize_config_difficulty(cfg)
    return cfg


def _batch(batch_size: int = 3, steps: int = 5) -> dict[str, torch.Tensor]:
    image = torch.ones(batch_size, steps, 3, 4, 4)
    return {
        "image": image.clone(),
        "radar_ra": torch.ones(batch_size, steps, 2, 2),
        "radar_da": torch.ones(batch_size, steps, 2, 2) * 2,
        "lidar": torch.ones(batch_size, steps, 3, 4, 4) * 3,
        "gps": torch.ones(batch_size, steps, 3) * 4,
        "input_beam": torch.zeros(batch_size, steps, dtype=torch.long),
        "target_beam": torch.ones(batch_size, 1, dtype=torch.long),
        "history_indices": torch.arange(steps).view(1, steps).expand(batch_size, -1),
        "target_index": torch.full((batch_size,), steps),
    }


def _apply(batch: dict[str, torch.Tensor], cfg: dict) -> dict:
    return apply_configured_difficulty(
        batch,
        cfg,
        DifficultyContext(stage="train", split="train", seed=7, step=0),
    ).batch


def test_history_prediction_window_aliases_sync_config_fields() -> None:
    cfg = load_config(
        None,
        [
            "temporal_missing.history_window=5",
            "temporal_missing.prediction_window=1",
            "temporal_missing.mode=none",
        ],
    )
    assert cfg["temporal_missing"]["history_window"] == 5
    assert cfg["temporal_missing"]["prediction_window"] == 1
    assert cfg["data"]["dataset"]["seq_len"] == 5
    assert cfg["model"]["seq_length"] == 5
    assert cfg["data"]["dataset"]["num_pred"] == 1
    assert cfg["model"]["primary"]["prediction_window"] == 1


def test_default_temporal_window_missing_is_enabled_for_new_experiments() -> None:
    cfg = load_config(None)
    assert cfg["temporal_missing"]["history_window"] == 5
    assert cfg["temporal_missing"]["prediction_window"] == 1
    assert cfg["temporal_missing"]["mode"] == "modality_frame_bernoulli"
    assert cfg["temporal_missing"]["prob"] == 0.2
    assert cfg["data"]["dataset"]["seq_len"] == 5
    assert cfg["model"]["primary"]["num_pred"] == 1


def test_frame_bernoulli_masks_whole_frames_and_zero_fills_inputs() -> None:
    out = _apply(_batch(batch_size=2, steps=5), _cfg("frame_bernoulli", prob=1.0, ensure_frame=True))
    mask = out["modality_temporal_mask"]
    assert mask.shape == (2, 5, 4)
    assert out["temporal_mask"].shape == (2, 5)
    assert bool(mask.any(dim=(1, 2)).all())
    missing_frames = ~out["temporal_mask"]
    assert bool((mask[missing_frames] == 0).all())
    assert torch.equal(out["target_beam"], torch.ones(2, 1, dtype=torch.long))
    assert out["image"][~out["temporal_mask"]].abs().sum().item() == 0.0


def test_modality_frame_bernoulli_keeps_temporal_mask_as_any_modality_available() -> None:
    out = _apply(_batch(batch_size=4, steps=5), _cfg("modality_frame_bernoulli", prob=0.5))
    mtm = out["modality_temporal_mask"]
    assert torch.equal(out["temporal_mask"], mtm.any(dim=2))
    assert torch.equal(out["available_modalities"], mtm.any(dim=1))
    assert mtm.shape == (4, 5, 4)


def test_block_missing_has_contiguous_missing_region_when_no_fallback_needed() -> None:
    out = _apply(_batch(batch_size=1, steps=5), _cfg("block", prob=1.0, block_len=2, ensure_frame=False))
    missing = (~out["temporal_mask"][0]).nonzero(as_tuple=False).flatten().tolist()
    assert len(missing) == 2
    assert missing[1] == missing[0] + 1
    assert out["temporal_missing_metadata"]["num_all_missing_fixed"] == 0


def test_fallback_prevents_all_missing_and_records_fix_count() -> None:
    out = _apply(_batch(batch_size=3, steps=5), _cfg("modality_frame_bernoulli", prob=1.0, ensure_frame=True))
    assert bool(out["modality_temporal_mask"].any(dim=(1, 2)).all())
    assert out["temporal_missing_metadata"]["num_all_missing_fixed"] == 3


def test_existing_modality_mask_combines_with_temporal_missing() -> None:
    batch = _batch(batch_size=2, steps=5)
    batch["image_valid_mask"] = torch.zeros(2, 5, dtype=torch.bool)
    out = _apply(batch, _cfg("modality_frame_bernoulli", prob=0.0))
    assert not bool(out["modality_temporal_mask"][:, :, 0].any())
    assert out["image"].abs().sum().item() == 0.0
    assert bool(out["modality_temporal_mask"][:, :, 1:].any())


def test_temporal_missing_supports_csi_modality() -> None:
    batch = {
        "csi": torch.ones(2, 5, 4, 2, 2),
        "input_beam": torch.zeros(2, 5, dtype=torch.long),
        "target_beam": torch.ones(2, 1, dtype=torch.long),
    }
    cfg = {
        "experiment": {"seed": 7},
        "model": {"primary": {"modalities": ["csi"]}},
        "temporal_missing": {
            "enabled": True,
            "mode": "modality_frame_bernoulli",
            "prob": 1.0,
            "apply": "train",
            "seed": 7,
        },
    }
    normalize_config_difficulty(cfg)
    out = _apply(batch, cfg)
    assert out["modality_temporal_mask"].shape == (2, 5, 1)
    assert out["csi_valid_mask"].shape == (2, 5)
    assert bool(out["modality_temporal_mask"].any(dim=(1, 2)).all())
    assert out["temporal_missing_metadata"]["modalities"] == ["csi"]


def test_masked_temporal_mean_ignores_missing_frames() -> None:
    values = torch.tensor([[[1.0], [10.0], [100.0]], [[5.0], [7.0], [9.0]]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    result = masked_temporal_mean(values, mask)
    assert torch.allclose(result, torch.tensor([[50.5], [7.0]]))
    assert not torch.isnan(result).any()
