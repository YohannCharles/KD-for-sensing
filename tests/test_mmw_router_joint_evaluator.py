import importlib.util
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "eval_mmw_router_joint_stress", ROOT / "scripts/eval_mmw_router_joint_stress.py"
)
assert SPEC is not None and SPEC.loader is not None
JOINT = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, JOINT)
SPEC.loader.exec_module(JOINT)


def test_compose_joint_batch_keeps_clean_corrupts_selected_and_drops_selected() -> None:
    clean = {
        "image": torch.full((2, 5, 3, 2, 2), 1.0),
        "radar_ra": torch.full((2, 5, 1, 2, 2), 2.0),
        "radar_da": torch.full((2, 5, 1, 2, 2), 3.0),
        "gps": torch.full((2, 5, 3), 4.0),
        "lidar": torch.full((2, 5, 3, 2, 2), 5.0),
    }
    corrupted = {key: value + 10.0 for key, value in clean.items()}
    # t0/image=Drop, t1/radar=Corrupt, t2/gps=Drop,
    # t3/lidar=Corrupt; every other cell remains Clean.
    states = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]

    result = JOINT.compose_joint_batch(clean, corrupted, states)

    assert torch.count_nonzero(result["image"][:, 0]) == 0
    assert torch.equal(result["radar_ra"][:, 1], corrupted["radar_ra"][:, 1])
    assert torch.equal(result["radar_da"][:, 1], corrupted["radar_da"][:, 1])
    assert torch.count_nonzero(result["gps"][:, 2]) == 0
    assert torch.equal(result["lidar"][:, 3], corrupted["lidar"][:, 3])
    assert torch.equal(result["image"][:, 4], clean["image"][:, 4])
    assert torch.equal(result["gps"][:, 4], clean["gps"][:, 4])
    assert result["modality_temporal_mask"].shape == (2, 5, 4)
    assert not bool(result["modality_temporal_mask"][:, 0, 0].any())
    assert not bool(result["modality_temporal_mask"][:, 2, 2].any())
    assert bool(result["modality_temporal_mask"][:, 1, 1].all())
    assert bool(result["modality_temporal_mask"][:, 3, 3].all())
    assert bool(result["available_modalities"].all())


def test_normalize_available_weights_excludes_unavailable_modalities() -> None:
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    available = torch.tensor([[True, False, True, False]])

    normalized = JOINT.normalize_available_weights(weights, available)

    assert torch.equal(normalized[:, [1, 3]], torch.zeros(1, 2))
    assert torch.allclose(normalized[:, [0, 2]], torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(normalized.sum(dim=1), torch.ones(1))


def test_dynamic_router_inner_config_is_accepted_but_claim_drift_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pth"
    checkpoint.touch()
    config = {
        "model": {"primary": {"router_variant": "h2r"}},
        "training": {"epochs": 10},
        "mmw_dynamic_router_screen": {
            "protocol": "mmw_dynamic_router_screen_v1",
            "candidate": "H2R-Label",
            "router_variant": "h2r",
            "supervision": "label_topology",
            "selection_split": "frozen_inner_validation_only",
            "seed": 1,
            "claim_eligible": False,
        },
    }
    assert JOINT.validate_router_screen_config(config, checkpoint) == "H2R-Label"
    config["mmw_dynamic_router_screen"]["claim_eligible"] = True
    with pytest.raises(ValueError, match="claim-ineligible dynamic Router"):
        JOINT.validate_router_screen_config(config, checkpoint)


def test_dynamic_power_evaluation_rejects_pre_ampfix_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pth"
    checkpoint.touch()
    config = {
        "model": {"primary": {"router_variant": "h2r"}},
        "training": {"epochs": 10},
        "mmw_dynamic_router_screen": {
            "protocol": "mmw_dynamic_router_screen_v1",
            "candidate": "H2R-Power",
            "router_variant": "h2r",
            "supervision": "beam_power",
            "selection_split": "frozen_inner_validation_only",
            "seed": 1,
            "claim_eligible": False,
        },
    }
    with pytest.raises(ValueError, match="rejects pre-fix"):
        JOINT.validate_router_screen_config(config, checkpoint)

    config["mmw_dynamic_router_screen"].update(
        utility_numeric_policy=JOINT.UTILITY_NUMERIC_POLICY,
        router_reliability_source_sha256=JOINT.sha256(JOINT.ROUTER_RELIABILITY_SOURCE),
    )
    assert JOINT.validate_router_screen_config(config, checkpoint) == "H2R-Power"
