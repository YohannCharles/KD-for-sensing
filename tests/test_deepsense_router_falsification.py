import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from kd_sensing.data.mmw.twc_evidence import build_fixed_mask_cache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "eval_deepsense_router_falsification",
    ROOT / "scripts/eval_deepsense_router_falsification.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, MODULE)
SPEC.loader.exec_module(MODULE)


def test_falsification_cache_is_fixed_unique_and_exactly_balanced() -> None:
    parent = build_fixed_mask_cache(seed=20260718)

    cache = MODULE.build_falsification_cache(parent)
    replay = MODULE.build_falsification_cache(parent)

    assert cache == replay
    assert len(cache["conditions"]) == 65
    assert [item["drop_count"] for item in cache["conditions"][:15]].count(3) == 4
    assert all(item["exact_cell_balance"] for item in cache["rate_balance_audit"])
    for audit in cache["rate_balance_audit"]:
        retained = np.asarray(audit["retained_per_cell"])
        assert retained.shape == (5, 4)
        assert np.all(retained == retained[0, 0])


def test_static_prior_branch_masks_and_renormalizes_unavailable_modalities() -> None:
    logits = np.asarray([[[4.0, 0.0], [0.0, 2.0], [1_000.0, 0.0]]])
    available = np.asarray([[True, True, False]])

    fused, weights = MODULE.static_prior_branch(logits, available, np.asarray([0.6, 0.3, 0.1]))

    assert np.allclose(weights, [[2 / 3, 1 / 3, 0.0]])
    assert np.allclose(fused, [[4 * 2 / 3, 2 * 1 / 3]])


def test_decision_requires_static_prior_gain_scene_consistency_and_shift_alignment() -> None:
    rows = []
    cells = ("Clean", "Drop2", "Drop3", "Token60", "Token80", "Token90")
    for scope in ("pooled", "day", "night", "scene31", "scene32", "scene33", "scene34"):
        for cell in cells:
            for fusion, delta in (("global_clean_prior", 0.0), ("learned", 0.01)):
                rows.append(
                    {
                        "scope": scope,
                        "cell": cell,
                        "fusion": fusion,
                        "adba": 0.8 + delta,
                        "normalized_gain": 0.7 + delta,
                        **{f"weight_{name}": value for name, value in zip(MODULE.MODALITIES, (0.4, 0.2, 0.2, 0.2))},
                        **{
                            f"oracle_frequency_{name}": value
                            for name, value in zip(MODULE.MODALITIES, (0.4, 0.2, 0.2, 0.2))
                        },
                    }
                )
    for row in rows:
        if row["scope"] == "night" and row["cell"] == "Clean" and row["fusion"] == "learned":
            row["weight_image"], row["weight_lidar"] = 0.2, 0.4
            row["oracle_frequency_image"], row["oracle_frequency_lidar"] = 0.2, 0.4

    decision = MODULE.build_decision(rows)

    assert decision["recommendation"] == "resume_formal_40epoch_router_evidence"
    assert decision["positive_scene_count"] == 4
    assert decision["day_night_weight_oracle_shift_cosine"] == pytest.approx(1.0)
