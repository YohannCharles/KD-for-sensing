import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "analyze_mmw_router_joint_static_prior",
    ROOT / "scripts/analyze_mmw_router_joint_static_prior.py",
)
assert SPEC is not None and SPEC.loader is not None
STATIC = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, STATIC)
SPEC.loader.exec_module(STATIC)


def test_static_prior_normalization_and_fusion_exclude_unavailable_modalities() -> None:
    weights = np.asarray([[0.2, 0.3, 0.1, 0.4]], dtype=np.float32)
    available = np.asarray([[True, False, True, False]])
    logits = np.zeros((1, 4, 3), dtype=np.float32)
    logits[0, 0, 0] = 3.0
    logits[0, 2, 2] = 3.0

    normalized = STATIC.normalize_available_weights(weights, available)
    fused = STATIC.fuse_logits(normalized, logits)

    assert np.allclose(normalized, [[2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0]])
    assert np.allclose(fused, [[2.0, 0.0, 1.0]])


def test_static_prior_claim_requires_positive_combined_lower_bounds() -> None:
    positive = [
        {
            "scope": "Joint40_60_80Combined",
            "control": "global_clean_prior",
            "metric": metric,
            "ci_low": 0.01,
        }
        for metric in ("adba", "normalized_gain")
    ]
    negative = [dict(row) for row in positive]
    negative[1]["ci_low"] = -0.001

    assert STATIC.claim_decision(positive)["dynamic_adaptation_supported"] is True
    decision = STATIC.claim_decision(negative)
    assert decision["dynamic_adaptation_supported"] is False
    assert decision["claim"] == "learned_non_uniform_fusion_only"


def test_paired_bootstrap_is_deterministic() -> None:
    deltas = np.linspace(-0.01, 0.02, 15)

    first = STATIC.paired_bootstrap(deltas, iterations=500, seed=7)
    second = STATIC.paired_bootstrap(deltas, iterations=500, seed=7)

    assert first == second
    assert first[0] < deltas.mean() < first[1]
