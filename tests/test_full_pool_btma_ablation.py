import numpy as np

from kd_sensing.baselines.btma_assignment import (
    BTMA_METHODS,
    fixed_proportion_assignment,
    random_balanced_scores,
    score_assignment,
)


def test_btma_has_exactly_the_six_preregistered_branches():
    assert BTMA_METHODS == (
        "b0_random_balanced", "b1_fixed_weak_schedule", "b2_kl_capacity",
        "b3_topology_risk_only", "b4_margin_only", "b5_risk_margin_full",
    )


def test_b0_hash_random_scores_are_repeatable_and_balanced_by_capacity():
    ids = [f"sample-{index:05d}" for index in range(1000)]
    first = random_balanced_scores(ids)
    second = random_balanced_scores(ids)
    assert np.array_equal(first, second)
    from kd_sensing.baselines.full_pool_candidate12 import capacity_constrained_assignment
    assigned = capacity_constrained_assignment(first, ids)
    counts = np.bincount(assigned, minlength=4)
    assert np.all(counts >= 150)
    assert np.all(counts <= 400)


def test_b1_uses_exact_historical_global_quota_without_scores():
    ids = [f"sample-{index:05d}" for index in range(37038)]
    assigned = fixed_proportion_assignment(ids, {"image": 1 / 37038, "lidar": 0.0, "radar": 20730 / 37038, "gps": 16307 / 37038})
    assert np.bincount(assigned, minlength=4).tolist() == [1, 0, 20730, 16307]
    assert np.array_equal(assigned, fixed_proportion_assignment(ids, {"image": 1 / 37038, "lidar": 0.0, "radar": 20730 / 37038, "gps": 16307 / 37038}))


def test_b2_to_b5_use_only_the_declared_scores_and_capacity():
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(1000, 4, 64))
    features = rng.normal(size=(1000, 4, 8))
    prototypes = rng.normal(size=(64, 8))
    labels = np.arange(1000) % 64
    distance = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :]) / 63
    ids = [f"sample-{index:05d}" for index in range(1000)]
    for method in ("b2_kl_capacity", "b3_topology_risk_only", "b4_margin_only", "b5_risk_margin_full"):
        scores, assigned, diagnostics = score_assignment(method, logits=logits, features=features, prototypes=prototypes, labels=labels, topology_distance=distance, sample_ids=ids)
        assert scores.shape == (1000, 4)
        assert diagnostics["risk_rank"].shape == (1000, 4)
        assert np.all(np.bincount(assigned, minlength=4) >= 150)
        assert np.all(np.bincount(assigned, minlength=4) <= 400)
