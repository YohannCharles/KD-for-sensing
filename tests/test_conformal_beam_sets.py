"""Correctness tests for the split-conformal beam set primitives.

The diagnostic these back is cheap to run and easy to misread, so the two
properties that would silently invalidate it -- a wrong finite-sample quantile
and trajectory leakage across the calibration/test split -- are asserted here
rather than eyeballed in the output table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kd_sensing.baselines import conformal_beam_sets as cbs


def _sample_ids(tracks: int, frames: int) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    domains: list[str] = []
    for track in range(tracks):
        domain = f"foggy/Town{track // 2:02d}_scene"
        for frame in range(frames):
            ids.append(f"mmw:foggy:scene:validation:foggy:Town:scene:cav_{track}:{frame:06d}")
            domains.append(domain)
    return ids, domains


class TestTrackIdentity:
    def test_frames_of_one_agent_share_a_track(self) -> None:
        first = cbs.track_key("mmw:a:b:validation:a:T:b:cav_1:000100", "foggy/scene")
        second = cbs.track_key("mmw:a:b:validation:a:T:b:cav_1:000101", "foggy/scene")
        assert first == second

    def test_different_agents_in_one_scene_are_different_tracks(self) -> None:
        first = cbs.track_key("mmw:a:b:validation:a:T:b:cav_1:000100", "foggy/scene")
        second = cbs.track_key("mmw:a:b:validation:a:T:b:cav_2:000100", "foggy/scene")
        assert first != second

    def test_same_agent_id_in_different_domains_stays_separate(self) -> None:
        first = cbs.track_key("mmw:a:b:validation:a:T:b:cav_1:000100", "foggy/scene_a")
        second = cbs.track_key("mmw:a:b:validation:a:T:b:cav_1:000100", "clear/scene_b")
        assert first != second

    def test_track_ids_are_dense_and_stable(self) -> None:
        ids, domains = _sample_ids(tracks=4, frames=3)
        tracks = cbs.track_ids(ids, domains)
        assert sorted(set(tracks.tolist())) == [0, 1, 2, 3]
        assert np.array_equal(tracks, cbs.track_ids(ids, domains))


class TestBlockSplit:
    def test_no_track_appears_on_both_sides(self) -> None:
        ids, domains = _sample_ids(tracks=20, frames=16)
        calibration = cbs.block_split(ids, domains, seed=7)
        tracks = cbs.track_ids(ids, domains)
        assert not set(tracks[calibration].tolist()) & set(tracks[~calibration].tolist())

    def test_split_is_close_to_the_requested_fraction(self) -> None:
        ids, domains = _sample_ids(tracks=40, frames=8)
        calibration = cbs.block_split(ids, domains, seed=3, fraction=0.5)
        assert 0.4 <= calibration.mean() <= 0.6

    def test_split_is_deterministic_for_a_seed(self) -> None:
        ids, domains = _sample_ids(tracks=12, frames=5)
        assert np.array_equal(
            cbs.block_split(ids, domains, seed=11), cbs.block_split(ids, domains, seed=11)
        )

    def test_different_seeds_give_different_splits(self) -> None:
        ids, domains = _sample_ids(tracks=30, frames=4)
        assert not np.array_equal(
            cbs.block_split(ids, domains, seed=1), cbs.block_split(ids, domains, seed=2)
        )

    def test_a_single_track_cannot_be_split(self) -> None:
        ids, domains = _sample_ids(tracks=1, frames=10)
        with pytest.raises(ValueError, match="two trajectories"):
            cbs.block_split(ids, domains)

    def test_fraction_outside_the_open_interval_is_rejected(self) -> None:
        ids, domains = _sample_ids(tracks=4, frames=4)
        for fraction in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="strictly inside"):
                cbs.block_split(ids, domains, fraction=fraction)


class TestRandomSplitControl:
    def test_it_does_leak_tracks_which_is_the_whole_point(self) -> None:
        ids, domains = _sample_ids(tracks=20, frames=16)
        calibration = cbs.random_split(len(ids), seed=5)
        tracks = cbs.track_ids(ids, domains)
        shared = set(tracks[calibration].tolist()) & set(tracks[~calibration].tolist())
        assert shared, "The control must leak, otherwise it is not a control."

    def test_it_hits_the_requested_fraction_exactly(self) -> None:
        calibration = cbs.random_split(1000, seed=5, fraction=0.4)
        assert int(calibration.sum()) == 400

    def test_it_rejects_a_degenerate_input(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            cbs.random_split(1)
        with pytest.raises(ValueError, match="strictly inside"):
            cbs.random_split(100, fraction=1.0)


class TestConformalQuantile:
    def test_uses_the_finite_sample_rank_not_the_plain_percentile(self) -> None:
        # n=9, alpha=0.1 -> ceil(10 * 0.9) = 9, so the threshold is the maximum.
        scores = np.arange(9, dtype=np.float64)
        assert cbs.conformal_quantile(scores, alpha=0.1) == 8.0
        # numpy's plain quantile would land below it; the difference is the point.
        assert np.quantile(scores, 0.9) < 8.0

    def test_returns_infinity_when_no_finite_threshold_can_promise_the_level(self) -> None:
        # n=5, alpha=0.1 -> ceil(6 * 0.9) = 6 > 5.
        assert math.isinf(cbs.conformal_quantile(np.arange(5, dtype=np.float64), alpha=0.1))

    def test_rejects_an_empty_calibration_set(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cbs.conformal_quantile(np.empty(0))

    def test_rejects_a_degenerate_alpha(self) -> None:
        for alpha in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="strictly inside"):
                cbs.conformal_quantile(np.arange(50, dtype=np.float64), alpha=alpha)


class TestCoverageValidity:
    def test_exchangeable_scores_reach_the_nominal_level(self) -> None:
        """The guarantee must hold for an arbitrary, even useless, score.

        The band is wide relative to the theory interval on purpose: with 150
        trials the Monte Carlo standard error is around 0.001, so a tighter
        assertion would flake without catching anything the deterministic rank
        test above does not already catch.
        """
        rng = np.random.default_rng(0)
        alpha = 0.1
        achieved = []
        for _ in range(150):
            probabilities = rng.dirichlet(np.ones(16), size=2000)
            labels = rng.integers(0, 16, size=2000)
            scores = cbs.nonconformity(probabilities)
            calibration = np.zeros(2000, dtype=bool)
            calibration[:1000] = True
            threshold = cbs.conformal_quantile(
                cbs.true_beam_scores(scores[calibration], labels[calibration]), alpha
            )
            achieved.append(cbs.coverage(scores[~calibration], labels[~calibration], threshold))
        assert 0.893 <= float(np.mean(achieved)) <= 0.908

    def test_a_sharper_score_gives_the_same_coverage_but_smaller_sets(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 16, size=800)
        sharp = np.full((800, 16), 0.01)
        sharp[np.arange(800), labels] = 0.85
        sharp /= sharp.sum(axis=1, keepdims=True)
        blunt = np.full((800, 16), 1.0 / 16.0)
        calibration = np.zeros(800, dtype=bool)
        calibration[:400] = True

        sizes = []
        for probabilities in (sharp, blunt):
            scores = cbs.nonconformity(probabilities)
            threshold = cbs.conformal_quantile(
                cbs.true_beam_scores(scores[calibration], labels[calibration]), alpha=0.1
            )
            assert cbs.coverage(scores[~calibration], labels[~calibration], threshold) >= 0.9
            sizes.append(float(np.mean(cbs.set_sizes(scores[~calibration], threshold))))
        assert sizes[0] < sizes[1]


class TestStratumThresholds:
    def test_each_stratum_gets_its_own_threshold(self) -> None:
        strata = np.array(["a"] * 100 + ["b"] * 100)
        scores = np.concatenate([np.linspace(0.0, 0.1, 100), np.linspace(0.5, 0.9, 100)])
        calibration = np.tile(np.repeat([True, False], 50), 2)
        thresholds, unseen = cbs.stratum_thresholds(
            scores, strata, calibration, alpha=0.1, fallback=99.0
        )
        assert not unseen.any()
        assert thresholds[strata == "a"].max() < thresholds[strata == "b"].min()

    def test_a_stratum_without_calibration_data_falls_back_and_is_flagged(self) -> None:
        strata = np.array(["seen"] * 100 + ["unseen"] * 40)
        scores = np.linspace(0.0, 1.0, 140)
        calibration = np.concatenate([np.repeat([True, False], 50), np.zeros(40, dtype=bool)])
        thresholds, unseen = cbs.stratum_thresholds(
            scores, strata, calibration, alpha=0.1, fallback=0.42
        )
        assert unseen.tolist() == [False] * 100 + [True] * 40
        assert np.all(thresholds[strata == "unseen"] == 0.42)
        assert np.all(thresholds[strata == "seen"] != 0.42)

    def test_the_fallback_is_finite_so_an_unseen_scene_costs_coverage(self) -> None:
        """An infinite fallback would report perfect coverage on unseen scenes."""
        strata = np.array(["unseen"] * 20)
        thresholds, unseen = cbs.stratum_thresholds(
            np.linspace(0.0, 1.0, 20), strata, np.zeros(20, dtype=bool), alpha=0.1, fallback=0.3
        )
        assert unseen.all()
        assert np.isfinite(thresholds).all()

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            cbs.stratum_thresholds(
                np.zeros(10), np.zeros(9), np.ones(10, dtype=bool), alpha=0.1, fallback=0.0
            )

    def test_stratified_coverage_beats_pooled_when_strata_differ(self) -> None:
        """The whole point: pooling two unequal strata under-covers the hard one."""
        rng = np.random.default_rng(4)
        easy = rng.random(2000) * 0.2
        hard = rng.random(2000) * 0.2 + 0.8
        scores = np.concatenate([easy, hard])
        strata = np.array(["easy"] * 2000 + ["hard"] * 2000)
        calibration = np.tile(np.repeat([True, False], 1000), 2)
        test = ~calibration

        pooled = cbs.conformal_quantile(scores[calibration], alpha=0.1)
        stratified, _ = cbs.stratum_thresholds(
            scores, strata, calibration, alpha=0.1, fallback=pooled
        )
        for name in ("easy", "hard"):
            member = test & (strata == name)
            assert np.mean(scores[member] <= stratified[member]) >= 0.88
        assert np.mean(scores[test & (strata == "hard")] <= pooled) < 0.85


class TestScoreHelpers:
    def test_nonconformity_rejects_a_non_matrix(self) -> None:
        with pytest.raises(ValueError, match=r"\[N, beams\]"):
            cbs.nonconformity(np.ones(8))

    def test_true_beam_scores_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            cbs.true_beam_scores(np.ones((4, 16)), np.zeros(5, dtype=np.int64))

    def test_set_sizes_counts_beams_below_the_threshold(self) -> None:
        scores = np.array([[0.1, 0.5, 0.9], [0.2, 0.3, 0.4]])
        assert cbs.set_sizes(scores, 0.45).tolist() == [1, 3]

    def test_an_infinite_threshold_keeps_every_beam(self) -> None:
        scores = np.random.default_rng(2).random((10, 16))
        assert cbs.set_sizes(scores, math.inf).tolist() == [16] * 10
