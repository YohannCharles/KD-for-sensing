"""Split-conformal beam candidate sets over the frozen U0 prototype geometry.

This module is the *diagnostic* stage of the set-valued route.  It answers one
question and nothing else:

    Does a single marginally-calibrated threshold under-cover on the degraded
    masks while over-covering on the full-modality mask?

If it does, conditioning the calibration on the availability pattern is
motivated and quantified.  If every mask already sits at the nominal level,
the conditioning buys nothing and the route should be narrowed rather than
dressed up.

Nothing here trains: scores come from U0's own frozen router via
``FrozenU0Head.reference_logits``.  The nonconformity score is therefore
already a quantity in the prototype metric space, because U0's ``_head_logits``
are cosine logits against ``BeamPrototypeBank``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# The nominal miscoverage rate.  Fixed here rather than passed around so the
# diagnostic cannot be quietly re-run at a level that flatters the result.
ALPHA = 0.1
CALIBRATION_FRACTION = 0.5
SPLIT_SEED = 20260726


def track_key(sample_id: str, domain: str) -> str:
    """Identify the trajectory a frame belongs to.

    Frames of one vehicle in one scene are strongly correlated, so they are the
    unit that must stay together across the calibration/test split.  The agent
    id is the second-to-last colon-separated field of the sample id; the last
    field is the frame number and is deliberately discarded.
    """
    parts = sample_id.split(":")
    if len(parts) < 2:
        raise ValueError(f"Sample id has no agent field: {sample_id!r}")
    return f"{domain}|{parts[-2]}"


def track_ids(sample_ids: Sequence[str], domains: Sequence[str]) -> np.ndarray:
    """Map every sample to an integer trajectory id."""
    if len(sample_ids) != len(domains):
        raise ValueError("sample_ids and domains must be the same length.")
    lookup: dict[str, int] = {}
    out = np.empty(len(sample_ids), dtype=np.int64)
    for position, (sample_id, domain) in enumerate(zip(sample_ids, domains)):
        key = track_key(str(sample_id), str(domain))
        out[position] = lookup.setdefault(key, len(lookup))
    return out


def block_split(
    sample_ids: Sequence[str],
    domains: Sequence[str],
    *,
    seed: int = SPLIT_SEED,
    fraction: float = CALIBRATION_FRACTION,
) -> np.ndarray:
    """Split into calibration/test at trajectory granularity.

    Returns a boolean array that is True for calibration samples.  Splitting at
    frame granularity would put adjacent frames of the same vehicle on both
    sides, which makes the calibration set look like the test set and inflates
    measured coverage.  Whole trajectories are assigned instead, greedily until
    the sample-count target is met -- tracks differ in length, so splitting on
    the *count of tracks* would not give a balanced split of samples.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie strictly inside (0, 1), got {fraction}.")
    tracks = track_ids(sample_ids, domains)
    unique = np.unique(tracks)
    if unique.size < 2:
        raise ValueError("Need at least two trajectories to split without leakage.")
    order = np.random.default_rng(seed).permutation(unique)
    target = fraction * len(tracks)
    calibration = np.zeros(len(tracks), dtype=bool)
    taken = 0
    for track in order:
        if taken >= target:
            break
        member = tracks == track
        calibration |= member
        taken += int(member.sum())
    if not calibration.any() or calibration.all():
        raise ValueError("Trajectory split degenerated to a single side.")
    return calibration


def random_split(
    count: int,
    *,
    seed: int = SPLIT_SEED,
    fraction: float = CALIBRATION_FRACTION,
) -> np.ndarray:
    """Split at frame granularity, ignoring trajectories.

    This is a *control*, not an alternative protocol.  It deliberately lets
    adjacent frames of one vehicle land on both sides, which restores
    exchangeability between the two halves at the price of leakage.  Comparing
    it against :func:`block_split` separates two very different explanations
    for a coverage shortfall: a broken calibration, which would show up under
    both splits, and a genuine shift between trajectories, which shows up only
    under the block split.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie strictly inside (0, 1), got {fraction}.")
    if count < 2:
        raise ValueError("Need at least two samples to split.")
    calibration = np.zeros(count, dtype=bool)
    chosen = np.random.default_rng(seed).permutation(count)[: max(1, int(round(fraction * count)))]
    calibration[chosen] = True
    return calibration


def nonconformity(probabilities: np.ndarray) -> np.ndarray:
    """Score every (sample, beam) pair; larger means less plausible.

    ``1 - p`` is the standard threshold score.  Conformal validity holds for
    any score function, so this choice affects set *size* only, never coverage.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be [N, beams], got {probabilities.shape}.")
    return 1.0 - probabilities


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal threshold with the finite-sample correction.

    The threshold is the ``ceil((n + 1) * (1 - alpha))``-th smallest calibration
    score, not the plain ``1 - alpha`` empirical quantile.  Without the ``n + 1``
    the guarantee is only asymptotic, and on the small per-mask strata this
    diagnostic cares about the difference is exactly where the interesting
    behaviour is.  When the rank exceeds ``n`` no finite threshold can promise
    the level, so the full label set is returned via an infinite threshold --
    a visible degenerate answer rather than a silently optimistic one.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot calibrate on an empty score set.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly inside (0, 1), got {alpha}.")
    rank = math.ceil((values.size + 1) * (1.0 - alpha))
    if rank > values.size:
        return math.inf
    return float(np.sort(values)[rank - 1])


def stratum_thresholds(
    true_scores: np.ndarray,
    strata: np.ndarray,
    calibration: np.ndarray,
    *,
    alpha: float = ALPHA,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Give every sample the threshold calibrated on its own stratum.

    Returns the per-sample threshold and a boolean flag marking samples whose
    stratum held no calibration data.  Those fall back to the supplied coarser
    threshold, which is what a deployed system must do when it meets a scene it
    never calibrated on.  Handing them an infinite threshold instead would let
    the table report perfect coverage while silently sweeping the whole
    codebook, so the fallback is explicit and counted.
    """
    true_scores = np.asarray(true_scores, dtype=np.float64)
    strata = np.asarray(strata)
    if not (true_scores.shape == strata.shape == calibration.shape):
        raise ValueError(
            f"Shape mismatch: scores {true_scores.shape}, strata {strata.shape}, "
            f"calibration {calibration.shape}."
        )
    thresholds = np.full(true_scores.shape, fallback, dtype=np.float64)
    unseen = np.ones(true_scores.shape, dtype=bool)
    for value in np.unique(strata):
        member = strata == value
        observed = member & calibration
        if not observed.any():
            continue
        thresholds[member] = conformal_quantile(true_scores[observed], alpha)
        unseen[member] = False
    return thresholds, unseen


def true_beam_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pick out the score of the beam that was actually best."""
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] != labels.size:
        raise ValueError(f"Shape mismatch: scores {scores.shape}, labels {labels.shape}.")
    return scores[np.arange(labels.size), labels]


def _per_sample(threshold: float | np.ndarray, count: int) -> np.ndarray:
    """Accept either one global threshold or one per sample."""
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        return np.full(count, float(values))
    if values.shape != (count,):
        raise ValueError(f"Expected a scalar threshold or one per sample ({count}), got {values.shape}.")
    return values


def coverage(scores: np.ndarray, labels: np.ndarray, threshold: float | np.ndarray) -> float:
    """Fraction of samples whose true beam survives the threshold."""
    scores = np.asarray(scores)
    return float(np.mean(true_beam_scores(scores, labels) <= _per_sample(threshold, scores.shape[0])))


def set_sizes(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Number of beams retained per sample."""
    scores = np.asarray(scores)
    return (scores <= _per_sample(threshold, scores.shape[0])[:, None]).sum(axis=1).astype(np.int64)


__all__ = [
    "ALPHA",
    "CALIBRATION_FRACTION",
    "SPLIT_SEED",
    "block_split",
    "conformal_quantile",
    "coverage",
    "nonconformity",
    "random_split",
    "set_sizes",
    "stratum_thresholds",
    "track_ids",
    "track_key",
    "true_beam_scores",
]
