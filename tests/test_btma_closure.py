"""Protocol tests for the read-only BTMA negative-result closure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

closure = pytest.importorskip("analyze_btma_closure")

from kd_sensing.baselines.btma_assignment import BTMA_METHODS  # noqa: E402


def _sample_id(domain: str, agent: str, frame: int) -> str:
    return f"mmw:{domain}:train:{domain}:Town03:{domain}:{agent}:{frame:06d}"


def test_published_protocol_exposes_every_field_the_closure_reads() -> None:
    """Fail here in milliseconds rather than after the model is on the GPU."""
    manifest = ROOT / "outputs/full_pool_capacity/protocol/split_manifest.json"
    if not manifest.is_file():
        pytest.skip("Full-pool protocol manifest is a local artifact.")
    import json

    protocol = json.loads(manifest.read_text(encoding="utf-8"))
    for field in ("protocol_fingerprint", "validation_sample_count", "train_sample_count", "outer_test_accessed"):
        assert field in protocol, f"closure reads protocol[{field!r}]"
    assert protocol["outer_test_accessed"] is False
    assert int(protocol["validation_sample_count"]) == 9180
    assert int(protocol["train_sample_count"]) == 37038


def test_blocks_follow_contiguous_frames_inside_each_domain_and_agent() -> None:
    ids, domains = [], []
    for domain in ("foggy/A", "sunny/B"):
        for agent in ("cav_1", "cav_2"):
            for frame in range(70):
                ids.append(_sample_id(domain.split("/")[1], agent, 1000 + frame))
                domains.append(domain)
    block, position = closure._temporal_blocks(np.asarray(ids, dtype=str), np.asarray(domains, dtype=str))

    # 4 sequences x ceil(70/32) = 4 x 3 blocks
    assert len(np.unique(block)) == 12
    assert sorted(np.bincount(block).tolist()) == sorted([32, 32, 6] * 4)
    # A block never spans two sequences.
    for value in np.unique(block):
        members = np.flatnonzero(block == value)
        assert len({domains[index] for index in members}) == 1
        assert len({ids[index].split(":")[-2] for index in members}) == 1
        assert np.array_equal(np.sort(position[members]), np.arange(position[members].min(), position[members].min() + len(members)))


def test_block_assignment_is_independent_of_row_order() -> None:
    ids = [_sample_id("A", "cav_1", 1000 + frame) for frame in range(50)]
    domains = ["foggy/A"] * 50
    forward = closure._temporal_blocks(np.asarray(ids, dtype=str), np.asarray(domains, dtype=str))[0]
    order = np.random.default_rng(0).permutation(50)
    shuffled = closure._temporal_blocks(np.asarray(ids, dtype=str)[order], np.asarray(domains, dtype=str)[order])[0]
    assert np.array_equal(forward[order], shuffled)


def test_frame_recovery_fails_closed_on_an_unparseable_identity() -> None:
    with pytest.raises(ValueError, match="Cannot recover"):
        closure._temporal_blocks(np.asarray(["mmw:broken:identity"], dtype=str), np.asarray(["foggy/A"], dtype=str))


def _write_predictions(root: Path, offsets: dict[str, float], count: int = 256) -> None:
    ids = [_sample_id("A", "cav_1", 1000 + frame) for frame in range(count)]
    block, position = closure._temporal_blocks(np.asarray(ids, dtype=str), np.asarray(["foggy/A"] * count, dtype=str))
    generator = np.random.default_rng(5)
    base = generator.random(count)
    for method, offset in offsets.items():
        metrics = np.zeros((count, len(closure.METRIC_NAMES)), dtype=np.float64)
        metrics[:, closure.METRIC_NAMES.index("top1")] = (base + offset > 0.5).astype(np.float64)
        metrics[:, closure.METRIC_NAMES.index("mae")] = base * 10.0 - offset
        destination = root / method / "validation_predictions.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            sample_id=np.asarray(ids, dtype=str),
            label=np.zeros(count, dtype=np.int64),
            anchor_logits=np.zeros((count, 64), dtype=np.float32),
            metrics=metrics,
            metric_names=np.asarray(closure.METRIC_NAMES, dtype=str),
            domain=np.asarray(["foggy/A"] * count, dtype=str),
            weather=np.asarray(["foggy"] * count, dtype=str),
            block_id=block,
            frame_position=position,
        )


def test_bootstrap_is_paired_and_reports_every_method_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closure, "BOOTSTRAP_DRAWS", 64)
    _write_predictions(tmp_path, {method: 0.0 for method in BTMA_METHODS})
    rows = closure.block_bootstrap(tmp_path)

    expected_pairs = len(BTMA_METHODS) * (len(BTMA_METHODS) - 1) // 2
    assert len(rows) == expected_pairs * len(closure.REPORT_METRICS)
    # Identical predictions must give an exactly zero paired difference and a
    # degenerate interval -- that is what proves the resampling is shared.
    for row in rows:
        assert row["difference"] == pytest.approx(0.0, abs=1e-12)
        assert row["ci_low"] == pytest.approx(0.0, abs=1e-12)
        assert row["ci_high"] == pytest.approx(0.0, abs=1e-12)
        assert row["crosses_zero"] is True
        assert row["block_length"] == closure.BLOCK_LENGTH


def test_bootstrap_detects_a_real_separation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closure, "BOOTSTRAP_DRAWS", 200)
    offsets = {method: 0.0 for method in BTMA_METHODS}
    offsets[BTMA_METHODS[-1]] = 0.45
    _write_predictions(tmp_path, offsets)
    rows = closure.block_bootstrap(tmp_path)
    separated = [
        row
        for row in rows
        if row["metric"] == "top1" and BTMA_METHODS[-1] in (row["left"], row["right"]) and not row["crosses_zero"]
    ]
    assert separated, "a large induced difference must produce at least one interval that excludes zero"


def test_bootstrap_refuses_to_overwrite_an_existing_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closure, "BOOTSTRAP_DRAWS", 8)
    _write_predictions(tmp_path, {method: 0.0 for method in BTMA_METHODS})
    closure.block_bootstrap(tmp_path)
    with pytest.raises(FileExistsError):
        closure.block_bootstrap(tmp_path)


def test_bootstrap_fails_closed_when_identities_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closure, "BOOTSTRAP_DRAWS", 8)
    _write_predictions(tmp_path, {method: 0.0 for method in BTMA_METHODS})
    path = tmp_path / BTMA_METHODS[1] / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    data["sample_id"] = np.asarray([f"other:{index}" for index in range(len(data["sample_id"]))], dtype=str)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="pairing is impossible"):
        closure.block_bootstrap(tmp_path)


def test_spearman_reports_nan_for_a_constant_series() -> None:
    constant = np.ones(50)
    varying = np.arange(50, dtype=np.float64)
    rho, p_value = closure._spearman(constant, varying)
    assert np.isnan(rho) and np.isnan(p_value)
    rho, _ = closure._spearman(varying, varying)
    assert rho == pytest.approx(1.0)


def test_preregistered_constants_are_fixed_before_any_interval() -> None:
    assert closure.BLOCK_LENGTH == 32
    assert closure.BOOTSTRAP_DRAWS == 2000
    assert closure.REPORT_METRICS == ("top1", "top3", "within3", "mae", "topology_risk", "distance_gt5")
