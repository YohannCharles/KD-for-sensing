import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def analysis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / "analyze_mmw_fused_feature_geometry.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_fused_feature_uses_final_prediction_slot(analysis):
    direct = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    temporal = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    assert torch.equal(analysis.select_fused_feature(direct), direct)
    assert torch.equal(analysis.select_fused_feature(temporal), temporal[:, -1, :])
    with pytest.raises(ValueError, match="must be"):
        analysis.select_fused_feature(torch.zeros(2, 3, 4, 5))


def test_extract_parser_defaults_to_seed1_and_accepts_explicit_seed(analysis):
    parser = analysis.build_parser()

    assert parser.parse_args(["extract", "--method", "T2"]).seed == 1
    assert parser.parse_args(["extract", "--method", "T2", "--seed", "3"]).seed == 3


def test_extract_method_resolves_requested_seed_and_uses_it_for_loader(
    analysis,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "runs"
    config = root / "generated_configs" / "T2_seed3.yaml"
    checkpoint = root / "T2" / "seed3" / "checkpoints" / "last.pth"
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text("experiment:\n  seed: 3\ndata:\n  dataloader: {}\nmodel:\n  primary: {}\n")
    checkpoint.touch()
    domains = [
        {"id": f"domain-{index}", "condition": "sunny", "scene": f"scene-{index}"}
        for index in range(15)
    ]
    validation = SimpleNamespace(dataset=SimpleNamespace(datasets=[object()] * 15, domain_inventory=domains))
    seen = {}

    monkeypatch.setattr(analysis, "_load_or_create_temporal_cache", lambda *args, **kwargs: {})
    monkeypatch.setattr(analysis, "feature_mask_specs", lambda cache: [])
    monkeypatch.setattr(analysis, "build_dataloaders", lambda cfg: {"validation": validation})
    monkeypatch.setattr(analysis, "build_device", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(analysis, "build_model", lambda cfg: torch.nn.Identity())
    monkeypatch.setattr(
        analysis,
        "load_model_state",
        lambda path, *args, **kwargs: seen.setdefault("checkpoint", Path(path)),
    )
    monkeypatch.setattr(
        analysis,
        "build_dataloader",
        lambda *args, experiment_seed, **kwargs: seen.setdefault("loader_seed", experiment_seed),
    )

    def fake_extract(*args, seed, target, **kwargs):
        seen["extract_seed"] = seed
        seen["target"] = target

    monkeypatch.setattr(analysis, "extract_domain", fake_extract)
    args = analysis.build_parser().parse_args(
        ["extract", "--method", "T2", "--seed", "3", "--max-domains", "1"]
    )

    analysis.extract_method(args, root, tmp_path / "features")

    assert seen["checkpoint"] == checkpoint
    assert seen["loader_seed"] == 3
    assert seen["extract_seed"] == 3


def test_condition_metrics_tracks_neighbor_preserving_shift(analysis):
    clean = np.eye(4, dtype=np.float32)
    labels = np.arange(4, dtype=np.int16)
    missing = np.roll(clean, -1, axis=1)
    centroids, centroid_labels = analysis.clean_beam_centroids(clean, labels)

    metrics = analysis.condition_metrics(
        clean,
        missing,
        labels,
        labels,
        np.asarray([1, 2, 3, 0], dtype=np.int16),
        centroids,
        centroid_labels,
    )

    assert metrics["centroid_assignment_same"] == pytest.approx(0.0)
    assert metrics["centroid_top1"] == pytest.approx(0.0)
    assert metrics["centroid_shift_within_3"] == pytest.approx(1.0)
    assert metrics["prediction_shift_within_3"] == pytest.approx(1.0)
    assert metrics["feature_cosine_distance"] == pytest.approx(1.0)


def test_clean_pca_is_deterministic_and_summary_uses_metric_direction(analysis):
    features = analysis._l2_normalize(
        np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    )
    first = analysis.fit_pca(features)
    second = analysis.fit_pca(features)
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert first[1].shape == (2, 3)

    rows = []
    for value in (0.2, 0.4):
        row = {"method": "T2", "method_label": "T2", "rate": 0.2, "sample_count": 4}
        for metric in analysis.FEATURE_METRICS:
            row[metric] = value if metric in analysis.LOWER_IS_BETTER else 1.0 - value
        rows.append(row)
    summary = analysis.summarize_by_rate(rows)[0]
    assert summary["feature_cosine_distance_worst"] == pytest.approx(0.4)
    assert summary["prediction_shift_within_3_worst"] == pytest.approx(0.6)


def _unit_circle(dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(64, dtype=np.int16)
    circle = np.column_stack(
        [np.cos(2.0 * np.pi * labels / 64), np.sin(2.0 * np.pi * labels / 64)]
    )
    if dim > 2:
        basis, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(dim, 2)))
        circle = circle @ basis.T
    return circle.astype(np.float64), labels


def test_signed_circular_offset_wraps_at_beam_boundary(analysis):
    clean = np.asarray([63, 0, 1, 62], dtype=np.int16)
    missing = np.asarray([0, 63, 63, 1], dtype=np.int16)

    np.testing.assert_array_equal(
        analysis.signed_circular_offset(clean, missing, num_classes=64),
        np.asarray([1, -1, -2, 3]),
    )


@pytest.mark.parametrize("dim", [2, 8])
def test_two_neighbor_isomap_recovers_unit_circle_phase(analysis, dim):
    values, labels = _unit_circle(dim)

    embedding, adjacency = analysis.knn_isomap(values, k=2)

    assert embedding.shape == (64, 2)
    assert adjacency.shape == (64, 64)
    assert adjacency.dtype == np.bool_
    assert np.array_equal(adjacency, adjacency.T)
    assert not np.any(np.diag(adjacency))
    reachable = {0}
    while True:
        expanded = reachable | set(np.flatnonzero(adjacency[list(reachable)].any(axis=0)))
        if expanded == reachable:
            break
        reachable = expanded
    assert len(reachable) == 64

    metrics = analysis.cycle_embedding_metrics(embedding, labels, num_classes=64)
    assert metrics["phase_consistency"] > 0.99
    assert metrics["angular_mae_beams"] < 0.2


def test_circle_cosine_similarity_decreases_with_circular_distance(analysis):
    values, labels = _unit_circle(dim=8)

    rows = analysis.similarity_by_circular_distance(values, labels, num_classes=64)

    assert [row["circular_distance"] for row in rows] == list(range(33))
    means = np.asarray([row["cosine_mean"] for row in rows])
    assert np.all(np.diff(means) < 0)
    np.testing.assert_allclose(means, np.cos(2.0 * np.pi * np.arange(33) / 64), atol=1e-12)
    assert [row["pair_count"] for row in rows] == [64] * 32 + [32]


def test_extract_domain_saves_paired_float32_task_outputs_and_old_bundle_stays_compatible(
    analysis,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(analysis, "prepare_evaluation_batch", lambda batch, **kwargs: batch)
    monkeypatch.setattr(
        analysis,
        "_mask_in_model_order",
        lambda *args, **kwargs: (torch.ones(5, 4, dtype=torch.bool), analysis.DEFAULT_TEMPORAL_MODALITIES),
    )
    monkeypatch.setattr(analysis, "apply_modality_temporal_mask_to_batch", lambda *args, **kwargs: None)
    logits = torch.full((2, 64), -2.0, dtype=torch.float32)
    logits[0, 1] = 3.0
    logits[1, 2] = 4.0
    step = SimpleNamespace(
        logits=logits,
        labels=torch.tensor([1, 2]),
        model_output=SimpleNamespace(output_features=torch.ones(2, 4)),
    )
    monkeypatch.setattr(analysis, "run_model_step", lambda *args, **kwargs: step)
    specs = [
        {
            "rate": 0.0 if index == 0 else 0.2 * (1 + (index - 1) // 16),
            "source_mask_index": index,
            "mask_type": "clean" if index == 0 else "modality_frame",
            "mask_digest": f"mask-{index}",
            "cache_checksum": "cache",
            "mask_item": {},
        }
        for index in range(65)
    ]
    split = tmp_path / "samples.csv"
    split.write_text("sample_id\na\nb\n")
    output_dir = tmp_path / "new"
    target = output_dir / "T2" / "domains" / "domain.npz"

    analysis.extract_domain(
        SimpleNamespace(modalities=analysis.DEFAULT_TEMPORAL_MODALITIES),
        [{"sample_id": ["a", "b"], "modality_mask": torch.ones(2, 4, dtype=torch.bool)}],
        {"experiment": {"seed": 2, "task": "fusion"}, "model": {"primary": {}}},
        torch.device("cpu"),
        specs,
        {"id": "domain", "condition": "sunny", "scene": "scene", "split_path": str(split)},
        seed=2,
        checkpoint_sha256="checkpoint",
        target=target,
        max_batches=None,
    )

    with np.load(target, allow_pickle=False) as payload:
        assert payload["logits"].shape == (65, 2, 64)
        assert payload["logits"].dtype == np.float32
        np.testing.assert_array_equal(payload["sample_ids"], ["a", "b"])
        assert int(payload["seed"].item()) == 2
    bundle = analysis.load_method_bundle(output_dir, "T2", allow_partial=True, require_task_outputs=True)
    assert bundle["logits"].shape == (65, 2, 64)
    assert bundle["seed"] == 2

    old_dir = tmp_path / "old"
    old_target = old_dir / "T2" / "domains" / "domain.npz"
    with np.load(target, allow_pickle=False) as payload:
        legacy = {key: payload[key].copy() for key in payload.files if key not in analysis.TASK_OUTPUT_FIELDS}
    analysis._atomic_save_npz(old_target, **legacy)
    legacy_bundle = analysis.load_method_bundle(old_dir, "T2", allow_partial=True)
    assert "logits" not in legacy_bundle
    with pytest.raises(ValueError, match="task-output fields are required"):
        analysis.load_method_bundle(old_dir, "T2", allow_partial=True, require_task_outputs=True)


@pytest.mark.parametrize("mismatch", ["labels", "sample_ids"])
def test_extract_domain_rejects_cross_mask_sample_order_drift(
    analysis,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
):
    monkeypatch.setattr(analysis, "prepare_evaluation_batch", lambda batch, **kwargs: batch)
    monkeypatch.setattr(
        analysis,
        "_mask_in_model_order",
        lambda *args, **kwargs: (torch.ones(5, 4, dtype=torch.bool), analysis.DEFAULT_TEMPORAL_MODALITIES),
    )
    monkeypatch.setattr(analysis, "apply_modality_temporal_mask_to_batch", lambda *args, **kwargs: None)
    calls = {"step": 0, "ids": 0}

    def fake_step(*args, **kwargs):
        calls["step"] += 1
        labels = [1, 2] if mismatch != "labels" or calls["step"] == 1 else [2, 1]
        return SimpleNamespace(
            logits=torch.zeros(2, 64),
            labels=torch.tensor(labels),
            model_output=SimpleNamespace(output_features=torch.ones(2, 4)),
        )

    def fake_ids(batch):
        calls["ids"] += 1
        return ["a", "b"] if mismatch != "sample_ids" or calls["ids"] == 1 else ["b", "a"]

    monkeypatch.setattr(analysis, "run_model_step", fake_step)
    monkeypatch.setattr(analysis, "sample_ids_from_batch", fake_ids)
    split = tmp_path / "samples.csv"
    split.write_text("sample_id\na\nb\n")
    specs = [
        {
            "rate": rate,
            "source_mask_index": index,
            "mask_type": "clean" if index == 0 else "modality_frame",
            "mask_digest": f"mask-{index}",
            "cache_checksum": "cache",
            "mask_item": {},
        }
        for index, rate in enumerate((0.0, 0.2))
    ]

    with pytest.raises(ValueError, match=mismatch):
        analysis.extract_domain(
            SimpleNamespace(modalities=analysis.DEFAULT_TEMPORAL_MODALITIES),
            [{"sample_id": ["a", "b"], "modality_mask": torch.ones(2, 4, dtype=torch.bool)}],
            {"experiment": {"seed": 1, "task": "fusion"}, "model": {"primary": {}}},
            torch.device("cpu"),
            specs,
            {"id": "domain", "condition": "sunny", "scene": "scene", "split_path": str(split)},
            seed=1,
            checkpoint_sha256="checkpoint",
            target=tmp_path / "unused.npz",
            max_batches=None,
        )


@pytest.mark.parametrize(
    "field",
    ["sample_csv_sha256", "sample_ids", "labels", "rates", "mask_digests", "cache_checksums", "seed"],
)
def test_cross_method_task_output_alignment_names_first_mismatch(analysis, field):
    def domain():
        return {
            "domain_id": "sunny/scene",
            "sample_csv_sha256": "csv",
            "sample_ids": np.asarray(["a", "b"]),
            "labels": np.asarray([1, 2], dtype=np.int16),
            "rates": np.asarray([0.0, 0.2], dtype=np.float32),
            "mask_digests": np.asarray(["clean", "missing"]),
            "cache_checksums": np.asarray(["cache", "cache"]),
            "logits": np.zeros((2, 2, 64), dtype=np.float32),
            "seed": 2,
        }

    reference = domain()
    candidate = domain()
    if field in {"sample_csv_sha256", "seed"}:
        candidate[field] = "other" if field == "sample_csv_sha256" else 3
    else:
        candidate[field] = candidate[field].copy()
        candidate[field][0] = "other" if candidate[field].dtype.kind in "US" else candidate[field][0] + 1
    bundles = {"T2": {"domains": [reference]}, "amber_full": {"domains": [candidate]}}

    with pytest.raises(ValueError, match=field):
        analysis.validate_cross_method_alignment(bundles, require_task_outputs=True)


def test_leave_one_out_centroid_excludes_the_sample_it_assigns(analysis):
    clean = np.asarray([[1, 0], [0, 1], [0.5, 0.866], [0.5, 0.866]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int16)
    centroids, centroid_labels, class_positions, leave_one_out = analysis.leave_one_out_centroid_context(
        clean,
        labels,
    )

    np.testing.assert_allclose(leave_one_out[0], [0, 1])
    assert analysis.nearest_centroid(analysis._l2_normalize(clean[:1]), centroids, centroid_labels)[0] == 0
    assignments = analysis.nearest_leave_one_out_centroid(
        clean,
        centroids,
        centroid_labels,
        class_positions,
        leave_one_out,
    )
    assert assignments[0] == 1


def test_signed_feature_shift_normalizes_each_method_rate_and_weights_domains_equally(analysis, monkeypatch):
    monkeypatch.setattr(analysis, "RATES", (0.0, 0.2, 0.4))
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int16)
    clean = np.eye(2, dtype=np.float32)[labels]
    small_domain_shift = clean.copy()
    small_domain_shift[:2] = small_domain_shift[:2, ::-1]
    large_domain_shift = clean.copy()
    large_domain_shift[2:] = large_domain_shift[2:, ::-1]
    all_shift = clean[:, ::-1]

    def item(features):
        return {
            "clean": clean,
            "labels": labels,
            "features": np.stack(features),
            "rates": np.asarray([0.0, 0.2, 0.4]),
            "domains": [{"labels": labels[:2]}, {"labels": labels[2:]}],
        }

    rows, _ = analysis.build_signed_feature_shift(
        {
            "T2": item([clean, small_domain_shift, large_domain_shift]),
            "amber_full": item([clean, clean, all_shift]),
        }
    )

    groups = {(row["method"], row["rate"]) for row in rows}
    assert groups == {("T2", 0.2), ("T2", 0.4), ("amber_full", 0.2), ("amber_full", 0.4)}
    for method, rate in groups:
        selected = [row for row in rows if row["method"] == method and row["rate"] == rate]
        assert sum(row["fraction"] for row in selected) == pytest.approx(1.0)
        assert sum(row["domain_macro_fraction"] for row in selected) == pytest.approx(1.0)

    t2_zero = {
        row["rate"]: row
        for row in rows
        if row["method"] == "T2" and row["signed_offset"] == 0
    }
    assert t2_zero[0.2]["fraction"] == pytest.approx(4 / 6)
    assert t2_zero[0.4]["fraction"] == pytest.approx(2 / 6)
    assert t2_zero[0.2]["domain_macro_fraction"] == pytest.approx(0.5)
    assert t2_zero[0.4]["domain_macro_fraction"] == pytest.approx(0.5)
