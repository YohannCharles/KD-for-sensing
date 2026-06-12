from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.cli import jepa_visual_analysis as jepa_cli
from kd_sensing.diagnostics import jepa_visual_analysis as jva
from kd_sensing.evaluation.metrics import calculate_dba_score


def test_synthetic_logits_sample_metrics_match_dba_helper() -> None:
    logits = np.asarray(
        [
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 7.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 3], dtype=np.int64)

    rows, summary = jva.sample_metrics_from_logits(logits, labels, model_name="synthetic", num_beams=8, dba_delta=5)
    official = calculate_dba_score(torch.tensor(logits).unsqueeze(1), torch.tensor(labels).unsqueeze(1), delta=5)

    assert rows[0]["top1"] == 0
    assert rows[0]["target_rank"] == 1
    assert rows[0]["top3_min_distance"] == pytest.approx(0.0)
    assert rows[1]["top1_error"] == pytest.approx(1.0)
    assert rows[1]["top3_min_distance"] == pytest.approx(1.0)
    assert rows[1]["dba_contribution"] == pytest.approx(0.8)
    assert summary["dba"] == pytest.approx(float(official[0]))


def test_analysis_config_parses_overrides(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    np.savez_compressed(cache, logits=np.zeros((1, 4), dtype=np.float32), labels=np.zeros((1,), dtype=np.int64))
    config = tmp_path / "analysis.yaml"
    config.write_text(
        "\n".join(
            [
                "models:",
                "  query:",
                f"    logits_cache: {cache}",
                "split:",
                "  evaluation_split: test",
                "outputs:",
                "  formats: [png]",
            ]
        ),
        encoding="utf-8",
    )

    cfg = jva.load_analysis_config(
        config,
        output_dir=tmp_path / "out",
        overrides=["sampling.seed=7", "figures.embedding=false", "outputs.formats=[png,pdf]"],
    )

    assert cfg["sampling"]["seed"] == 7
    assert cfg["figures"]["embedding"] is False
    assert cfg["outputs"]["output_dir"] == str(tmp_path / "out")
    assert cfg["outputs"]["formats"] == ["png", "pdf"]


def test_cached_analysis_writes_manifest_outputs_and_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_cache = tmp_path / "base_logits.npz"
    query_cache = tmp_path / "query_logits.npz"
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    sample_ids = np.asarray(["s0", "s1", "s2"], dtype=object)
    base_logits = np.asarray(
        [
            [0.0, 0.0, 0.0, 9.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )
    query_logits = np.asarray(
        [
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
        ],
        dtype=np.float32,
    )
    metadata = json.dumps([{"scene": 32, "global_index": idx} for idx in range(3)])
    np.savez_compressed(base_cache, logits=base_logits, labels=labels, sample_ids=sample_ids, metadata_json=metadata)
    np.savez_compressed(
        query_cache,
        logits=query_logits,
        labels=labels,
        sample_ids=sample_ids,
        metadata_json=metadata,
        embedding=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    config = tmp_path / "analysis.yaml"
    config.write_text(
        "\n".join(
            [
                "models:",
                "  fair_base:",
                f"    logits_cache: {base_cache}",
                "  gps_query_pool:",
                f"    logits_cache: {query_cache}",
                "split:",
                "  evaluation_split: test",
                "sampling:",
                "  seed: 3",
                "  query_model: gps_query_pool",
                "  baseline_model: fair_base",
                "  cases_per_group: 1",
                "figures:",
                "  embedding: true",
                "  error_anatomy: true",
                "  attention: true",
                "  case_studies: true",
                "  robustness: true",
                "embeddings:",
                "  method: umap",
                "  layers: [output_features]",
                "outputs:",
                "  formats: [png]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(jva, "_umap_reducer", lambda **_: (_ for _ in ()).throw(ModuleNotFoundError("umap missing")))

    result = jva.run_jepa_visual_analysis(
        analysis_config=config,
        output_dir=tmp_path / "out",
        force=True,
        command=["test"],
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

    assert manifest["version"] == jva.ANALYSIS_VERSION
    assert manifest["models"]["gps_query_pool"]["metrics"]["sample_count"] == 3
    assert any(item["path"] == "tables/sample_predictions_gps_query_pool.csv" for item in manifest["outputs"])
    assert any(item["path"] == "tables/comparison_samples.csv" for item in manifest["outputs"])
    assert (tmp_path / "out" / "cache" / "embeddings_gps_query_pool.npz").exists()
    assert any("attention_unavailable:gps_query_pool" in warning for warning in manifest["warnings"])
    assert any("embedding_reducer_fallback:umap_unavailable" in warning for warning in manifest["warnings"])


def test_metadata_sanitizer_handles_none_and_nested_values() -> None:
    batch = [
        {"x": torch.tensor([1]), "metadata": {"sample_id": None, "nested": {"a": [None, "b"]}}},
        {"x": torch.tensor([2]), "metadata": {"sample_id": "two", "nested": {"a": ["1", "c"]}}},
    ]

    collated = jva.safe_metadata_collate(batch)

    assert collated["metadata"]["sample_id"] == ["", "two"]
    assert list(collated["metadata"]["nested"]["a"][0]) == ["", "1"]


def test_metadata_rows_for_batch_pads_missing_rows() -> None:
    rows = jva._metadata_rows_for_batch({"scene": "32"}, batch_size=3)

    assert rows == [{"scene": "32"}, {}, {}]


def test_jepa_visual_analysis_cli_returns_success_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_jepa_visual_analysis(**kwargs):
        return {"manifest": "manifest.json", "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(jepa_cli, "run_jepa_visual_analysis", fake_run_jepa_visual_analysis)

    exit_code = jepa_cli.main(["--analysis-config", "config.yaml", "--dry-run"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"manifest": "manifest.json", "dry_run": True}
