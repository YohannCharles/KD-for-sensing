import csv
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


def test_gps_query_evidence_package_uses_synthetic_metrics_attention_and_cases(tmp_path: Path) -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    sample_ids = np.asarray(["gain", "regression", "near", "failure"], dtype=object)
    metadata = json.dumps(
        [
            {"scene": 31, "scene_group": "Scene31", "condition": "P0_clean_current", "global_index": 0},
            {"scene": 32, "scene_group": "S32-S34", "condition": "P1_current_frame_missing_history_available", "global_index": 1},
            {"scene": 33, "scene_group": "S32-S34", "condition": "P2_semantic_occlusion_history_available", "global_index": 2},
            {"scene": 34, "scene_group": "S32-S34", "condition": "P3_plausible_wrong_gps_current_image", "global_index": 3},
        ]
    )
    mean_cache = tmp_path / "mean.npz"
    query_cache = tmp_path / "query.npz"
    anchor_cache = tmp_path / "anchor.npz"
    mean_logits = np.asarray(
        [
            [0, 0, 0, 9, 8, 7, 0, 0],
            [0, 9, 8, 7, 0, 0, 0, 0],
            [0, 0, 7, 9, 8, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 9, 8],
        ],
        dtype=np.float32,
    )
    query_logits = np.asarray(
        [
            [9, 8, 7, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 8, 7, 0],
            [0, 0, 8, 9, 7, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 9, 8],
        ],
        dtype=np.float32,
    )
    attention = np.asarray(
        [
            [[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]],
            [[[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]],
            [[[0.25, 0.25, 0.25, 0.25], [0.4, 0.2, 0.2, 0.2]]],
            [[[0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(mean_cache, logits=mean_logits, labels=labels, sample_ids=sample_ids, metadata_json=metadata)
    np.savez_compressed(
        query_cache,
        logits=query_logits,
        labels=labels,
        sample_ids=sample_ids,
        metadata_json=metadata,
        attention=attention,
        token_grid_shape=np.asarray([2, 2], dtype=np.int64),
    )
    np.savez_compressed(anchor_cache, logits=query_logits, labels=labels, sample_ids=sample_ids, metadata_json=metadata)

    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "model,condition,scene_group,metric,value,sample_count,split,seed,checkpoint_selection,label_space,metric_profile",
                "gps_query_pool,P0_clean_current,Scene31,dba,0.70,4,test,0,best,beam8,linear_dba",
                "mean_pool,P0_clean_current,Scene31,dba,0.55,4,test,0,best,beam8,linear_dba",
                "image_gps_anchor,P0_clean_current,Scene31,dba,0.80,4,test,0,best,beam8,linear_dba",
                "gps_query_pool,P1_current_frame_missing_history_available,S32-S34,dba,0.60,4,test,0,best,beam8,linear_dba",
                "mean_pool,P1_current_frame_missing_history_available,S32-S34,dba,0.50,4,test,0,best,beam8,linear_dba",
                "image_gps_anchor,P1_current_frame_missing_history_available,S32-S34,dba,0.61,4,test,0,best,beam8,linear_dba",
                "gps_query_pool,P2_semantic_occlusion_history_available,S32-S34,dba,0.62,4,test,0,best,beam8,linear_dba",
                "mean_pool,P2_semantic_occlusion_history_available,S32-S34,dba,0.51,4,test,0,best,beam8,linear_dba",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "analysis.yaml"
    provenance = [
        "      split: test",
        "      scene_set: S31-S34",
        "      seed: 0",
        "      checkpoint_selection: best",
        "      label_space: beam8",
        "      metric_profile: linear_dba",
    ]
    config.write_text(
        "\n".join(
            [
                "models:",
                "  mean_pool:",
                f"    logits_cache: {mean_cache}",
                "    provenance:",
                *provenance,
                "  gps_query_pool:",
                f"    logits_cache: {query_cache}",
                "    provenance:",
                *provenance,
                "  image_gps_anchor:",
                f"    logits_cache: {anchor_cache}",
                "    provenance:",
                *provenance,
                "sampling:",
                "  seed: 0",
                "  query_model: gps_query_pool",
                "  baseline_model: mean_pool",
                "  cases_per_group: 1",
                "  near_distance_threshold: 0",
                "  far_distance_threshold: 1",
                "figures:",
                "  embedding: false",
                "  error_anatomy: false",
                "  attention: true",
                "  case_studies: true",
                "  robustness: false",
                "outputs:",
                "  formats: [png]",
                "evidence:",
                "  enabled: true",
                "  model_pairs:",
                "    - name: gps_query_vs_mean",
                "      query_model: gps_query_pool",
                "      baseline_model: mean_pool",
                "  anchor_baselines:",
                "    - name: strong_anchor",
                "      model: image_gps_anchor",
                "  metrics:",
                f"    p0_p5: {metrics}",
                "  claim_gate:",
                "    metric: dba",
                "    min_clean_delta: 0.01",
                "    min_mean_delta: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    result = jva.run_jepa_visual_analysis(analysis_config=config, output_dir=tmp_path / "out", force=True)
    out = Path(result["output_dir"])
    evidence_manifest = json.loads((out / "evidence_manifest.json").read_text(encoding="utf-8"))
    delta_rows = list(csv.DictReader((out / "tables" / "paired_delta_by_condition.csv").open(encoding="utf-8")))
    attention_rows = list(csv.DictReader((out / "tables" / "attention_summary.csv").open(encoding="utf-8")))
    case_rows = list(csv.DictReader((out / "tables" / "case_selection.csv").open(encoding="utf-8")))
    claim_rows = list(csv.DictReader((out / "tables" / "claim_gate_summary.csv").open(encoding="utf-8")))

    assert evidence_manifest["model_pairs"][0]["comparability_status"] == "strict"
    assert delta_rows[0]["comparability_status"] == "strict"
    assert any(row["status"] == "supported" for row in claim_rows if row["claim"] == "gps_query_paired_effectiveness")
    assert attention_rows[0]["token_grid_height"] == "2"
    assert attention_rows[0]["aggregation_method"] == "mean_time_query"
    assert {"query_gain", "query_regression", "shared_near_miss", "shared_failure"}.issubset({row["group"] for row in case_rows})
    assert any((out / "cases").glob("query_gain_*.json"))
    assert any("attention_overlay_unavailable:gps_query_pool" in warning for warning in evidence_manifest["warnings"])


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
