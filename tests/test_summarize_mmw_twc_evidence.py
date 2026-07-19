import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kd_sensing.data.mmw.twc_evidence import build_fixed_mask_cache


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "S1", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
SEEDS = (1, 2, 3, 4, 5)


def _load_summary():
    path = ROOT / "scripts" / "summarize_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _prepare_protocol(summary, tmp_path: Path) -> tuple[Path, dict]:
    cache = build_fixed_mask_cache(seed=20260717)
    cache_path = tmp_path / "fixed_mask_cache.json"
    _write_json(cache_path, cache)
    domains = []
    weathers = ("sunny", "rainy", "foggy")
    for index in range(15):
        outer = tmp_path / "splits" / f"domain_{index}" / "outer.csv"
        outer.parent.mkdir(parents=True, exist_ok=True)
        outer.write_text("sample\n1\n", encoding="utf-8")
        domains.append(
            {
                "id": f"{weathers[index % 3]}/scene_{index % 5}",
                "condition": weathers[index % 3],
                "scene": f"scene_{index % 5}",
                "split": {
                    "outer_evidence": {
                        "csv": str(outer),
                        "sha256": _sha256_file(outer),
                        "row_count": 10 + index,
                    }
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "domains": domains,
        "fixed_mask_cache": {
            "path": str(cache_path),
            "sha256": _sha256_file(cache_path),
            "cache_checksum": cache["checksum"],
            "condition_count": len(cache["conditions"]),
        },
    }
    manifest["manifest_sha256"] = summary._sha256_payload(manifest)
    manifest_path = tmp_path / "protocol_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _row(method: str, seed: int, domain: dict, domain_index: int, mask: dict, mask_index: int, manifest: dict) -> dict[str, object]:
    method_bias = {
        "T2": 0.04,
        "S1": 0.02,
        "masktrain_cls": 0.0,
        "amber_full": -0.01,
        "rmbp_mm": -0.04,
        "amr_net_4m": -0.005,
    }[method]
    score = 0.45 + method_bias + seed * 0.001 + domain_index * 0.0001 - mask_index * 0.00001
    baseline = method in {"masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m"}
    missing_components = {
        "masktrain_cls": "none",
        "amber_full": "historical_beam_index",
        "rmbp_mm": "partial_beam_measurement; unimodal_pretraining",
        "amr_net_4m": "paper_native_three_modality_window2",
    }.get(method, "none")
    return {
        "method": method,
        "seed": seed,
        "domain_id": domain["id"],
        "condition": domain["condition"],
        "scene": domain["scene"],
        "sample_csv_sha256": domain["split"]["outer_evidence"]["sha256"],
        "sample_count": 10 + domain_index,
        "expected_sample_count": 10 + domain_index,
        "coverage_status": "complete",
        "partial_request": "False",
        "checkpoint": f"/runs/{method}/seed{seed}/checkpoints/last.pth",
        "checkpoint_sha256": f"checkpoint-{method}-{seed}",
        "checkpoint_role": "last",
        "checkpoint_policy": "fixed_epoch_last_pth",
        "metric_profile": "64_beam_ula_dft_phase_cycle_topk_progressive_top3_dba_v1",
        "eval_family": mask["family"],
        "pattern": mask["pattern"],
        "mask_type": mask["mask_type"],
        "requested_missing_rate": mask["requested_missing_rate"],
        "missing_rate": mask["requested_missing_rate"],
        "available_modalities": ",".join(mask["available_modalities"]),
        "drop_count": mask["drop_count"],
        "mask_digest": mask["mask_digest"],
        "observed_missing_rate": mask["observed_missing_rate"],
        "protocol_id": manifest["protocol_id"],
        "protocol_kind": manifest["protocol_kind"],
        "protocol_manifest_sha256": manifest["manifest_sha256"],
        "confirmation_split_manifest_sha256": "confirmation-split-sha",
        "split_role": "outer_evidence",
        "training_role": "confirmation_train",
        "training_mask_seed": seed,
        "training_mask_seed_algorithm": "sha256(base_seed,balanced_pattern_schedule,epoch); sample=(step*train_batch_size+row)%600",
        "smoke_preflight": False,
        "training_batch_size": 64,
        "training_epochs": 40,
        "evaluation_mask_cache_sha256": manifest["fixed_mask_cache"]["sha256"],
        "evaluation_mask_cache_checksum": manifest["fixed_mask_cache"]["cache_checksum"],
        "topology_id": "ula_dft_phase_cycle_v1" if method in {"T2", "S1"} else "not_applicable",
        "topology_descriptor_sha256": "topology-local" if method in {"T2", "S1"} else "not_applicable",
        "topology_mapping_sha256": "topology-map" if method in {"T2", "S1"} else "not_applicable",
        "evaluation_topology_id": "ula_dft_phase_cycle_v1",
        "evaluation_topology_descriptor_sha256": "evaluation-topology",
        "config_recipe_sha256": f"recipe-{method}-{seed}",
        "reproduction_scope": "local_adaptation" if baseline else "project_mainline",
        "paper_equivalent": "False",
        "temporal_result_scope": "local_adaptation_diagnostic" if baseline else "confirmation_mainline",
        "baseline_adaptation_scope": "reduced_local_baseline" if baseline else "project_mainline",
        "omitted_paper_inputs_json": json.dumps([missing_components]) if baseline else "[]",
        "omitted_paper_training_stages_json": "[\"unimodal_pretraining\"]" if method == "rmbp_mm" else "[]",
        "top1": score,
        "top3": min(0.99, score + 0.14),
        "top5": min(0.99, score + 0.23),
        "within_1": min(0.99, score + 0.08),
        "within_3": min(0.99, score + 0.20),
        "adba": 0.3 - method_bias,
        "mae": 2.5 - method_bias,
        "normalized_gain": min(1.0, score + 0.3),
        "gain_loss_db": 3.0 - method_bias,
        "spectral_efficiency_ratio_0db": min(1.0, score + 0.25),
        "spectral_efficiency_loss_0db": 0.3 - method_bias,
        "spectral_efficiency_ratio_10db": min(1.0, score + 0.2),
        "spectral_efficiency_loss_10db": 0.7 - method_bias,
        "spectral_efficiency_ratio_20db": min(1.0, score + 0.15),
        "spectral_efficiency_loss_20db": 1.2 - method_bias,
    }


def _prepare_complete_evaluation(summary, tmp_path: Path) -> tuple[Path, Path]:
    manifest_path, manifest = _prepare_protocol(summary, tmp_path)
    cache = json.loads((tmp_path / "fixed_mask_cache.json").read_text(encoding="utf-8"))
    eval_dir = tmp_path / "eval"
    for method in METHODS:
        for seed in SEEDS:
            rows = [
                _row(method, seed, domain, domain_index, mask, mask_index, manifest)
                for domain_index, domain in enumerate(manifest["domains"])
                for mask_index, mask in enumerate(cache["conditions"])
            ]
            target = eval_dir / method / f"seed{seed}" / "metrics.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return eval_dir, manifest_path


def test_twc_summary_requires_full_fixed_matrix_and_exports_tables(tmp_path: Path) -> None:
    summary = _load_summary()
    eval_dir, manifest_path = _prepare_complete_evaluation(summary, tmp_path)
    output_dir = tmp_path / "summary"

    result = summary.summarize(
        eval_dir,
        manifest_path,
        output_dir,
        bootstrap_iterations=30,
        bootstrap_seed=7,
    )

    assert len(result["coverage"]) == 30
    assert len(result["paper_table"]) == 6 * 6
    assert len(result["paired_ci"]) == 5 * 6 * 15
    assert len({row["condition_identity_sha256"] for row in result["coverage"]}) == 1
    assert (output_dir / "paper_main_table.csv").is_file()
    assert (output_dir / "paired_bootstrap_ci.csv").is_file()
    assert (output_dir / "plot_manifest.json").is_file()
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "post-selection confirmation" in markdown
    assert "## Main ADBA" in markdown
    assert "## Secondary Top-1" in markdown
    assert "amber_full" in markdown
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["request"]["primary_metric"] == "adba"
    assert provenance["request"]["secondary_metric"] == "top1"

    target = eval_dir / "T2" / "seed1" / "metrics.csv"
    rows = list(csv.DictReader(target.open(newline="", encoding="utf-8")))
    original_digest = rows[0]["mask_digest"]
    rows[0]["mask_digest"] = "0" * 64
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Condition identity"):
        summary.summarize(eval_dir, manifest_path, tmp_path / "other_summary", bootstrap_iterations=5)

    rows[0]["mask_digest"] = original_digest
    rows[0]["coverage_status"] = "partial"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Partial strict evidence"):
        summary.summarize(eval_dir, manifest_path, tmp_path / "partial_summary", bootstrap_iterations=5)


def test_twc_summary_refuses_missing_seed_csv(tmp_path: Path) -> None:
    summary = _load_summary()
    manifest_path, _ = _prepare_protocol(summary, tmp_path)
    eval_dir = tmp_path / "eval"
    with pytest.raises(FileNotFoundError, match="Missing strict evidence metrics CSV"):
        summary.summarize(eval_dir, manifest_path, tmp_path / "summary", bootstrap_iterations=5)


def test_twc_summary_refuses_smoke_preflight_rows(tmp_path: Path) -> None:
    summary = _load_summary()
    eval_dir, manifest_path = _prepare_complete_evaluation(summary, tmp_path)
    target = eval_dir / "T2" / "seed1" / "metrics.csv"
    rows = list(csv.DictReader(target.open(newline="", encoding="utf-8")))
    rows[0]["smoke_preflight"] = "True"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Smoke-preflight"):
        summary.summarize(eval_dir, manifest_path, tmp_path / "summary", bootstrap_iterations=5)
