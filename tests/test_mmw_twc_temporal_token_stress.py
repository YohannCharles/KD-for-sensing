import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kd_sensing.data.mmw.twc_temporal_token_stress import (
    MASKS_PER_RATE,
    PRIMARY_STRESS_RATES,
    SINGLE_CELL_MASK_COUNT,
    SINGLE_CELL_RATE,
    STRESS_PROTOCOL_ID,
    STRESS_PROTOCOL_KIND,
    STRESS_RATES,
    build_balanced_temporal_token_stress_cache,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "S1", "amber_full", "rmbp_mm")
SEEDS = (1, 2, 3, 4, 5)


def _load_summary():
    path = ROOT / "scripts" / "summarize_mmw_twc_temporal_token_stress.py"
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


def test_balanced_temporal_token_cache_has_exact_cardinality_and_balance():
    cache = build_balanced_temporal_token_stress_cache(seed=20260720)
    replay = build_balanced_temporal_token_stress_cache(seed=20260720)

    assert cache["checksum"] == replay["checksum"]
    assert len(cache["conditions"]) == 1 + MASKS_PER_RATE * len(PRIMARY_STRESS_RATES) + SINGLE_CELL_MASK_COUNT
    for rate, audit in zip(STRESS_RATES, cache["rate_balance_audit"]):
        retained = 20 - int(round(rate * 20))
        expected_count = SINGLE_CELL_MASK_COUNT if rate == SINGLE_CELL_RATE else MASKS_PER_RATE
        conditions = [item for item in cache["conditions"] if item["requested_missing_rate"] == rate]
        assert len(conditions) == expected_count
        assert {item["retained_token_count"] for item in conditions} == {retained}
        assert {item["dropped_token_count"] for item in conditions} == {20 - retained}
        assert set(audit["per_cell_retained_counts"]) == {expected_count * retained // 20}
        assert audit["per_modality_missing_rates"] == pytest.approx([rate] * 4)
        assert audit["per_frame_missing_rates"] == pytest.approx([rate] * 5)
        assert sum(audit["per_mask_modality_composition_histogram"].values()) == expected_count

    ordinary_80 = [item for item in cache["conditions"] if item["requested_missing_rate"] == 0.8]
    assert any(len(set(item["per_modality_retained_counts"])) > 1 for item in ordinary_80)

    extreme = [item for item in cache["conditions"] if item["requested_missing_rate"] == 0.95]
    retained_cells = [
        tuple((time, modality) for time, row in enumerate(item["modality_temporal_mask"]) for modality, value in enumerate(row) if value)
        for item in extreme
    ]
    assert len(set(retained_cells)) == 20


def test_token_stress_summary_exports_main_and_single_cell_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _load_summary()
    cache = build_balanced_temporal_token_stress_cache(seed=20260720)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    domains = []
    for index in range(15):
        split = tmp_path / "splits" / f"domain_{index}.csv"
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text("sample\n1\n", encoding="utf-8")
        domains.append(
            {
                "id": f"weather{index % 3}/scene{index % 5}",
                "condition": f"weather{index % 3}",
                "scene": f"scene{index % 5}",
                "split": {"outer_evidence": {"csv": str(split), "sha256": _sha256_file(split), "row_count": 1}},
            }
        )
    extension = {
        "protocol_id": STRESS_PROTOCOL_ID,
        "protocol_kind": STRESS_PROTOCOL_KIND,
        "path": str(tmp_path / "extension.json"),
        "manifest_sha256": "e" * 64,
        "parent_training_protocol": {
            "protocol_id": "mmw_twc_outer_v1",
            "protocol_manifest_sha256": "p" * 64,
            "fixed_mask_cache_sha256": "a" * 64,
            "fixed_mask_cache_checksum": "b" * 64,
        },
        "fixed_mask_cache": {
            "resolved_path": str(cache_path),
            "sha256": _sha256_file(cache_path),
            "cache_checksum": cache["checksum"],
        },
        "domains": domains,
    }
    monkeypatch.setattr(summary, "load_temporal_token_stress_protocol", lambda _: extension)

    eval_dir = tmp_path / "eval"
    for method_index, method in enumerate(METHODS):
        for seed in SEEDS:
            rows = []
            for domain_index, domain in enumerate(domains):
                for condition_index, mask in enumerate(cache["conditions"]):
                    score = 0.45 + method_index * -0.02 + seed * 0.001 + domain_index * 0.0001 - condition_index * 0.00001
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "domain_id": domain["id"],
                            "sample_csv_sha256": domain["split"]["outer_evidence"]["sha256"],
                            "expected_sample_count": 1,
                            "sample_count": 1,
                            "coverage_status": "complete",
                            "partial_request": "False",
                            "eval_family": mask["family"],
                            "pattern": mask["pattern"],
                            "mask_type": mask["mask_type"],
                            "mask_digest": mask["mask_digest"],
                            "requested_missing_rate": mask["requested_missing_rate"],
                            "observed_missing_rate": mask["observed_missing_rate"],
                            "mask_matrix_json": json.dumps(mask["modality_temporal_mask"], separators=(",", ":")),
                            "token_count": mask["token_count"],
                            "retained_token_count": mask["retained_token_count"],
                            "dropped_token_count": mask["dropped_token_count"],
                            "per_modality_retained_counts_json": json.dumps(mask["per_modality_retained_counts"], separators=(",", ":")),
                            "per_modality_dropped_counts_json": json.dumps(mask["per_modality_dropped_counts"], separators=(",", ":")),
                            "per_frame_retained_counts_json": json.dumps(mask["per_frame_retained_counts"], separators=(",", ":")),
                            "per_frame_dropped_counts_json": json.dumps(mask["per_frame_dropped_counts"], separators=(",", ":")),
                            "mask_set_index": mask["mask_set_index"],
                            "mask_set_size": mask["mask_set_size"],
                            "mask_balance_policy": mask["mask_balance_policy"],
                            "protocol_id": "mmw_twc_outer_v1",
                            "protocol_manifest_sha256": "p" * 64,
                            "evaluation_extension_id": STRESS_PROTOCOL_ID,
                            "evaluation_extension_kind": STRESS_PROTOCOL_KIND,
                            "evaluation_extension_manifest_sha256": "e" * 64,
                            "evaluation_extension_parent_protocol_id": "mmw_twc_outer_v1",
                            "evaluation_extension_parent_protocol_manifest_sha256": "p" * 64,
                            "evaluation_mask_cache_sha256": _sha256_file(cache_path),
                            "evaluation_mask_cache_checksum": cache["checksum"],
                            "evaluation_extension_token_count": 20,
                            "evaluation_extension_mask_type": "modality_frame",
                            "evaluation_extension_rates_json": json.dumps(cache["rates"], separators=(",", ":")),
                            "evaluation_extension_masks_per_rate": MASKS_PER_RATE,
                            "evaluation_extension_single_cell_mask_count": SINGLE_CELL_MASK_COUNT,
                            "evaluation_extension_per_rate_mask_counts_json": json.dumps(
                                cache["per_rate_mask_counts"], separators=(",", ":"), sort_keys=True
                            ),
                            "evaluation_extension_rate_balance_audit_json": json.dumps(
                                cache["rate_balance_audit"], separators=(",", ":")
                            ),
                            "evaluation_extension_balance_policy": cache["balance_policy"],
                            "training_batch_size": 64,
                            "training_epochs": 40,
                            "checkpoint_role": "last",
                            "checkpoint_policy": "fixed_epoch_last_pth",
                            "metric_profile": "test_profile",
                            "training_mask_seed": seed,
                            "training_mask_seed_algorithm": "sha256(base_seed,balanced_pattern_schedule,epoch); sample=(step*train_batch_size+row)%600",
                            "checkpoint_sha256": f"checkpoint-{method}-{seed}",
                            "config_recipe_sha256": f"recipe-{method}-{seed}",
                            "top1": score,
                            "top3": score + 0.1,
                            "top5": score + 0.2,
                            "within_1": score + 0.05,
                            "within_3": score + 0.15,
                            "adba": 0.9 - score,
                            "mae": 1.0 - score,
                        }
                    )
            path = eval_dir / method / f"seed{seed}" / "metrics.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    output = tmp_path / "summary"
    result = summary.summarize(eval_dir, tmp_path / "extension.json", output, bootstrap_iterations=5, bootstrap_seed=7)

    assert len(result["paper_table"]) == 4 * 7
    assert len(result["extreme_table"]) == 4
    assert len(result["coverage"]) == 20
    assert (output / "paper_temporal_token_stress_table.csv").is_file()
    assert (output / "paper_single_cell_95_table.csv").is_file()
    assert (output / "mask_balance_audit.csv").is_file()
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "SingleCell95" in markdown
