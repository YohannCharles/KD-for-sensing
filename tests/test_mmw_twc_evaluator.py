import csv
import importlib.util
import json
from pathlib import Path

import pytest

from kd_sensing.data.mmw.twc_evidence import build_fixed_mask_cache


ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / "eval_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location("mmw_twc_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strict_evaluator_accepts_only_immutable_full_digest_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    cache = build_fixed_mask_cache(seed=23)
    cache_path = tmp_path / "fixed_mask_cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    protocol = {
        "fixed_mask_cache": {
            "path": str(cache_path),
            "sha256": evaluator._sha256_file(cache_path),
            "cache_checksum": cache["checksum"],
            "condition_count": len(cache["conditions"]),
        }
    }

    loaded = evaluator._load_immutable_cache(protocol)

    assert len(loaded["conditions"]) == 131
    assert all(len(item["mask_digest"]) == 64 for item in loaded["conditions"])
    cache["conditions"][0]["modality_temporal_mask"][0][0] = False
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        evaluator._load_immutable_cache(protocol)


def test_strict_evaluator_rejects_non_outer_domain_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    protocol, confirmation, config = _strict_config_fixture(tmp_path, evaluator)
    confirmation_path = tmp_path / "confirmation_train_splits_manifest.json"
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")

    resolved = evaluator._validate_confirmation_config(config, protocol, root=tmp_path, method="T2", seed=1)

    assert resolved["manifest_sha256"] == confirmation["manifest_sha256"]
    config["data"]["dataset"]["domains"][0]["test_csv_name"] = str(tmp_path / "wrong.csv")
    config["mmw_twc_evidence"]["config_recipe_sha256"] = evaluator.canonical_mmw_twc_evidence_config_sha256(config)
    with pytest.raises(ValueError, match="outer_evidence CSV"):
        evaluator._validate_confirmation_config(config, protocol, root=tmp_path, method="T2", seed=1)


def test_strict_evaluator_rows_include_summary_identity_and_within_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    split = tmp_path / "outer.csv"
    _write_csv(split)
    condition = build_fixed_mask_cache(seed=23)["conditions"][0]
    metrics = {
        "evaluated_sample_count": 1,
        "evaluated_batch_count": 1,
        "coverage_complete": True,
        "top1": 0.5,
        "top3": 0.7,
        "top5": 0.8,
        "within_1": 0.9,
        "within_3": 1.0,
            "adba": 0.6,
            "mae": 1.2,
            "normalized_gain": 0.8,
            "gain_loss_db": 1.0,
            "spectral_efficiency_ratio_0db": 0.9,
            "spectral_efficiency_loss_0db": 0.1,
            "spectral_efficiency_ratio_10db": 0.85,
            "spectral_efficiency_loss_10db": 0.2,
            "spectral_efficiency_ratio_20db": 0.8,
            "spectral_efficiency_loss_20db": 0.3,
    }
    provenance = {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "protocol_manifest_sha256": "a" * 64,
        "confirmation_split_manifest_sha256": "b" * 64,
        "split_role": "outer_evidence",
        "evaluation_mask_cache_sha256": "c" * 64,
        "evaluation_mask_cache_checksum": "d" * 64,
        "topology_id": "ula_dft_phase_cycle_v1",
        "topology_descriptor_sha256": "e" * 64,
        "topology_mapping_sha256": "f" * 64,
        "reproduction_scope": "project_mainline",
        "paper_equivalent": False,
        "temporal_result_scope": "post_selection_confirmation_mainline",
        "architecture_scope": "project_mainline_u_mask_beam",
        "baseline_adaptation_scope": "project_mainline_u_mask_beam",
        "omitted_paper_inputs_json": "[]",
        "omitted_paper_training_stages_json": "[]",
        "config_sha256": "g" * 64,
    }

    rows = evaluator._domain_rows(
        [metrics],
        conditions=[condition],
        domain={"id": "sunny/scene", "condition": "sunny", "scene": "scene", "split_path": str(split), "sample_count": 1},
        provenance=provenance,
        partial=False,
    )

    row = rows[0]
    assert row["within_1"] == pytest.approx(0.9)
    assert row["split_role"] == "outer_evidence"
    assert len(row["mask_digest"]) == 64
    assert json.loads(row["mask_matrix_json"])[0]


def test_strict_evaluator_requires_explicit_partial_opt_in(monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)

    with pytest.raises(ValueError, match="allow_partial"):
        evaluator._validate_request(method="T2", seed=1, max_batches=1, max_domains=None, allow_partial=False)


def test_strict_evaluator_refuses_smoke_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    protocol, confirmation, config = _strict_config_fixture(tmp_path, evaluator)
    (tmp_path / "confirmation_train_splits_manifest.json").write_text(json.dumps(confirmation), encoding="utf-8")
    config["mmw_twc_evidence"]["smoke_preflight"] = True
    config["mmw_twc_evidence"]["config_recipe_sha256"] = evaluator.canonical_mmw_twc_evidence_config_sha256(config)

    with pytest.raises(ValueError, match="smoke_preflight"):
        evaluator._validate_confirmation_config(config, protocol, root=tmp_path, method="T2", seed=1)


def test_strict_evaluator_binds_unique_plan_job_and_topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    config = tmp_path / "generated_configs" / "T2_seed1.yaml"
    config.parent.mkdir()
    config.write_text("experiment: {}\n", encoding="utf-8")
    run_dir = tmp_path / "T2" / "seed1"
    run_dir.mkdir(parents=True)
    control = tmp_path / "generated_control_configs" / "T2_seed1.yaml"
    control.parent.mkdir()
    control.write_text("experiment: {}\n", encoding="utf-8")
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    topology_path = tmp_path / "topology_manifest.json"
    descriptor = {"topology_id": "ula_dft_phase_cycle_v1"}
    topology_path.write_text(
        json.dumps({"descriptor": descriptor, "descriptor_sha256": evaluator._sha256_payload(descriptor)}), encoding="utf-8"
    )
    job = {
        "variant": "T2",
        "method": "T2",
        "seed": 1,
        "matched_control": "T2",
        "variant_role": "main",
        "allowed_config_diff": [],
        "config_path": str(config),
        "config_sha256": evaluator._sha256_file(config),
        "matched_control_config_path": str(control),
        "matched_control_config_sha256": evaluator._sha256_file(control),
        "run_dir": str(run_dir),
        "evaluation_output_path": str(tmp_path / "eval_outer" / "T2" / "seed1" / "metrics.csv"),
    }
    plan = {
        "schema_version": 2,
        "protocol": "mmw_twc_confirmation_training_v1",
        "request": {
            "plan_schema_version": 2,
            "comparison_contract_version": 1,
            "protocol_manifest_sha256": "a" * 64,
            "topology_descriptor_sha256": evaluator._sha256_payload(descriptor),
            "batch_size": 64,
            "epochs": 40,
            "smoke": False,
        },
        "request_sha256": "",
        "protocol_manifest": str(protocol_path),
        "topology_manifest": str(topology_path),
        "confirmation_splits_manifest": str(tmp_path / "confirmation_train_splits_manifest.json"),
        "jobs": [job],
    }
    plan["request_sha256"] = evaluator._sha256_payload(plan["request"])
    plan["plan_sha256"] = evaluator._plan_sha256(plan)
    (tmp_path / "training_manifest_test.json").write_text(json.dumps(plan), encoding="utf-8")

    binding = evaluator._validate_training_plan_binding(
        root=tmp_path,
        method="T2",
        seed=1,
        artifacts={"config": config.resolve(), "run_dir": run_dir.resolve(), "checkpoint": run_dir / "checkpoints" / "last.pth"},
        protocol_path=protocol_path,
        protocol={"protocol_id": "mmw_twc_outer_v1", "manifest_sha256": "a" * 64},
    )

    assert binding["plan_sha256"] == plan["plan_sha256"]
    assert binding["evaluation_topology_descriptor_sha256"] == evaluator._sha256_payload(descriptor)


def test_strict_evaluator_accepts_current_schema_v3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evaluator = _load_evaluator(monkeypatch)
    config = tmp_path / "generated_configs" / "T2_seed1.yaml"
    config.parent.mkdir()
    config.write_text("experiment: {}\n", encoding="utf-8")
    run_dir = tmp_path / "T2" / "seed1"
    run_dir.mkdir(parents=True)
    control = tmp_path / "generated_control_configs" / "T2_seed1.yaml"
    control.parent.mkdir()
    control.write_text("experiment: {}\n", encoding="utf-8")
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    topology_path = tmp_path / "topology_manifest.json"
    descriptor = {"topology_id": "ula_dft_phase_cycle_v1"}
    topology_path.write_text(
        json.dumps({"descriptor": descriptor, "descriptor_sha256": evaluator._sha256_payload(descriptor)}), encoding="utf-8"
    )
    job = {
        "variant": "T2",
        "method": "T2",
        "display_name": "T2",
        "seed": 1,
        "matched_control": "T2",
        "variant_role": "main",
        "allowed_config_diff": [],
        "config_path": str(config),
        "config_sha256": evaluator._sha256_file(config),
        "matched_control_config_path": str(control),
        "matched_control_config_sha256": evaluator._sha256_file(control),
        "run_dir": str(run_dir),
        "evaluation_output_path": str(tmp_path / "eval_outer" / "T2" / "seed1" / "metrics.csv"),
    }
    plan = {
        "schema_version": 3,
        "protocol": "mmw_twc_fair_pattern_training_v1",
        "request": {
            "plan_schema_version": 3,
            "comparison_contract_version": 2,
            "protocol_manifest_sha256": "a" * 64,
            "topology_descriptor_sha256": evaluator._sha256_payload(descriptor),
            "batch_size": 64,
            "epochs": 40,
            "smoke": False,
        },
        "request_sha256": "",
        "protocol_manifest": str(protocol_path),
        "topology_manifest": str(topology_path),
        "confirmation_splits_manifest": str(tmp_path / "confirmation_train_splits_manifest.json"),
        "jobs": [job],
    }
    plan["request_sha256"] = evaluator._sha256_payload(plan["request"])
    plan["plan_sha256"] = evaluator._plan_sha256(plan)
    (tmp_path / "training_manifest_v3_test.json").write_text(json.dumps(plan), encoding="utf-8")

    binding = evaluator._validate_training_plan_binding(
        root=tmp_path,
        method="T2",
        seed=1,
        artifacts={"config": config.resolve(), "run_dir": run_dir.resolve(), "checkpoint": run_dir / "checkpoints" / "last.pth"},
        protocol_path=protocol_path,
        protocol={"protocol_id": "mmw_twc_outer_v1", "manifest_sha256": "a" * 64},
    )

    assert binding["plan_sha256"] == plan["plan_sha256"]


def _strict_config_fixture(tmp_path: Path, evaluator):
    protocol_domains = []
    confirmation_domains = []
    config_domains = []
    for index in range(15):
        domain_id = f"weather{index // 5}/scene{index}"
        train = tmp_path / f"train_{index}.csv"
        validation = tmp_path / f"validation_{index}.csv"
        outer = tmp_path / f"outer_{index}.csv"
        _write_csv(train)
        _write_csv(validation)
        _write_csv(outer)
        train_item = {"csv": str(train), "sha256": evaluator._sha256_file(train), "row_count": 1}
        validation_item = {"csv": str(validation), "sha256": evaluator._sha256_file(validation), "row_count": 1}
        outer_item = {"csv": str(outer), "sha256": evaluator._sha256_file(outer), "row_count": 1}
        condition, scene = domain_id.split("/")
        protocol_domains.append(
            {
                "id": domain_id,
                "condition": condition,
                "scene": scene,
                "split": {"outer_evidence": outer_item},
            }
        )
        confirmation_domains.append(
            {
                "id": domain_id,
                "condition": condition,
                "scene": scene,
                "inner_validation": validation_item,
                "confirmation_train": train_item,
                "outer_evidence": outer_item,
            }
        )
        config_domains.append(
            {
                "id": domain_id,
                "condition": condition,
                "scene": scene,
                "train_csv_name": str(train),
                "val_csv_name": str(validation),
                "test_csv_name": str(outer),
            }
        )
    confirmation = {"schema_version": 1, "domains": confirmation_domains}
    confirmation["manifest_sha256"] = evaluator._sha256_payload(confirmation)
    protocol = {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "manifest_sha256": "m" * 64,
        "fixed_mask_cache": {"sha256": "c" * 64, "cache_checksum": "d" * 64},
        "domains": protocol_domains,
    }
    config = {
        "experiment": {"name": "T2", "seed": 1},
        "training": {"final_test": {"enabled": False}},
        "mmw_all_weather_protocol": {"split_tag": "mmw_twc_outer_v1"},
        "mmw_twc_evidence": {
            "protocol_id": "mmw_twc_outer_v1",
            "protocol_manifest_sha256": "m" * 64,
            "evaluation_mask_cache_sha256": "c" * 64,
            "evaluation_mask_cache_checksum": "d" * 64,
            "confirmation_split_manifest_sha256": confirmation["manifest_sha256"],
            "training_role": "confirmation_train",
            "training_mask_seed": 1,
            "training_mask_seed_algorithm": "test",
            "domain_sampling_seed": 1,
            "smoke_preflight": False,
            "topology_id": "ula_dft_phase_cycle_v1",
            "topology_descriptor_sha256": "e" * 64,
            "topology_mapping_sha256": "f" * 64,
            "evaluation_topology_id": "ula_dft_phase_cycle_v1",
            "evaluation_topology_descriptor_sha256": "e" * 64,
        },
        "data": {"dataset": {"domains": config_domains}},
    }
    config["mmw_twc_evidence"]["config_recipe_sha256"] = evaluator.canonical_mmw_twc_evidence_config_sha256(config)
    return protocol, confirmation, config


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        writer.writerow({"id": "sample"})
