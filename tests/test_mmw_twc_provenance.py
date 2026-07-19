import importlib.util
from pathlib import Path

import pytest
import yaml

from kd_sensing.config import load_config
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.utils.artifact_registry import (
    canonical_mmw_twc_evidence_config_sha256,
    canonical_payload_sha256,
    training_profile_checkpoint_provenance,
    validate_evaluation_training_profile_provenance,
)


TRAINING_MASK_SEED_ALGORITHM = (
    "sha256(base_seed,balanced_pattern_schedule,epoch); sample=(step*train_batch_size+row)%600"
)


def _evidence(*, seed: int = 1, topology_id: str = "permuted_index_v1", mapping_sha256: str) -> dict:
    return {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_manifest_sha256": "a" * 64,
        "confirmation_split_manifest_sha256": "b" * 64,
        "training_role": "confirmation_train",
        "smoke_preflight": False,
        "training_mask_seed": seed,
        "training_mask_seed_algorithm": TRAINING_MASK_SEED_ALGORITHM,
        "domain_sampling_seed": seed,
        "evaluation_mask_cache_sha256": "c" * 64,
        "evaluation_mask_cache_checksum": "d" * 64,
        "topology_id": topology_id,
        "topology_descriptor_sha256": "not_applicable",
        "topology_mapping_sha256": mapping_sha256,
        "evaluation_topology_id": "ula_dft_phase_cycle_v1",
        "evaluation_topology_descriptor_sha256": "e" * 64,
    }


def _permuted_config(*, temporal_seed: int = 1, algorithm: str = TRAINING_MASK_SEED_ALGORITHM) -> dict:
    permutation = list(range(64))
    permutation[1], permutation[32] = permutation[32], permutation[1]
    mapping_sha256 = canonical_payload_sha256({"id": "permuted_index_v1", "permutation": permutation})
    config = {
        "experiment": {"seed": 1},
        "training": {"final_test": {"enabled": False}, "resume": False},
        "temporal_missing": {
            "seed": temporal_seed,
            "mode": "balanced_pattern_schedule",
            "schedule_id": "mmw_fair_pattern_v1",
            "panel_size": 600,
            "condition_counts": {
                "clean": 120,
                "drop1": 60,
                "drop2": 60,
                "drop3": 60,
                "token20": 60,
                "token40": 60,
                "token60": 60,
                "token80": 60,
                "token90": 60,
            },
        },
        "data": {"domain_balanced_sampling": {"seed": 1}},
        "loss": {
            "u_mask_beam_jepa": {
                "use_beam_prototype_alignment": True,
                "prototype_target_circular": True,
                "prototype_topology": {"id": "permuted_index_v1", "permutation": permutation},
            }
        },
    }
    config["mmw_twc_evidence"] = _evidence(mapping_sha256=mapping_sha256)
    config["mmw_twc_evidence"]["training_mask_seed_algorithm"] = algorithm
    config["mmw_twc_evidence"]["config_recipe_sha256"] = canonical_mmw_twc_evidence_config_sha256(config)
    return config


def test_twc_provenance_binds_permuted_mapping_into_checkpoint_and_evaluation():
    config = _permuted_config()
    expected = training_profile_checkpoint_provenance(config)
    recorded = {"mmw_twc_evidence": expected["mmw_twc_evidence"]}

    assert expected["mmw_twc_evidence"]["topology_mapping_sha256"] == config["mmw_twc_evidence"]["topology_mapping_sha256"]
    assert expected["mmw_twc_evidence"]["evaluation_topology_id"] == "ula_dft_phase_cycle_v1"
    validate_evaluation_training_profile_provenance(config, recorded)

    # The same topology id is not enough: a different permutation changes the
    # counterfactual and must fail before a checkpoint can be published or used.
    config["loss"]["u_mask_beam_jepa"]["prototype_topology"]["permutation"][1] = 1
    config["loss"]["u_mask_beam_jepa"]["prototype_topology"]["permutation"][32] = 32
    # Keep the whole-recipe digest current so this specifically verifies the
    # topology mapping binding rather than merely detecting generic YAML drift.
    config["mmw_twc_evidence"]["config_recipe_sha256"] = canonical_mmw_twc_evidence_config_sha256(config)
    with pytest.raises(ValueError, match="topology provenance"):
        training_profile_checkpoint_provenance(config)


def test_twc_provenance_binds_actual_training_mask_seed_and_algorithm():
    with pytest.raises(ValueError, match="temporal_missing.seed"):
        training_profile_checkpoint_provenance(_permuted_config(temporal_seed=2))

    with pytest.raises(ValueError, match="training_mask_seed_algorithm"):
        training_profile_checkpoint_provenance(_permuted_config(algorithm="opaque"))


def test_twc_provenance_binds_the_entire_training_recipe():
    config = _permuted_config()
    config["loss"]["u_mask_beam_jepa"]["beam_label_sigma"] = 0.5

    with pytest.raises(ValueError, match="config_recipe_sha256"):
        training_profile_checkpoint_provenance(config)


def test_twc_recipe_hash_allows_only_resume_as_a_runtime_control():
    config = _permuted_config()
    frozen = config["mmw_twc_evidence"]["config_recipe_sha256"]

    config["training"]["resume"] = True
    assert canonical_mmw_twc_evidence_config_sha256(config) == frozen

    config["training"]["different_recipe_field"] = "not-a-runtime-control"
    assert canonical_mmw_twc_evidence_config_sha256(config) != frozen


def test_twc_recipe_hash_survives_generated_yaml_config_loading(tmp_path: Path):
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location("twc_launcher_provenance_test", launcher_path)
    assert spec is not None and spec.loader is not None
    twc_launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twc_launcher)
    all_weather_launcher = twc_launcher._all_weather_launcher()
    domains = [
        {
            "id": f"weather{index // 5}/scene{index}",
            "condition": f"weather{index // 5}",
            "scene": f"scene{index}",
            "data_root": "dataset/MMW/sunny",
            "train_csv_name": f"/tmp/twc_train_{index}.csv",
            "val_csv_name": f"/tmp/twc_validation_{index}.csv",
            "test_csv_name": f"/tmp/twc_test_{index}.csv",
        }
        for index in range(15)
    ]
    protocol = {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "manifest_sha256": "a" * 64,
        "fixed_mask_cache": {"sha256": "b" * 64, "cache_checksum": "c" * 64},
    }
    topology = {
        "path": "/tmp/twc_topology_manifest.json",
        "descriptor_sha256": "e" * 64,
        "descriptor": {"topology_id": "ula_dft_phase_cycle_v1"},
    }
    config = twc_launcher.build_confirmation_config(
        all_weather_launcher,
        "T2",
        tmp_path / "outputs",
        seed=1,
        batch_size=64,
        epochs=40,
        domains=domains,
        protocol=protocol,
        confirmation_splits={"manifest_sha256": "d" * 64},
        topology=topology,
    )
    config_path = tmp_path / "twc.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    reloaded = load_config(config_path)

    assert canonical_mmw_twc_evidence_config_sha256(reloaded) == config["mmw_twc_evidence"]["config_recipe_sha256"]
    training_profile_checkpoint_provenance(reloaded)


@pytest.mark.parametrize(
    "variant",
    (
        "T2",
        "S1",
        "masktrain_cls",
        "amber_full",
        "rmbp_mm",
        "amr_net_4m",
        "T2-NoBPA",
        "T2-TopologyLinear",
        "T2-TopologyPermuted",
        "T2-CLS",
        "T2-NoRouterOracle",
        "T2-ReliabilityOnly",
        "T2-Uniform",
        "T2-WholeOnly",
        "T2-BPA2CMA",
    ),
)
def test_twc_generated_variant_configs_survive_strict_provenance_reload(tmp_path: Path, variant: str):
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location(f"twc_launcher_variant_{variant}", launcher_path)
    assert spec is not None and spec.loader is not None
    twc_launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twc_launcher)
    all_weather_launcher = twc_launcher._all_weather_launcher()
    domains = [
        {
            "id": f"weather{index // 5}/scene{index}",
            "condition": f"weather{index // 5}",
            "scene": f"scene{index}",
            "data_root": "dataset/MMW/sunny",
            "train_csv_name": f"/tmp/twc_train_{index}.csv",
            "val_csv_name": f"/tmp/twc_validation_{index}.csv",
            "test_csv_name": f"/tmp/twc_test_{index}.csv",
        }
        for index in range(15)
    ]
    protocol = {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "manifest_sha256": "a" * 64,
        "fixed_mask_cache": {"sha256": "b" * 64, "cache_checksum": "c" * 64},
    }
    topology = {
        "path": "/tmp/twc_topology_manifest.json",
        "descriptor_sha256": "e" * 64,
        "descriptor": {"topology_id": "ula_dft_phase_cycle_v1"},
    }
    config = twc_launcher.build_confirmation_config(
        all_weather_launcher,
        variant,
        tmp_path / "outputs",
        seed=1,
        batch_size=64,
        epochs=40,
        domains=domains,
        protocol=protocol,
        confirmation_splits={"manifest_sha256": "d" * 64},
        topology=topology,
    )
    config_path = tmp_path / f"{variant}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    reloaded = load_config(config_path)

    training_profile_checkpoint_provenance(reloaded)
    if bool(reloaded.get("loss", {}).get("u_mask_beam_jepa", {}).get("enabled", False)):
        u_mask_beam_jepa_config(reloaded)
