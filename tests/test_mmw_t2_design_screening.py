import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.utils.artifact_registry import (
    canonical_t2_design_config_sha256,
    t2_design_candidate_recipe_sha256,
    training_profile_checkpoint_provenance,
    validate_evaluation_training_profile_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_launcher(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / "launch_mmw_t2_design_screening.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_capacity_candidates_materialize_h4_and_change_only_the_declared_capacity(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)
    control = launcher.build_design_config("H4-control", tmp_path, seed=1, batch_size=32)
    d96 = launcher.build_design_config("D96", tmp_path, seed=1, batch_size=32)
    router = launcher.build_design_config("RouterH32", tmp_path, seed=1, batch_size=32)
    gps = launcher.build_design_config("GPSH128", tmp_path, seed=1, batch_size=32)

    profile = control["mmw_all_weather_protocol"]["training_profile"]
    assert profile["id"] == "umask_h4_v1"
    assert profile["sha256"]
    assert control["training"]["optimizer"] == {"type": "adamw"}
    assert control["training"]["weight_decay"] == pytest.approx(3.0e-4)
    assert control["scheduler"] == {"type": "cosine_warm_restarts", "T_0": 40, "T_mult": 1, "eta_min": 1.0e-6}
    assert control["mmw_t2_design_screening"]["development_only"] is True
    assert control["mmw_t2_design_screening"]["claim_eligible"] is False
    assert control["training"]["final_test"] == {
        "enabled": False,
        "reason": "development_inner_validation_only",
    }

    assert d96["model"]["primary"]["d_model"] == 96
    assert {entry["output_dim"] for entry in d96["model"]["primary"]["encoders"].values()} == {96}
    assert router["model"]["primary"]["router_hidden_dim"] == 32
    assert gps["model"]["primary"]["encoders"]["gps"]["hidden_size"] == 128
    assert d96["mmw_t2_design_screening"]["matched_control"] == "H4-control"
    launcher._assert_candidate_matches_control("D96", d96, control)
    illegal = {**d96, "training": {**d96["training"], "lr": 1.0e-3}}
    with pytest.raises(ValueError, match="outside its allowlist"):
        launcher._assert_candidate_matches_control("D96", illegal, control)


def test_structure_and_objective_candidates_keep_contract_boundaries(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)
    fusion = launcher.build_design_config("FusionReliabilityMean", tmp_path, seed=1, batch_size=32)
    temporal = launcher.build_design_config("TemporalAttention", tmp_path, seed=1, batch_size=32)
    jitter = launcher.build_design_config("GPSJitter005", tmp_path, seed=1, batch_size=32)
    cma = launcher.build_design_config("CMA-w010-t020", tmp_path, seed=1, batch_size=32)
    kl = launcher.build_design_config("KL-w050", tmp_path, seed=1, batch_size=32)

    assert fusion["model"]["primary"]["fusion_type"] == "reliability_mean"
    assert fusion["loss"]["u_mask_beam_jepa"]["router_oracle_weight"] == 0.0
    assert u_mask_beam_jepa_config(fusion)["fusion_type"] == "reliability_mean"
    assert temporal["model"]["primary"]["temporal_pooling"] == {"enabled": True, "type": "masked_attention"}
    assert jitter["model"]["primary"]["encoders"]["gps"]["normalized_feature_jitter_std"] == pytest.approx(0.05)

    cma_loss = cma["loss"]["u_mask_beam_jepa"]
    assert cma_loss["use_beam_prototype_alignment"] is False
    assert cma_loss["lambda_proto"] == 0.0
    assert cma_loss["lambda_modality_proto"] == 0.0
    assert cma_loss["use_amber_cma_analogue"] is True
    assert cma["mmw_t2_design_screening"]["matched_control"] == "NoBPA-control"
    assert kl["loss"]["u_mask_beam_jepa"]["superset_consistency"]["kl_weight"] == pytest.approx(0.5)


def test_waves_and_promotion_guard_are_deterministic(monkeypatch) -> None:
    launcher = _load_launcher(monkeypatch)

    assert len(launcher.WAVES["capacity"]) == 8
    assert len(launcher.WAVES["structure"]) == 8
    assert len(launcher.WAVES["bpa"]) == 8
    assert len(launcher.WAVES["objective"]) == 8
    control = {"j": 60.0, "clean": 64.0, "modality_missing_mean": 58.0, "temporal_drop80": 55.0}
    qualified = {"j": 60.5, "clean": 63.5, "modality_missing_mean": 57.5, "temporal_drop80": 54.5}
    rejected = {"j": 61.0, "clean": 63.4, "modality_missing_mean": 58.0, "temporal_drop80": 55.0}

    assert launcher.qualifies_for_promotion(control, qualified) is True
    assert launcher.qualifies_for_promotion(control, rejected) is False
    with pytest.raises(ValueError, match="must include"):
        launcher.qualifies_for_promotion(control, {"j": 61.0})


def test_profile_and_candidate_checkpoint_provenance_fail_closed(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)
    config = launcher.build_design_config("D48", tmp_path, seed=1, batch_size=32, inner_split_fingerprint="unit-test")
    provenance = training_profile_checkpoint_provenance(config)

    assert provenance["training_profile"]["id"] == "umask_h4_v1"
    assert provenance["t2_design_screening"]["candidate_id"] == "D48"
    validate_evaluation_training_profile_provenance(config, provenance)

    mismatched = {**provenance, "t2_design_screening": {**provenance["t2_design_screening"], "candidate_id": "D96"}}
    with pytest.raises(ValueError, match="design-screening provenance"):
        validate_evaluation_training_profile_provenance(config, mismatched)


def test_design_fingerprints_ignore_cli_runtime_metadata(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)
    config = launcher.build_design_config("D48", tmp_path, seed=1, batch_size=32, inner_split_fingerprint="unit-test")
    screen = config["mmw_t2_design_screening"]

    config.setdefault("runtime", {}).update(
        {
            "cli_config_path": "outputs/generated_configs/D48_seed1.yaml",
            "run_dir": "outputs/D48/seed1",
        }
    )

    assert canonical_t2_design_config_sha256(config) == screen["config_sha256"]
    assert t2_design_candidate_recipe_sha256(config) == screen["candidate_recipe_sha256"]
    assert training_profile_checkpoint_provenance(config)["t2_design_screening"]["candidate_id"] == "D48"


def test_existing_manifest_recomputes_config_and_probe_identity(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)
    config = launcher.build_design_config(
        "D48",
        tmp_path,
        seed=1,
        batch_size=32,
        inner_split_fingerprint="inner-split",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    screen = config["mmw_t2_design_screening"]
    job = {
        "candidate": "D48",
        "gpu": 0,
        "config_path": config_path.name,
        "status": "planned",
        "config_sha256": screen["config_sha256"],
        "candidate_recipe_sha256": screen["candidate_recipe_sha256"],
        "training_profile_sha256": screen["training_profile_sha256"],
        "inner_split_fingerprint": screen["inner_split_fingerprint"],
    }
    identity = {
        "candidate": "D48",
        "training_profile_id": launcher.H4_PROFILE,
        "training_profile_sha256": screen["training_profile_sha256"],
        "candidate_recipe_sha256": screen["candidate_recipe_sha256"],
    }
    manifest = {
        "protocol": "mmw_t2_design_screening_v1",
        "seed": 1,
        "batch_size": 32,
        "training_profile_id": launcher.H4_PROFILE,
        "inner_split_fingerprint": "inner-split",
        "candidate_fingerprints": {"D48": screen["config_sha256"]},
        "candidate_recipe_fingerprints": {"D48": screen["candidate_recipe_sha256"]},
        "profile_fingerprints": {"D48": screen["training_profile_sha256"]},
        "jobs": [job],
    }
    probe = {
        "selected_common_batch_size": 32,
        "required_identities": {0: identity},
        "records": {"0": [{**identity, "status": "safe"}]},
    }
    monkeypatch.setattr(launcher, "ROOT", tmp_path)

    launcher._validate_existing_manifest(
        manifest,
        variants=("D48",),
        gpus=(0,),
        seed=1,
        batch_size=32,
        batch_probe=probe,
    )

    tampered = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tampered["model"]["primary"]["d_model"] = 49
    config_path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256|fingerprint"):
        launcher._validate_existing_manifest(
            manifest,
            variants=("D48",),
            gpus=(0,),
            seed=1,
            batch_size=32,
            batch_probe=probe,
        )


def test_launcher_aborts_started_siblings_when_a_process_cannot_start(monkeypatch, tmp_path) -> None:
    launcher = _load_launcher(monkeypatch)

    class Process:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):  # noqa: ARG002
            return -15 if self.terminated else 0

    process = Process()
    calls = iter((process, OSError("synthetic startup failure")))

    def fake_popen(*_args, **_kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    (tmp_path / "logs").mkdir()
    manifest_path = tmp_path / "design_manifest.json"
    manifest = {
        "jobs": [
            {"candidate": "H4-control", "gpu": 0, "config_path": "control.yaml", "log_path": "logs/control.log"},
            {"candidate": "D48", "gpu": 1, "config_path": "d48.yaml", "log_path": "logs/d48.log"},
        ]
    }

    assert launcher.launch_jobs(manifest_path, manifest) == 1
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert process.terminated is True
    assert persisted["jobs"][0]["status"] == "aborted"
    assert persisted["jobs"][1]["status"] == "failed_to_start"
