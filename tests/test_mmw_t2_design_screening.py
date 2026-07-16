import importlib.util
from pathlib import Path

import pytest

from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.utils.artifact_registry import (
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

    assert d96["model"]["primary"]["d_model"] == 96
    assert {entry["output_dim"] for entry in d96["model"]["primary"]["encoders"].values()} == {96}
    assert router["model"]["primary"]["router_hidden_dim"] == 32
    assert gps["model"]["primary"]["encoders"]["gps"]["hidden_size"] == 128
    assert d96["mmw_t2_design_screening"]["matched_control"] == "H4-control"


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
    config = launcher.build_design_config("D48", tmp_path, seed=1, batch_size=32)
    provenance = training_profile_checkpoint_provenance(config)

    assert provenance["training_profile"]["id"] == "umask_h4_v1"
    assert provenance["t2_design_screening"]["candidate_id"] == "D48"
    validate_evaluation_training_profile_provenance(config, provenance)

    mismatched = {**provenance, "t2_design_screening": {**provenance["t2_design_screening"], "candidate_id": "D96"}}
    with pytest.raises(ValueError, match="design-screening provenance"):
        validate_evaluation_training_profile_provenance(config, mismatched)
