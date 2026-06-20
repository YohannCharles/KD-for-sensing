from kd_sensing.diagnostics.geometry_prior_beam_fusion import (
    aggregate_geometry_prior_diagnostics,
    build_geometry_prior_claim_gate,
    build_geometry_prior_diagnostics_bundle_manifest,
)


def _manifest() -> dict:
    return {
        "geometry_prior_claim_gate": {
            "baseline_group": "resnet_image_gps",
            "gps_query_baseline_group": "jepa_gps_query_pool",
            "candidate_groups": ["geometry_prior_fusion"],
            "clean_regression_threshold_dba": 0.02,
            "claim_margin_dba": 0.0,
        },
        "comparability": {
            "mode": "strict",
            "keys": ["split", "sample_count", "beam_label_space", "seed"],
        },
    }


def test_geometry_prior_diagnostics_mark_missing_fields_unavailable():
    rows = [
        {
            "model": "geometry_prior_logit_fusion",
            "group": "geometry_prior_fusion",
            "condition": "P0_clean_current",
            "split": "test",
            "seed": 17,
            "dba": 0.88,
            "sample_count": 1088,
            "beam_label_space": "beam64",
            "prior_entropy": 1.2,
            "prior_weight_mean": 0.2,
            "image_weight_mean": 0.8,
        },
        {
            "model": "image_resnet_gps",
            "group": "resnet_image_gps",
            "condition": "P0_clean_current",
            "split": "test",
            "seed": 17,
            "dba": 0.885,
            "sample_count": 1088,
            "beam_label_space": "beam64",
        },
    ]

    diagnostics = aggregate_geometry_prior_diagnostics(rows, manifest=_manifest())
    bundle = build_geometry_prior_diagnostics_bundle_manifest(
        _manifest(),
        diagnostics=diagnostics,
        claim_gate={"claim_status": "pending"},
    )

    prior_row = diagnostics["prior_quality"][0]
    branch_row = diagnostics["branch_weights"][0]
    assert prior_row["prior_standalone_dba"] == "unavailable"
    assert prior_row["prior_entropy"] == 1.2
    assert branch_row["prior_image_agreement"] == "unavailable"
    assert branch_row["status"] == "available"
    assert diagnostics["strict_comparison"][0]["claim_gate_status"] == "ready"
    assert bundle["diagnostics"]["missing_fields_are_unavailable"] is True
    assert bundle["diagnostics"]["prior_quality_rows"] == 2


def test_geometry_prior_claim_gate_failed_pending_and_pass_states():
    manifest = _manifest()
    baseline = {
        "model": "image_resnet_gps",
        "group": "resnet_image_gps",
        "condition": "P0_clean_current",
        "dba": 0.885,
        "comparability_status": "passed",
    }
    failed = build_geometry_prior_claim_gate(
        [
            baseline,
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P0_clean_current",
                "dba": 0.84,
                "comparability_status": "passed",
            },
        ],
        manifest,
    )
    pending = build_geometry_prior_claim_gate(
        [
            baseline,
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P0_clean_current",
                "dba": 0.875,
                "comparability_status": "pending",
            },
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P1_current_frame_missing_history_available",
                "dba": 0.86,
                "comparability_status": "pending",
            },
        ],
        manifest,
    )
    passed = build_geometry_prior_claim_gate(
        [
            baseline,
            {
                "model": "image_resnet_gps",
                "group": "resnet_image_gps",
                "condition": "P1_current_frame_missing_history_available",
                "dba": 0.84,
                "comparability_status": "passed",
            },
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P0_clean_current",
                "dba": 0.884,
                "comparability_status": "passed",
                "status": "real_forward",
                "evidence_scope": "real_forward",
            },
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P1_current_frame_missing_history_available",
                "dba": 0.86,
                "comparability_status": "passed",
                "status": "real_forward",
                "evidence_scope": "real_forward",
            },
        ],
        manifest,
    )

    assert failed["claim_status"] == "failed"
    assert failed["candidate_statuses"][0]["reason"] == "clean_regression_exceeds_threshold"
    assert pending["claim_status"] == "pending"
    assert pending["candidate_statuses"][0]["reason"] == "comparability_unavailable"
    assert passed["claim_status"] == "pass"
    assert passed["candidate_statuses"][0]["clean_gate_pass"] is True
    assert passed["advantage_only_cannot_upgrade_primary_claim"] is True

    delegated = build_geometry_prior_claim_gate(
        [
            baseline,
            {
                "model": "image_resnet_gps",
                "group": "resnet_image_gps",
                "condition": "P1_current_frame_missing_history_available",
                "dba": 0.84,
                "comparability_status": "passed",
                "status": "delegated_evaluate",
            },
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P0_clean_current",
                "dba": 0.884,
                "comparability_status": "passed",
                "status": "delegated_evaluate",
            },
            {
                "model": "geometry_prior_logit_fusion",
                "group": "geometry_prior_fusion",
                "condition": "P1_current_frame_missing_history_available",
                "dba": 0.86,
                "comparability_status": "passed",
                "status": "delegated_evaluate",
            },
        ],
        manifest,
    )

    assert delegated["claim_status"] == "pending"
    assert delegated["candidate_statuses"][0]["real_perturbation_forward"] is False
    assert delegated["candidate_statuses"][0]["reason"] == "delegated_clean_only_perturbations_not_real_forward"
