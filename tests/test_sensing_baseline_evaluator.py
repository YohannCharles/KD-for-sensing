from __future__ import annotations

import pytest
import torch

from kd_sensing.eval.beam_probe_diagnostic import EXPECTED_MODALITIES
from kd_sensing.eval.sensing_baseline import (
    build_baseline_probe_evidence,
    collect_sensing_baseline_observations,
    validate_baseline_checkpoint,
    validate_sensing_only_baseline_config,
)
from kd_sensing.utils.checkpoint import publish_checkpoint


class _TaskForwardBaseline(torch.nn.Module):
    modalities = EXPECTED_MODALITIES

    def forward(self, *, image_batch, radar_batch, gps_batch, lidar_batch, missing_mask=None, **_):
        del radar_batch, gps_batch, lidar_batch
        batch = image_batch.shape[0]
        # Distinct logits verify that the collector uses TaskForwardResult.logits
        # and applies softmax to the selected future slot.
        logits = torch.zeros(batch, 1, 64, device=image_batch.device)
        logits[:, 0, 3] = 5.0
        if missing_mask is not None:
            logits[:, 0, 4] = missing_mask[:, 0].float()
        return {"logits": logits, "output_features": None, "input_features": None}


def _batch() -> dict:
    return {
        "image": torch.zeros(2, 5, 3, 1, 1),
        "radar_ra": torch.zeros(2, 5, 1, 128, 64),
        "radar_da": torch.zeros(2, 5, 1, 128, 64),
        "gps": torch.zeros(2, 5, 3),
        "lidar": torch.zeros(2, 5, 3, 1, 1),
        "target_beam": torch.tensor([[3], [4]]),
        "metadata": [
            {
                "condition": "sunny",
                "scenario": "scene",
                "stable_sample_id": "mmw:sunny:scene:validation:s0",
                "source_sample_id": "s0",
            },
            {
                "condition": "sunny",
                "scenario": "scene",
                "stable_sample_id": "mmw:sunny:scene:validation:s1",
                "source_sample_id": "s1",
            },
        ],
    }


def test_collector_uses_future_logits_and_enumerates_all_masks() -> None:
    cfg = {
        "experiment": {"task": "fusion"},
        "model": {"primary": {"seq_length": 5, "num_pred": 1}},
    }
    records = collect_sensing_baseline_observations(
        _TaskForwardBaseline(), [_batch()], cfg, device="cpu"
    )
    assert records["logits"].shape == (30, 64)
    assert records["probabilities"].shape == (30, 64)
    assert tuple(records["logits"][0].topk(1).indices.tolist()) == (3,)
    assert len(set(records["pattern"])) == 15
    evidence = build_baseline_probe_evidence(
        records,
        source={"data_protocol": {"validation_sample_count": 2}},
    )
    assert len(evidence.sample_id) == 30
    assert evidence.pred_prob.shape == (30, 64)
    assert all(abs(float(row.sum()) - 1.0) < 1e-5 for row in evidence.pred_prob)


def test_baseline_evidence_rejects_incomplete_mask_matrix() -> None:
    records = {
        "modalities": list(EXPECTED_MODALITIES),
        "pattern": ["full"],
        "sample_id": ["s0"],
        "labels": torch.tensor([0]),
        "available": torch.ones(1, 4, dtype=torch.bool),
        "logits": torch.zeros(1, 64),
        "probabilities": torch.full((1, 64), 1 / 64),
    }
    with pytest.raises(ValueError, match="15 non-empty"):
        build_baseline_probe_evidence(
            records,
            source={"data_protocol": {"validation_sample_count": 1}},
        )


def _config_pair() -> tuple[dict, dict]:
    protocol = {
        "protocol_id": "mmw_id_stratified_block_v1",
        "test_evaluated": False,
        "outer_test_accessed": False,
        "validation_sample_count": 1,
    }
    topology = {
        "experiment": {"seed": 1, "train_seed": 1},
        "data_protocol": protocol,
        "model": {
            "primary": {
                "type": "four_modal_topology_predictor",
                "modalities": list(EXPECTED_MODALITIES),
                "seq_length": 5,
                "num_pred": 1,
            }
        },
        "loss": {
            "four_modal_topology": {
                "prototype_topology": {
                    "id": "ula_dft_phase_cycle_v1",
                    "descriptor_sha256": "d" * 64,
                    "audit_sha256": "a" * 64,
                }
            }
        },
    }
    baseline = {
        "experiment": {"seed": 1, "train_seed": 1},
        "data_protocol": dict(protocol),
        "data": {"dataset": {"type": "mmw"}},
        "model": {
            "primary": {
                "type": "modular_sequence",
                "modalities": list(EXPECTED_MODALITIES),
                "seq_length": 5,
                "num_pred": 1,
                "num_classes": 64,
                "representation_core": {"type": "rmbp_channel_attention_fusion"},
            }
        },
        "training": {"final_test": {"enabled": False}},
    }
    return baseline, topology


def test_baseline_topology_binding_rejects_protocol_or_history_mismatch() -> None:
    baseline, topology = _config_pair()
    baseline["data_protocol"]["split_seed"] = 9
    with pytest.raises(ValueError, match="one MMW protocol"):
        validate_sensing_only_baseline_config(baseline, topology)
    baseline, topology = _config_pair()
    baseline["model"]["primary"]["history_beam_index"] = True
    with pytest.raises(ValueError, match="history-beam"):
        validate_sensing_only_baseline_config(baseline, topology)


def test_baseline_checkpoint_requires_validation_best_role(tmp_path) -> None:
    baseline, topology = _config_pair()
    checkpoint = tmp_path / "last.pth"
    torch.save(
        {
            "checkpoint_role": "last",
            "model_metadata": {
                "type": "modular_sequence",
                "modalities": list(EXPECTED_MODALITIES),
                "representation_core_type": "rmbp_channel_attention_fusion",
            },
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="validation_best"):
        validate_baseline_checkpoint(checkpoint, baseline, topology)


def test_baseline_checkpoint_uses_recorded_config_when_model_metadata_is_absent(tmp_path) -> None:
    baseline, topology = _config_pair()
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": "validation_best",
        "model_metadata": None,
        "data_protocol": baseline["data_protocol"],
        "experiment_seed": 1,
        "resume_contract": {"config": baseline},
    }
    checkpoint, _ = publish_checkpoint(payload, tmp_path, "best.pth")
    loaded, digest = validate_baseline_checkpoint(checkpoint, baseline, topology)
    assert loaded["model_metadata"] is None
    assert len(digest) == 64
