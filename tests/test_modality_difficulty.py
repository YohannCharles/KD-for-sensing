from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd_sensing.config.io import load_config
from kd_sensing.data.difficulty import (
    DifficultyContext,
    DifficultyOperatorOutcome,
    apply_configured_difficulty,
    apply_difficulty_pipeline,
    normalize_config_difficulty,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import GPS_QUERY_ADVANTAGE_CONDITION_IDS, PREDICTIVE_JEPA_CONDITION_IDS
from kd_sensing.diagnostics import jepa_gps_shortcut_benchmark as bench
from kd_sensing.engine.batch import forward_model
from kd_sensing.engine.batch_step import BatchStepRunner
from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.runtime import prepare_task_inputs
from kd_sensing.engine.training_extensions import ExtensionContext
from kd_sensing.modalities import difficulty_metadata_fields
from kd_sensing.registries import DIFFICULTY_OPERATORS, RegistryError


ROOT = Path(__file__).resolve().parents[1]


class _GpsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.last_gps_batch: torch.Tensor | None = None

    def forward(self, gps_batch=None, **kwargs):  # noqa: ANN001, ARG002
        self.last_gps_batch = gps_batch.detach().cpu().clone()
        return {"logits": self.logits.expand(gps_batch.shape[0], 1, 4)}


class _DisabledGradScaler:
    def is_enabled(self) -> bool:
        return False


def _base_cfg() -> dict:
    return {
        "experiment": {"task": "fusion", "objective": "beam", "seed": 7},
        "data": {"dataset": {}},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length": 3,
            "num_classes": 4,
            "primary": {"modalities": ["gps"]},
        },
        "training": {"transfer": {"non_blocking": False}, "amp": {"enabled": False}},
        "evaluation": {"k_values": [1], "dba_delta": 5},
    }


def _batch() -> dict:
    return {
        "gps": torch.arange(6, dtype=torch.float32).reshape(2, 3, 1),
        "image": torch.ones(2, 3, 3, 4, 4, dtype=torch.float32),
        "target_beam": torch.tensor([[0], [2]]),
        "beam_power": torch.arange(8, dtype=torch.float32).reshape(2, 1, 4),
        "metadata": {"sample_id": ["a", "b"], "split": ["train", "train"]},
    }


def _advantage_batch() -> dict:
    image = torch.zeros(4, 3, 1, 4, 4, dtype=torch.float32)
    image[0] = 0.10
    image[1] = 0.12
    image[2] = 0.80
    image[3] = 0.82
    return {
        "gps": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        "image": image,
        "target_beam": torch.tensor([[0], [4], [8], [12]]),
        "beam_power": torch.arange(16, dtype=torch.float32).reshape(4, 1, 4),
        "metadata": {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "split": ["test", "test", "test", "test"],
            "scene": ["scene32", "scene32", "scene32", "scene32"],
        },
    }


def _delay_profile(stage: str = "train", *, severity: float = 1.0, seed: int = 3):
    return normalize_difficulty_profiles(
        [
            {
                "id": "gps_delay",
                "stage": stage,
                "split": "train" if stage == "train" else "test",
                "condition": "delay",
                "severity": severity,
                "seed": seed,
                "fallback": "zero_fill",
                "operators": [{"type": "temporal_delay", "modality": "gps", "max_delay_steps": 1}],
            }
        ],
        default_seed=seed,
    )[0]


def test_profile_digest_is_stable_and_validation_rejects_bad_profiles() -> None:
    profile_a = normalize_difficulty_profiles(
        [
            {
                "id": "gps_async",
                "stage": "train",
                "condition": "async",
                "severity": 1,
                "seed": 5,
                "fallback": "forward_fill",
                "operators": [{"type": "temporal_delay", "modality": "gps", "max_delay_steps": 1}],
            }
        ]
    )[0]
    profile_b = normalize_difficulty_profiles(
        [
            {
                "fallback": "forward_fill",
                "operators": [{"max_delay_steps": 1, "modality": "gps", "type": "temporal_delay"}],
                "seed": 5,
                "severity": 1,
                "condition": "async",
                "stage": "train",
                "id": "gps_async",
            }
        ]
    )[0]
    profile_c = normalize_difficulty_profiles(
        [
            {
                "id": "gps_async",
                "stage": "train",
                "condition": "async",
                "severity": 2,
                "seed": 5,
                "fallback": "forward_fill",
                "operators": [{"type": "temporal_delay", "modality": "gps", "max_delay_steps": 1}],
            }
        ]
    )[0]

    assert profile_a.digest == profile_b.digest
    assert profile_a.digest != profile_c.digest
    with pytest.raises(RegistryError, match="gps_magic_noise.*difficulty_operators"):
        normalize_difficulty_profiles([{"id": "bad", "operator": "gps_magic_noise"}])
    with pytest.raises(ValueError, match="pseudo modality 'delayed_gps'.*canonical modality 'gps'"):
        normalize_difficulty_profiles([{"id": "pseudo", "operator": {"type": "gps_clean", "modality": "delayed_gps"}}])
    with pytest.raises(ValueError, match="attempts to move target fields"):
        normalize_difficulty_profiles([{"id": "shift", "operator": "gps_clean", "target_shift": True}])
    with pytest.raises(ValueError, match="Allowed stages"):
        normalize_difficulty_profiles([{"id": "stage", "stage": "preprocess_dataset_files", "operator": "gps_clean"}])


def test_scenario_d_profile_normalizes_canonical_conditions_and_rejects_bad_config() -> None:
    profiles = normalize_difficulty_profiles(
        [
            {
                "id": f"profile_{index}",
                "stage": "benchmark",
                "condition": condition,
                "operators": [{"type": "scenario_d_image_observability"}],
            }
            for index, condition in enumerate(
                [
                    "D0_full_image",
                    "D1_weather",
                    "D2_low_light",
                    "D3_motion_blur",
                    "D4_partial_occlusion",
                    "D5_frame_dropout",
                    "D6_burst_missing",
                    "D7_joint_worst_case",
                ]
            )
        ],
        default_stage="benchmark",
    )

    assert [profile.condition for profile in profiles] == [
        "D0_full_image",
        "D1_weather",
        "D2_low_light",
        "D3_motion_blur",
        "D4_partial_occlusion",
        "D5_frame_dropout",
        "D6_burst_missing",
        "D7_joint_worst_case",
    ]
    d7 = profiles[-1]
    assert d7.severity == 7.0
    assert d7.operators[0].modality == "image"
    assert d7.operators[0].affected_modalities == ("image",)
    assert d7.operators[0].params["image_occlusion_prob"] == 0.5
    assert d7.operators[0].params["image_burst_dropout_prob"] == 0.5
    assert d7.operators[0].params["max_burst_len"] == 3

    with pytest.raises(ValueError, match="Unknown Scenario D image observability condition 'D9_magic'.*Available D-levels"):
        normalize_difficulty_profiles(
            [{"id": "bad", "condition": "D9_magic", "operator": "scenario_d_image_observability"}],
            default_stage="benchmark",
        )
    with pytest.raises(ValueError, match="image_dropout_prob=.*must be in \\[0, 1\\]"):
        normalize_difficulty_profiles(
            [
                {
                    "id": "bad_prob",
                    "condition": "D5_frame_dropout",
                    "operator": {"type": "scenario_d_image_observability", "image_dropout_prob": 1.5},
                }
            ]
        )
    with pytest.raises(ValueError, match="max_burst_len must be positive"):
        normalize_difficulty_profiles(
            [
                {
                    "id": "bad_burst",
                    "condition": "D6_burst_missing",
                    "operator": {"type": "scenario_d_image_observability", "max_burst_len": 0},
                }
            ]
        )
    with pytest.raises(ValueError, match="pseudo modality 'missing_image_modality'.*canonical modality 'image'"):
        normalize_difficulty_profiles(
            [
                {
                    "id": "pseudo_image",
                    "condition": "D0_full_image",
                    "operator": {"type": "scenario_d_image_observability", "modality": "missing_image_modality"},
                }
            ]
        )


def test_predictive_jepa_profile_normalizes_p_levels_and_rejects_unknown_condition() -> None:
    profiles = normalize_difficulty_profiles(
        [
            {
                "id": f"predictive_{index}",
                "stage": "benchmark",
                "condition": condition,
                "operators": [{"type": "predictive_jepa_robustness"}],
            }
            for index, condition in enumerate(PREDICTIVE_JEPA_CONDITION_IDS)
        ],
        default_stage="benchmark",
    )

    assert [profile.condition for profile in profiles] == list(PREDICTIVE_JEPA_CONDITION_IDS)
    p4 = profiles[4]
    assert p4.severity == 4.0
    assert p4.operators[0].type == "predictive_jepa_robustness"
    assert p4.operators[0].modality == "image"
    assert p4.operators[0].affected_modalities == ("image", "gps")
    assert p4.operators[0].params["current_frame_missing"] is True
    assert p4.operators[0].params["semantic_occlusion"] is True
    assert p4.operators[0].params["plausible_wrong_gps"] is True

    with pytest.raises(ValueError, match="Unknown Predictive JEPA robustness condition 'P9_magic'.*Available P-levels"):
        normalize_difficulty_profiles(
            [{"id": "bad", "condition": "P9_magic", "operator": "predictive_jepa_robustness"}],
            default_stage="benchmark",
        )
    with pytest.raises(ValueError, match="history_window must be positive"):
        normalize_difficulty_profiles(
            [
                {
                    "id": "bad_history",
                    "condition": "P1_current_frame_missing_history_available",
                    "operator": {"type": "predictive_jepa_robustness", "history_window": 0},
                }
            ],
            default_stage="benchmark",
        )


def test_gps_query_advantage_profile_normalizes_without_expanding_p_suite() -> None:
    profile = normalize_difficulty_profiles(
        [
            {
                "id": "advantage_a2",
                "stage": "benchmark",
                "condition": "A2_visual_ambiguous_wrong_gps",
                "operator": {"type": "predictive_jepa_robustness", "min_beam_offset": 4},
            }
        ],
        default_stage="benchmark",
    )[0]

    assert GPS_QUERY_ADVANTAGE_CONDITION_IDS == (
        "A0_visual_ambiguous_peer",
        "A1_beam_offset_wrong_gps",
        "A2_visual_ambiguous_wrong_gps",
    )
    assert list(PREDICTIVE_JEPA_CONDITION_IDS) == [
        "P0_clean_current",
        "P1_current_frame_missing_history_available",
        "P2_semantic_occlusion_history_available",
        "P3_plausible_wrong_gps_current_image",
        "P4_joint_predictive_recovery",
        "P5_novel_weather_history_available",
    ]
    assert profile.condition == "A2_visual_ambiguous_wrong_gps"
    assert profile.severity == 12.0
    assert profile.operators[0].params["visual_ambiguous_peer"] is True
    assert profile.operators[0].params["beam_offset_constrained_wrong_gps"] is True
    assert profile.operators[0].params["min_beam_offset"] == 4


def test_image_observability_transform_is_deterministic_and_preserves_targets() -> None:
    context = DifficultyContext(stage="benchmark", split="test", seed=11, sample_ids=("a", "b"))
    clean_profile = normalize_difficulty_profiles(
        [
            {
                "id": "clean_d0",
                "stage": "benchmark",
                "condition": "D0_full_image",
                "operator": "scenario_d_image_observability",
            }
        ],
        default_stage="benchmark",
    )[0]
    clean = apply_difficulty_pipeline(_batch(), clean_profile, context)
    assert torch.equal(clean.batch["image"], _batch()["image"])
    assert clean.batch["image_valid_mask"].tolist() == [[True, True, True], [True, True, True]]
    assert torch.allclose(clean.batch["image_observability_score"], torch.ones(2, 3))

    physical_profile = normalize_difficulty_profiles(
        [
            {
                "id": "physical_d4",
                "stage": "benchmark",
                "condition": "D4_partial_occlusion",
                "operators": [
                    {
                        "type": "scenario_d_image_observability",
                        "image_occlusion_prob": 1.0,
                        "image_occlusion_ratio": 0.5,
                    }
                ],
            }
        ],
        default_stage="benchmark",
    )[0]
    physical = apply_difficulty_pipeline(_batch(), physical_profile, context)
    assert physical.batch["image"].shape == _batch()["image"].shape
    assert physical.batch["image"].dtype == _batch()["image"].dtype
    assert bool(physical.batch["image_valid_mask"].all())
    assert physical.batch["image_observability_score"].max().item() < 1.0
    assert torch.equal(physical.batch["target_beam"], _batch()["target_beam"])
    assert physical.batch["metadata"]["sample_id"] == ["a", "b"]

    d7_profile = normalize_difficulty_profiles(
        [
            {
                "id": "joint_d7",
                "stage": "benchmark",
                "condition": "D7_joint_worst_case",
                "operators": [
                    {
                        "type": "scenario_d_image_observability",
                        "image_occlusion_prob": 1.0,
                        "image_occlusion_ratio": 0.5,
                        "image_burst_dropout_prob": 1.0,
                        "max_burst_len": 2,
                    }
                ],
            }
        ],
        default_stage="benchmark",
    )[0]
    first = apply_difficulty_pipeline(_batch(), d7_profile, context)
    second = apply_difficulty_pipeline(_batch(), d7_profile, context)

    assert torch.equal(first.batch["image"], second.batch["image"])
    assert torch.equal(first.batch["image_valid_mask"], second.batch["image_valid_mask"])
    assert torch.equal(first.batch["image_burst_dropout_mask"], second.batch["image_burst_dropout_mask"])
    assert torch.equal(first.batch["image_observability_score"], second.batch["image_observability_score"])
    assert not bool(first.batch["image_valid_mask"].all())
    assert bool(first.batch["image_burst_dropout_mask"].any())
    assert first.batch["image_observability_score"].min().item() == 0.0
    assert "partial_occlusion" in first.batch["image_degradation_metadata"]["corruption_types"]
    assert "burst_missing" in first.batch["image_degradation_metadata"]["corruption_types"]
    assert first.batch["image_degradation_metadata"]["physical_corruption_keeps_valid"] is True
    assert first.batch["image_degradation_metadata"]["missing_invalidates_frame"] is True
    assert first.batch["image_observability_replay"]["condition"] == "D7_joint_worst_case"
    assert torch.equal(first.batch["target_beam"], _batch()["target_beam"])
    assert torch.equal(first.batch["beam_power"], _batch()["beam_power"])


def test_predictive_jepa_pipeline_is_deterministic_no_label_shift_and_no_future_leak() -> None:
    context = DifficultyContext(stage="benchmark", split="test", seed=23, sample_ids=("a", "b"))
    profile = normalize_difficulty_profiles(
        [
            {
                "id": "predictive_p4",
                "stage": "benchmark",
                "condition": "P4_joint_predictive_recovery",
                "operators": [{"type": "predictive_jepa_robustness", "history_window": 2}],
            }
        ],
        default_stage="benchmark",
    )[0]

    first = apply_difficulty_pipeline(_batch(), profile, context)
    second = apply_difficulty_pipeline(_batch(), profile, context)

    assert torch.equal(first.batch["image"], second.batch["image"])
    assert torch.equal(first.batch["gps"], second.batch["gps"])
    assert torch.equal(first.batch["image_valid_mask"], second.batch["image_valid_mask"])
    assert torch.equal(first.batch["gps_source_sample_index"], second.batch["gps_source_sample_index"])
    assert torch.equal(first.batch["target_beam"], _batch()["target_beam"])
    assert torch.equal(first.batch["beam_power"], _batch()["beam_power"])
    assert first.batch["metadata"]["sample_id"] == ["a", "b"]
    assert first.batch["difficulty"]["condition"] == "P4_joint_predictive_recovery"

    current_step = 2
    assert first.batch["image_valid_mask"].tolist() == [[True, True, False], [True, True, False]]
    assert first.batch["image_current_missing_mask"][:, current_step].tolist() == [True, True]
    assert first.batch["image_observability_score"][:, current_step].tolist() == [0.0, 0.0]
    assert torch.equal(first.batch["image"][:, :2], _batch()["image"][:, :2])
    ranges = first.batch["image_degradation_metadata"]["history_source_range"]
    assert ranges[current_step] == [0, 1]
    assert all(item is None or item[1] < index for index, item in enumerate(ranges))

    assert first.batch["gps_counterfactual_mask"][:, current_step].tolist() == [True, True]
    assert not torch.equal(first.batch["gps"][:, current_step], _batch()["gps"][:, current_step])
    gps_metadata = first.batch["gps_counterfactual_metadata"]
    assert gps_metadata["counterfactual_input_intervention"] is True
    assert gps_metadata["counterfactual_status"] == "counterfactual_peer_replacement"
    assert gps_metadata["scene_constraint"] == "same_split_or_batch"
    assert "mean_l2" in gps_metadata["distance_criteria"]
    replay = first.batch["predictive_jepa_replay_metadata"]
    assert replay["condition"] == "P4_joint_predictive_recovery"
    assert replay["image"]["history_source_range"][current_step] == [0, 1]
    assert replay["gps"]["source_sample_index"] == "gps_source_sample_index"


def test_predictive_jepa_conditions_cover_image_variants_and_gps_fallback() -> None:
    context = DifficultyContext(stage="benchmark", split="test", seed=31, sample_ids=("solo",))
    p2 = normalize_difficulty_profiles(
        [
            {
                "id": "predictive_p2",
                "stage": "benchmark",
                "condition": "P2_semantic_occlusion_history_available",
                "operator": {"type": "predictive_jepa_robustness", "history_window": 2},
            }
        ],
        default_stage="benchmark",
    )[0]
    p5 = normalize_difficulty_profiles(
        [
            {
                "id": "predictive_p5",
                "stage": "benchmark",
                "condition": "P5_novel_weather_history_available",
                "operator": {"type": "predictive_jepa_robustness", "history_window": 2},
            }
        ],
        default_stage="benchmark",
    )[0]
    p3 = normalize_difficulty_profiles(
        [
            {
                "id": "predictive_p3",
                "stage": "benchmark",
                "condition": "P3_plausible_wrong_gps_current_image",
                "operator": {"type": "predictive_jepa_robustness", "history_window": 2},
            }
        ],
        default_stage="benchmark",
    )[0]

    p2_result = apply_difficulty_pipeline(_batch(), p2, DifficultyContext(stage="benchmark", split="test", seed=31, sample_ids=("a", "b")))
    assert p2_result.batch["image_valid_mask"].all()
    assert "semantic_occlusion_proxy" in p2_result.batch["image_degradation_metadata"]["corruption_types"]
    assert p2_result.batch["image_semantic_frame_mask"][:, -1].tolist() == [True, True]

    p5_result = apply_difficulty_pipeline(_batch(), p5, DifficultyContext(stage="benchmark", split="test", seed=31, sample_ids=("a", "b")))
    assert p5_result.batch["image_valid_mask"].all()
    assert "novel_weather" in p5_result.batch["image_degradation_metadata"]["corruption_types"]
    assert p5_result.batch["image_observability_score"][:, -1].max().item() < 1.0

    solo_batch = _batch()
    solo_batch["gps"] = solo_batch["gps"][:1]
    solo_batch["image"] = solo_batch["image"][:1]
    solo_batch["target_beam"] = solo_batch["target_beam"][:1]
    solo_batch["beam_power"] = solo_batch["beam_power"][:1]
    solo_batch["metadata"] = {"sample_id": ["solo"], "split": ["test"]}
    fallback = apply_difficulty_pipeline(solo_batch, p3, context)
    assert fallback.batch["gps_counterfactual_metadata"]["counterfactual_status"] == "counterfactual_fallback_jitter"
    assert fallback.batch["gps_counterfactual_metadata"]["fallback_reason"] == "insufficient_batch_peer_pool"
    assert [warning.code for warning in fallback.warnings] == ["predictive_jepa_plausible_wrong_gps_fallback"]


def test_gps_query_advantage_difficulty_is_deterministic_and_beam_offset_constrained() -> None:
    profile = normalize_difficulty_profiles(
        [
            {
                "id": "advantage_a2",
                "stage": "benchmark",
                "condition": "A2_visual_ambiguous_wrong_gps",
                "operator": {
                    "type": "predictive_jepa_robustness",
                    "history_window": 2,
                    "min_beam_offset": 4,
                    "scene_constraint": "same_split_or_batch",
                    "visual_ambiguous_top_k": 2,
                },
            }
        ],
        default_stage="benchmark",
    )[0]
    context = DifficultyContext(stage="benchmark", split="test", seed=41, sample_ids=("s0", "s1", "s2", "s3"))

    first = apply_difficulty_pipeline(_advantage_batch(), profile, context)
    second = apply_difficulty_pipeline(_advantage_batch(), profile, context)
    changed_seed = apply_difficulty_pipeline(
        _advantage_batch(),
        profile,
        DifficultyContext(stage="benchmark", split="test", seed=42, sample_ids=("s0", "s1", "s2", "s3")),
    )

    assert torch.equal(first.batch["image"], second.batch["image"])
    assert torch.equal(first.batch["gps"], second.batch["gps"])
    assert torch.equal(first.batch["gps_source_sample_index"], second.batch["gps_source_sample_index"])
    assert first.batch["predictive_jepa_replay_metadata"] == second.batch["predictive_jepa_replay_metadata"]
    assert torch.equal(first.batch["target_beam"], _advantage_batch()["target_beam"])
    assert torch.equal(first.batch["beam_power"], _advantage_batch()["beam_power"])
    assert first.batch["metadata"]["sample_id"] == ["s0", "s1", "s2", "s3"]

    visual = first.batch["visual_ambiguous_hard_negative_metadata"]
    assert visual["fallback_count"] == 0
    assert all(offset is not None and offset >= 4 for offset in visual["beam_offset"])
    assert all(peer_id in {"s0", "s1", "s2", "s3"} for peer_id in visual["peer_sample_id"])
    assert first.batch["predictive_jepa_replay_metadata"]["image"]["visual_ambiguous_peer"]["min_beam_offset"] == 4.0

    gps_metadata = first.batch["gps_counterfactual_metadata"]
    assert gps_metadata["counterfactual_status"] == "counterfactual_peer_replacement"
    assert gps_metadata["fallback_count"] == 0
    assert all(offset >= 4 for offset in gps_metadata["beam_offset_criteria"]["offsets"])
    assert all(size >= 1 for size in gps_metadata["selection_pool_size"])
    assert gps_metadata["peer_sample_id"] == [
        first.batch["metadata"]["sample_id"][int(index)]
        for index in first.batch["gps_source_sample_index"][:, -1].tolist()
    ]
    assert not torch.equal(first.batch["gps"][:, -1], _advantage_batch()["gps"][:, -1])

    changed_offsets = changed_seed.batch["gps_counterfactual_metadata"]["beam_offset_criteria"]["offsets"]
    assert all(offset >= 4 for offset in changed_offsets)
    assert changed_seed.batch["predictive_jepa_replay_metadata"]["gps"]["selection_pool_size"] == gps_metadata["selection_pool_size"]


def test_cxd_difficulty_profile_preserves_labels_soft_targets_and_split_metadata() -> None:
    suite = bench.normalize_suite_config({"id": "scenario_cxd", "type": "scenario_c_x_d_image_observability"})
    gps_condition = next(item for item in suite["scenario_c_conditions"] if item["id"] == "C4_severe_async")
    image_condition = next(item for item in suite["scenario_d_conditions"] if item["id"] == "D7_joint_worst_case")
    profile = bench._difficulty_profile_from_cxd_pair(
        suite,
        gps_condition=gps_condition,
        image_condition=image_condition,
        seed=19,
    )
    batch = _batch()
    batch["soft_target"] = torch.eye(4, dtype=torch.float32)[[0, 2]].unsqueeze(1)
    batch["split_metadata"] = {"fold": "unit", "rows": [1, 2]}
    batch["metadata"] = {
        **batch["metadata"],
        "split_metadata": {"fold": "unit", "rows": [1, 2]},
    }
    before = {
        "target_beam": batch["target_beam"].clone(),
        "beam_power": batch["beam_power"].clone(),
        "soft_target": batch["soft_target"].clone(),
        "sample_id": list(batch["metadata"]["sample_id"]),
        "split": list(batch["metadata"]["split"]),
        "split_metadata": dict(batch["metadata"]["split_metadata"]),
    }

    result = apply_difficulty_pipeline(
        batch,
        profile,
        DifficultyContext(stage="benchmark", split="test", seed=19, sample_ids=("a", "b")),
    )

    assert torch.equal(result.batch["target_beam"], before["target_beam"])
    assert torch.equal(result.batch["beam_power"], before["beam_power"])
    assert torch.equal(result.batch["soft_target"], before["soft_target"])
    assert result.batch["metadata"]["sample_id"] == before["sample_id"]
    assert result.batch["metadata"]["split"] == before["split"]
    assert result.batch["metadata"]["split_metadata"] == before["split_metadata"]
    assert result.batch["difficulty"]["condition"] == "C4_severe_async+D7_joint_worst_case"
    assert result.batch["metadata"]["difficulty_profiles"][0]["profile"]["metadata"]["gps_condition"] == "C4_severe_async"
    assert result.batch["metadata"]["difficulty_profiles"][0]["profile"]["metadata"]["image_condition"] == "D7_joint_worst_case"


def test_image_observability_metadata_fields_are_queryable() -> None:
    fields = difficulty_metadata_fields("image")

    for key in (
        "image_valid_mask",
        "image_observability_score",
        "image_dropout_mask",
        "image_burst_dropout_mask",
        "image_degradation_metadata",
    ):
        assert key in fields
    assert "not target supervision" in fields["image_observability_score"].lower()


def test_batch_mapping_passes_reliability_metadata_only_for_opt_in_models() -> None:
    batch = _batch()
    batch.update(
        {
            "image_valid_mask": torch.tensor([[True, False, True], [True, True, False]]),
            "image_observability_score": torch.tensor([[1.0, 0.2, 0.8], [0.9, 0.7, 0.0]]),
            "image_dropout_mask": torch.tensor([[False, True, False], [False, False, True]]),
            "image_burst_dropout_mask": torch.tensor([[False, False, False], [False, False, True]]),
            "gps_valid_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "gps_delay_steps": torch.tensor([[0, 1, 4], [0, 2, 3]]),
            "metadata": {
                "sample_id": ["a", "b"],
                "split": ["test", "test"],
                "benchmark_perturbation": {
                    "gps_condition": "C4_severe_async",
                    "image_condition": "D6_burst_missing",
                },
            },
        }
    )

    baseline_inputs = prepare_task_inputs(
        batch,
        "fusion",
        model_cfg={"modalities": ["image", "gps"], "image_profile": "rgb_imagenet"},
        seq_length=2,
        num_pred=2,
        device=torch.device("cpu"),
    )
    assert "image_valid_mask" not in baseline_inputs
    assert "gps_delay_steps" not in baseline_inputs

    aware_inputs = prepare_task_inputs(
        batch,
        "fusion",
        model_cfg={
            "modalities": ["image", "gps"],
            "image_profile": "rgb_imagenet",
            "observability_aware_fusion": {"enabled": True},
        },
        seq_length=2,
        num_pred=2,
        device=torch.device("cpu"),
    )

    assert aware_inputs["image_valid_mask"].tolist() == [[False, True, True], [True, False, True]]
    assert torch.allclose(
        aware_inputs["image_observability_score"],
        torch.tensor([[0.2, 0.8, 1.0], [0.7, 0.0, 1.0]]),
    )
    assert aware_inputs["gps_valid_mask"].tolist() == [[True, False, True], [False, False, True]]
    assert torch.allclose(
        aware_inputs["gps_delay_steps"],
        torch.tensor([[1.0, 4.0, 0.0], [2.0, 3.0, 0.0]]),
    )
    assert aware_inputs["benchmark_condition_metadata"]["gps_condition"] == "C4_severe_async"
    assert aware_inputs["benchmark_condition_metadata"]["image_condition"] == "D6_burst_missing"

    geometry_inputs = prepare_task_inputs(
        batch,
        "fusion",
        model_cfg={
            "modalities": ["image", "gps"],
            "image_profile": "rgb_imagenet",
            "geometry_prior": {"enabled": True},
        },
        seq_length=2,
        num_pred=2,
        device=torch.device("cpu"),
    )
    assert "gps_valid_mask" in geometry_inputs
    assert "gps_delay_steps" in geometry_inputs
    assert geometry_inputs["benchmark_condition_metadata"]["gps_condition"] == "C4_severe_async"

    missing = dict(batch)
    missing.pop("image_observability_score")
    with pytest.raises(ValueError, match="required reliability metadata 'image_observability_score' is missing"):
        prepare_task_inputs(
            missing,
            "fusion",
            model_cfg={
                "modalities": ["image", "gps"],
                "image_profile": "rgb_imagenet",
                "observability_aware_fusion": {"enabled": True},
            },
            seq_length=2,
            num_pred=1,
            device=torch.device("cpu"),
        )
    geometry_missing = prepare_task_inputs(
        missing,
        "fusion",
        model_cfg={
            "modalities": ["image", "gps"],
            "image_profile": "rgb_imagenet",
            "geometry_prior": {"enabled": True},
        },
        seq_length=2,
        num_pred=1,
        device=torch.device("cpu"),
    )
    assert "image_observability_score" not in geometry_missing
    assert "gps_valid_mask" in geometry_missing


def test_forward_model_filters_unaccepted_reliability_metadata() -> None:
    class _FusionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_valid_mask = None

        def forward(self, image_batch=None, gps_batch=None, image_valid_mask=None):  # noqa: ANN001
            self.image_valid_mask = image_valid_mask
            return {"logits": torch.zeros(image_batch.shape[0], 1, 4)}

    model = _FusionModel()
    image = torch.randn(2, 3, 3, 8, 8)
    gps = torch.randn(2, 3, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    dropout = torch.zeros(2, 3, dtype=torch.bool)

    output = forward_model(
        model,
        "fusion",
        image_batch=image,
        gps_batch=gps,
        image_valid_mask=valid,
        image_dropout_mask=dropout,
    )

    assert output["logits"].shape == (2, 1, 4)
    assert model.image_valid_mask is valid


def test_pipeline_is_deterministic_shape_safe_and_blocks_target_mutation() -> None:
    profile = normalize_difficulty_profiles(
        [
            {
                "id": "mixed",
                "stage": "train",
                "severity": 0.5,
                "seed": 9,
                "operators": [
                    {"type": "gps_missing", "modality": "gps", "dropout_prob": 0.5},
                    {"type": "image_occlusion", "modality": "image"},
                ],
            }
        ]
    )[0]
    context = DifficultyContext(stage="train", split="train", seed=9, sample_ids=("a", "b"))
    first = apply_difficulty_pipeline(_batch(), profile, context)
    second = apply_difficulty_pipeline(_batch(), profile, context)

    assert torch.equal(first.batch["gps"], second.batch["gps"])
    assert torch.equal(first.batch["image"], second.batch["image"])
    assert first.batch["gps"].shape == _batch()["gps"].shape
    assert first.batch["image"].dtype == _batch()["image"].dtype
    assert torch.equal(first.batch["target_beam"], _batch()["target_beam"])
    assert torch.equal(first.batch["beam_power"], _batch()["beam_power"])
    assert "difficulty_profiles" in first.batch["metadata"]

    delay = apply_difficulty_pipeline(_batch(), _delay_profile(), context)
    source = delay.batch["gps_source_index"]
    current = torch.arange(source.shape[1]).reshape(1, -1)
    assert bool(((source == -1) | (source <= current)).all())

    class _TargetMutator:
        def __init__(self, **params):  # noqa: ANN003
            pass

        def __call__(self, batch, *, config, profile, context):  # noqa: ANN001, ARG002
            batch["target_beam"] = batch["target_beam"] + 1
            return DifficultyOperatorOutcome()

    DIFFICULTY_OPERATORS.register("unit_target_mutator", force=True)(_TargetMutator)
    bad_profile = normalize_difficulty_profiles([{"id": "bad", "operator": "unit_target_mutator"}])[0]
    with pytest.raises(RuntimeError, match="changed protected field 'target_beam'"):
        apply_difficulty_pipeline(_batch(), bad_profile, context)


def test_load_config_normalizes_difficulty_after_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "difficulty.yaml"
    config_path.write_text(
        f"""
_base_: {ROOT / 'configs/gps/lightweight.yaml'}
difficulty:
  profiles:
    gps_async:
      stage: train
      split: train
      condition: gps_async
      severity: 1
      seed: 5
      fallback: forward_fill
      operators:
        - type: temporal_delay
          modality: gps
          max_delay_steps: 1
""",
        encoding="utf-8",
    )

    base = load_config(config_path)
    overridden = load_config(config_path, ["difficulty.profiles.gps_async.severity=2"])

    assert base["difficulty"]["profiles"][0]["id"] == "gps_async"
    assert base["difficulty"]["profiles"][0]["digest"] != overridden["difficulty"]["profiles"][0]["digest"]
    assert overridden["difficulty"]["profiles"][0]["severity"] == 2.0
    assert "difficulty" not in load_config(ROOT / "configs/gps/lightweight.yaml")


def test_train_and_evaluation_hooks_are_stage_scoped() -> None:
    eval_cfg = _base_cfg()
    eval_cfg["difficulty"] = {"profiles": [_delay_profile(stage="evaluation").to_dict()]}
    model = _GpsModel()
    criterion = torch.nn.CrossEntropyLoss()

    result = run_evaluation_pass(model, [_batch()], eval_cfg, criterion, torch.device("cpu"))

    assert model.last_gps_batch is not None
    assert model.last_gps_batch[0, :, 0].tolist() == [0.0, 0.0, 1.0]
    assert result.metadata[0]["difficulty_profiles"][0]["profile_digest"] == eval_cfg["difficulty"]["profiles"][0]["digest"]

    train_only_cfg = _base_cfg()
    train_only_cfg["difficulty"] = {"profiles": [_delay_profile(stage="train").to_dict()]}
    clean_model = _GpsModel()
    run_evaluation_pass(clean_model, [_batch()], train_only_cfg, criterion, torch.device("cpu"))
    assert clean_model.last_gps_batch is not None
    assert clean_model.last_gps_batch[0, :, 0].tolist() == [0.0, 1.0, 2.0]

    train_cfg = _base_cfg()
    train_cfg["difficulty"] = {"profiles": [_delay_profile(stage="train").to_dict()]}
    train_model = _GpsModel()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1)
    context = ExtensionContext(
        cfg=train_cfg,
        task="fusion",
        model_cfg=train_cfg["model"],
        training_cfg=train_cfg["training"],
        primary_model=train_model,
        task_criterion=criterion,
        run_dir=ROOT,
        device=torch.device("cpu"),
        num_pred=1,
        num_classes=4,
        seq_length=3,
        non_blocking=False,
    )
    runner = BatchStepRunner(
        cfg=train_cfg,
        task="fusion",
        model_cfg=train_cfg["model"],
        training_cfg=train_cfg["training"],
        optimizer=optimizer,
        grad_scaler=_DisabledGradScaler(),
        amp_enabled=False,
        amp_dtype=torch.float32,
        extension_context=context,
        extensions=[],
        extension_states=[],
    )

    batch_result = runner.run(_batch(), epoch=0, step=0, current_alpha=0.0)

    assert train_model.last_gps_batch is not None
    assert train_model.last_gps_batch[0, :, 0].tolist() == [0.0, 0.0, 1.0]
    assert torch.equal(batch_result.batch["target_beam"], _batch()["target_beam"])


def test_benchmark_wrapper_uses_shared_difficulty_pipeline_and_records_provenance() -> None:
    batch = {
        "gps": torch.arange(5, dtype=torch.float32).reshape(1, 5, 1),
        "target_beam": torch.tensor([[3]]),
        "metadata": {"sample_id": ["toy"]},
    }
    suite = {
        "id": "delay",
        "type": "temporal_delay",
        "modality": "gps",
        "severities": [2],
        "fallback": "zero_fill",
    }

    result, warnings = bench.apply_benchmark_perturbation(batch, suite, severity=2, seed=17)

    assert warnings == []
    assert result["gps"].flatten().tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert result["metadata"]["benchmark_perturbation"]["difficulty_profile_digest"] == result["difficulty"]["profile_digest"]
    provenance = bench._benchmark_difficulty_provenance({"perturbation_suites": [bench.normalize_suite_config(suite)], "seeds": [17]})
    assert provenance[0]["profile"]["digest"] == result["difficulty"]["profile_digest"]


def test_apply_configured_difficulty_noop_path_preserves_batch_object_semantics() -> None:
    batch = _batch()
    result = apply_configured_difficulty(batch, _base_cfg(), DifficultyContext(stage="train", split="train"))

    assert result.metadata == {"enabled": False, "state": "clean"}
    assert torch.equal(result.batch["gps"], batch["gps"])


def test_normalize_config_difficulty_supports_data_and_evaluation_locations() -> None:
    cfg = {
        "experiment": {"seed": 1},
        "data": {"difficulty": {"profiles": [{"id": "train_clean", "operator": "gps_clean"}]}},
        "evaluation": {"difficulty": {"profiles": [{"id": "eval_clean", "operator": "image_clean"}]}},
    }

    profiles = normalize_config_difficulty(cfg)

    assert [profile.id for profile in profiles] == ["train_clean", "eval_clean"]
    assert cfg["difficulty"]["profiles"][0]["stages"] == ["train"]
    assert cfg["difficulty"]["profiles"][1]["stages"] == ["evaluation"]
