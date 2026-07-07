from pathlib import Path

import torch
import torch.nn as nn

from kd_sensing.baselines.amber_lite import amber_lite_summary_row, normalize_missing_modality_suite
from kd_sensing.config import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.difficulty.pipeline import apply_configured_difficulty
from kd_sensing.data.difficulty.schema import DifficultyContext
from kd_sensing.models.modular import AmberLiteMissingModalityTransformerCore, ModularSequenceModel
from kd_sensing.registries import ENCODERS


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("amber_lite_test_identity", force=True)
class AmberLiteTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"amber_lite_test_identity expects [B, T, D], got {tuple(batch.shape)}.")
        return batch[..., : self.output_dim]


def test_amber_lite_forward_full_single_and_multi_missing_modalities() -> None:
    model = _amber_lite_test_model()
    batch = _synthetic_modalities()

    full = model(**batch)
    missing_image = model(**batch, image_valid_mask=torch.tensor([[False, False, False], [True, True, True]]))
    missing_multi = model(
        **batch,
        radar_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
        gps_valid_mask=torch.tensor([[True, False, True], [False, False, True]]),
        lidar_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
    )

    assert isinstance(model.representation_core, AmberLiteMissingModalityTransformerCore)
    assert full["logits"].shape == (2, 3, 6)
    assert missing_image["missing_modality_metadata"]["missing_counts"]["image"] == 3
    assert missing_multi["missing_modality_metadata"]["missing_counts"]["radar"] == 6
    assert missing_multi["missing_modality_metadata"]["missing_counts"]["lidar"] == 6
    assert model.training_strategy_metadata()["reproduction_scope"] == "amber_lite_local"
    assert model.training_strategy_metadata()["consumes_missing_modality_metadata"] is True


def test_modality_dropout_policy_is_deterministic_and_preserves_targets() -> None:
    batch = {
        "image": torch.ones(2, 3, 3, 4, 4),
        "radar_ra": torch.ones(2, 3, 1, 128, 64),
        "radar_da": torch.ones(2, 3, 1, 128, 64),
        "gps": torch.ones(2, 3, 3),
        "lidar": torch.ones(2, 3, 3, 4, 4),
        "target_beam": torch.tensor([[1, 2], [3, 4]]),
        "beam_power": torch.randn(2, 2, 6),
        "sample_id": ["a", "b"],
        "split": "train",
    }
    cfg = {
        "experiment": {"seed": 5},
        "difficulty": {
            "profiles": [
                {
                    "id": "amber_lite_train",
                    "stage": "train",
                    "split": "train",
                    "severity": 1.0,
                    "seed": 7,
                    "fallback": "zero_fill",
                    "affected_modalities": ["image", "radar", "gps", "lidar"],
                    "operators": [
                        {
                            "type": "amber_lite_modality_dropout",
                            "modality": "image",
                            "affected_modalities": ["image", "radar", "gps", "lidar"],
                            "rates": {"image": 1.0, "radar": 1.0, "gps": 1.0, "lidar": 1.0},
                        }
                    ],
                }
            ]
        },
    }

    context = DifficultyContext(stage="train", split="train", seed=5, epoch=0, step=2)
    first = apply_configured_difficulty(batch, cfg, context).batch
    second = apply_configured_difficulty(batch, cfg, context).batch

    assert torch.equal(first["target_beam"], batch["target_beam"])
    assert torch.equal(first["beam_power"], batch["beam_power"])
    assert first["sample_id"] == batch["sample_id"]
    assert first["split"] == batch["split"]
    for modality in ("image", "radar", "gps", "lidar"):
        assert torch.equal(first[f"{modality}_dropout_mask"], second[f"{modality}_dropout_mask"])
        assert first[f"{modality}_valid_mask"].any().item() is False
    assert first["missing_modality_metadata"]["fallback_count"] == 0
    assert set(first["missing_modality_metadata"]["rates"]) == {"image", "radar", "gps", "lidar"}


def test_amber_lite_suite_and_summary_keep_claim_boundary() -> None:
    manifest = safe_load_yaml((ROOT / "configs/diagnostics/amber_lite_missing_modality_eval.yaml").read_text())
    suite = normalize_missing_modality_suite(manifest)

    ids = {condition["id"] for condition in suite["conditions"]}
    assert {"clean", "missing_image", "missing_lidar", "missing_radar", "missing_gps", "poor_image", "wrong_gps", "async_gps"} <= ids
    assert suite["reproduction_scope"] == "amber_lite_local"
    assert suite["conditions"][0]["expected_availability_mask"] == {
        "image": True,
        "radar": True,
        "gps": True,
        "lidar": True,
    }

    row = amber_lite_summary_row(
        model="amber_lite",
        source="fixture",
        metrics_by_condition={
            "clean": {"top1": 0.1, "top3": 0.2, "top5": 0.3, "dba": 0.4, "beam_distance": 2.0},
            "missing_gps": {"status": "pending"},
        },
        comparability={"split": "test", "sample_count": 2},
        real_metrics=False,
        lidar_artifact_available=False,
        radar_artifact_available=False,
    )

    assert row["reproduction_scope"] == "amber_lite_local"
    assert row["status"] == "pending"
    assert row["strict_ranking_eligible"] is False
    assert row["condition_metrics"][0]["dba"] == 0.4


def test_amber_lite_config_loads_and_records_dropout_digest() -> None:
    cfg = load_config(ROOT / "configs/fusion/amber_lite_missing_modality.yaml")

    primary = cfg["model"]["primary"]
    profile = cfg["difficulty"]["profiles"][0]
    operator = profile["operators"][0]
    assert primary["type"] == "modular_sequence"
    assert primary["representation_core"]["type"] == "amber_lite_missing_modality_transformer"
    assert primary["consume_missing_modality_metadata"] is True
    assert primary["encoders"]["image"]["pretrained"] is False
    assert primary["encoders"]["image"]["weights"] is None
    assert primary["paper_metadata"]["baseline_scope"] == "local_experimental_baseline"
    assert cfg["output"]["dir"].startswith("outputs/analysis/local_baselines/amber_lite_missing_modality")
    assert profile["affected_modalities"] == ["image", "radar", "gps", "lidar"]
    assert operator["affected_modalities"] == ["image", "radar", "gps", "lidar"]
    assert operator["digest"]


def _amber_lite_test_model() -> ModularSequenceModel:
    modalities = ["image", "radar", "gps", "lidar"]
    return ModularSequenceModel(
        modalities=modalities,
        encoders={modality: {"type": "amber_lite_test_identity", "output_dim": 8} for modality in modalities},
        projectors={modality: {"type": "identity"} for modality in modalities},
        representation_core={
            "type": "amber_lite_missing_modality_transformer",
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 3,
        },
        heads={"beam": {"type": "beam_head"}},
        feature_size=8,
        d_model=8,
        num_classes=6,
        num_pred=1,
        paper_metadata={"reproduction_scope": "amber_lite_local"},
    )


def _synthetic_modalities() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.randn(2, 3, 8),
        "radar_batch": torch.randn(2, 3, 8),
        "gps_batch": torch.randn(2, 3, 8),
        "lidar_batch": torch.randn(2, 3, 8),
    }
