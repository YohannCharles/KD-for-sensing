from pathlib import Path

import pytest
import torch

from kd_sensing.config import load_config
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import (
    _beam_classification_metrics,
    evaluate_missing_matrix,
    evaluate_missing_matrix_single_pass,
)
from kd_sensing.evaluation.metrics import calculate_dba_score
from kd_sensing.utils.missing_patterns import get_default_missing_patterns, make_fixed_missing_mask


ROOT = Path(__file__).resolve().parents[1]


def test_mmw_missing_pattern_matrix_uses_four_current_modalities() -> None:
    patterns = get_default_missing_patterns(["image", "radar", "gps", "lidar"])

    assert patterns["full"] == [1, 1, 1, 1]
    assert patterns["missing_image"] == [0, 1, 1, 1]
    assert patterns["gps_only"] == [0, 0, 1, 0]
    assert make_fixed_missing_mask(2, patterns["missing_gps"], device="cpu").shape == (2, 4)


def test_eval_matrix_reports_fixed_current_patterns() -> None:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml", overrides=["temporal_missing.history_window=1"])
    patterns = get_default_missing_patterns(["image", "radar", "gps", "lidar"])
    results = evaluate_missing_matrix(
        FakeUMaskModel(),
        [
            {
                "image": torch.ones(2, 1, 1, 1, 1),
                "radar_ra": torch.ones(2, 1, 1, 128, 64),
                "radar_da": torch.ones(2, 1, 1, 128, 64),
                "gps": torch.ones(2, 1, 1),
                "lidar": torch.ones(2, 1, 1, 1, 1),
                "target_beam": torch.tensor([[0], [1]]),
            }
        ],
        torch.device("cpu"),
        ["image", "radar", "gps", "lidar"],
        patterns=patterns,
        cfg=cfg,
        prediction_index="last",
    )

    names = {row["pattern"] for row in results}
    assert {"full", "missing_image", "gps_only", "avg_missing"} <= names
    assert all(row["num_samples"] == 2 for row in results)
    assert all(row["metric_profile"] == "64_beam_circular_topk_progressive_top3_dba_v1" for row in results)


def test_single_pass_eval_matches_pattern_first_metrics() -> None:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml", overrides=["temporal_missing.history_window=1"])
    patterns = {"full": [1, 1, 1, 1], "gps_only": [0, 0, 1, 0]}
    batches = [
        {
            "image": torch.ones(2, 1, 1, 1, 1),
            "radar_ra": torch.ones(2, 1, 1, 128, 64),
            "radar_da": torch.ones(2, 1, 1, 128, 64),
            "gps": torch.ones(2, 1, 1),
            "lidar": torch.ones(2, 1, 1, 1, 1),
            "target_beam": torch.tensor([[0], [1]]),
        }
    ]
    args = (torch.device("cpu"), ["image", "radar", "gps", "lidar"])
    expected = evaluate_missing_matrix(FakeUMaskModel(), batches, *args, patterns=patterns, cfg=cfg)
    actual = evaluate_missing_matrix_single_pass(FakeUMaskModel(), batches, *args, patterns=patterns, cfg=cfg)

    assert [row["pattern"] for row in actual] == [row["pattern"] for row in expected]
    for expected_row, actual_row in zip(expected, actual):
        for metric in ("num_samples", "loss", "top1", "top3", "top5", "mae"):
            assert actual_row[metric] == pytest.approx(expected_row[metric])


def test_fixed_mask_adba_uses_progressive_top3_dba() -> None:
    logits = torch.full((1, 64), -10.0)
    logits[0, 4], logits[0, 63], logits[0, 2] = 8.0, 7.0, 6.0
    target = torch.tensor([0])
    cfg = {"evaluation": {"dba_delta": 5, "dba_distance_mode": "circular"}}

    metrics = _beam_classification_metrics(logits, target, cfg)
    expected = float(calculate_dba_score(logits, target, 5, distance_mode="circular")[0])

    assert metrics["adba"] == pytest.approx(expected)
    assert metrics["adba"] != pytest.approx(metrics["top1_proximity_dba"])


class FakeUMaskModel(torch.nn.Module):
    def forward(self, *, image_batch, missing_mask=None, **_):
        batch_size = image_batch.shape[0]
        missing_mask = (
            missing_mask
            if missing_mask is not None
            else torch.ones(batch_size, 4, dtype=torch.bool, device=image_batch.device)
        )
        logits = torch.zeros(batch_size, 1, 3, device=image_batch.device)
        logits[:, :, 0] = 1.0
        return {
            "logits": logits,
            "global_reliability": missing_mask.float().mean(dim=1),
            "modality_reliability": missing_mask.unsqueeze(-1).float(),
            "missing_mask": missing_mask,
        }
