import json

import pytest
import torch

from kd_sensing.eval.export import save_results_csv, save_results_json, save_results_markdown
from kd_sensing.eval.metrics import reliability_error_stats
from kd_sensing.eval.missing_patterns import (
    get_default_missing_patterns,
    make_fixed_missing_mask,
    resolve_missing_patterns,
    sample_eval_random_missing_mask,
)
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import evaluate_missing_matrix
from kd_sensing.cli.eval_u_mask_beam_jepa_matrix import _resolve_split


def test_default_missing_patterns_for_four_modalities():
    patterns = get_default_missing_patterns(["image", "radar", "lidar", "gps"])

    assert patterns == {
        "full": [1, 1, 1, 1],
        "missing_image": [0, 1, 1, 1],
        "missing_radar": [1, 0, 1, 1],
        "missing_lidar": [1, 1, 0, 1],
        "missing_gps": [1, 1, 1, 0],
        "only_image": [1, 0, 0, 0],
        "only_radar": [0, 1, 0, 0],
        "only_lidar": [0, 0, 1, 0],
        "only_gps": [0, 0, 0, 1],
        "missing_image_radar": [0, 0, 1, 1],
        "missing_image_lidar": [0, 1, 0, 1],
        "missing_image_gps": [0, 1, 1, 0],
        "missing_radar_lidar": [1, 0, 0, 1],
        "missing_radar_gps": [1, 0, 1, 0],
        "missing_lidar_gps": [1, 1, 0, 0],
        "non_gps_only": [1, 1, 1, 0],
    }


def test_resolve_missing_patterns_accepts_argparse_default_list():
    assert resolve_missing_patterns(["default"], ["image", "radar", "gps", "lidar"]) == get_default_missing_patterns(
        ["image", "radar", "gps", "lidar"]
    )


def test_make_fixed_missing_mask_shape_and_contents():
    mask = make_fixed_missing_mask(2, [1, 0, 1, 1], device="cpu")

    assert mask.shape == (2, 4)
    assert torch.equal(mask, torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.float32))


def test_random_missing_mask_keeps_at_least_one_modality_available():
    mask = sample_eval_random_missing_mask(64, 4, p_missing=1.0, ensure_at_least_one=True)

    assert mask.shape == (64, 4)
    assert torch.all(mask.sum(dim=1) >= 1)


def test_reliability_error_stats_correct_wrong_reliability():
    logits = torch.tensor([[5.0, 1.0], [0.0, 3.0], [2.0, 0.0]])
    target = torch.tensor([0, 0, 1])
    global_reliability = torch.tensor([0.9, 0.2, 0.1])
    modality_reliability = torch.tensor([[[0.8], [0.4]], [[0.1], [0.3]], [[0.2], [0.6]]])
    missing_mask = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)

    stats = reliability_error_stats(logits, target, global_reliability, modality_reliability, missing_mask)

    assert stats["mean_global_reliability_correct"] == pytest.approx(0.9)
    assert stats["mean_global_reliability_wrong"] == pytest.approx(0.15)
    assert stats["mean_available_modality_reliability"] == pytest.approx((0.8 + 0.1 + 0.3 + 0.6) / 4)


def test_export_helpers_write_files(tmp_path):
    results = [{"pattern": "full", "mask": "1,1", "num_samples": 2, "top1": 1.0, "top5": 1.0}]

    save_results_csv(results, tmp_path / "eval_matrix.csv")
    save_results_json(results, tmp_path / "eval_matrix.json")
    save_results_markdown(results, tmp_path / "eval_matrix.md")

    assert "pattern,mask" in (tmp_path / "eval_matrix.csv").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "eval_matrix.json").read_text(encoding="utf-8"))[0]["pattern"] == "full"
    assert "| pattern |" in (tmp_path / "eval_matrix.md").read_text(encoding="utf-8")


def test_eval_matrix_runner_with_fake_model_and_dataloader():
    model = _FakeUMaskModel()
    dataloader = [
        {"x": torch.ones(2, 1), "target": torch.tensor([[0, 1], [1, 2]])},
        {"x": torch.ones(1, 1), "target": torch.tensor([[2, 0]])},
    ]

    results = evaluate_missing_matrix(
        model,
        dataloader,
        torch.device("cpu"),
        ["image", "radar", "lidar", "gps"],
        random_missing=[0.5],
        prediction_index="last",
    )

    names = {row["pattern"] for row in results}
    assert {"full", "missing_image", "only_gps", "non_gps_only", "random_0.5"} <= names
    assert all(row["num_samples"] == 3 for row in results)
    assert all("top1" in row and "mean_available_modality_reliability" in row for row in results)


def test_cli_val_split_falls_back_to_test_when_validation_is_absent():
    assert _resolve_split({"train": object(), "test": object()}, "val") == "test"


class _FakeUMaskModel(torch.nn.Module):
    def forward(self, batch=None, *, missing_mask=None, **kwargs):
        if batch is None:
            batch = kwargs
        batch_size = int(next(value for value in batch.values() if torch.is_tensor(value)).shape[0])
        base = torch.tensor(
            [
                [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
                [[0.0, 2.0, 3.0], [2.0, 0.0, 3.0]],
            ],
            device=missing_mask.device,
        )
        logits = base[:batch_size].clone()
        if batch_size > logits.shape[0]:
            logits = logits.repeat(batch_size, 1, 1)[:batch_size]
        global_reliability = missing_mask.float().mean(dim=1)
        return {
            "logits": logits,
            "global_reliability": global_reliability,
            "modality_reliability": missing_mask.unsqueeze(-1) * 0.75,
            "missing_mask": missing_mask,
        }
