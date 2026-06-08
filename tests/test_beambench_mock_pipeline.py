from __future__ import annotations

from pathlib import Path

from kd_sensing.baselines.beambench.mock import create_mock_dataset
from kd_sensing.baselines.beambench.pipeline import MockTrainingConfig, evaluate_checkpoint, train_mock_baseline


def test_mock_pipeline_trains_saves_loads_and_evaluates(tmp_path: Path):
    data_root = tmp_path / "mock_dataset"
    output_dir = tmp_path / "mock_run"
    create_mock_dataset(data_root, rows=8, num_beams=8)

    report = train_mock_baseline(
        MockTrainingConfig(
            data_root=str(data_root),
            csv="ml_challenge_mock_multi_modal.csv",
            output_dir=str(output_dir),
            num_beams=8,
            epochs=1,
            batch_size=4,
            seed=7,
            device="cpu",
        )
    )

    checkpoint = Path(report["checkpoint_path"])
    assert report["mock_data"] is True
    assert checkpoint.exists()
    assert report["metrics"]["valid_label_count"] == 8
    eval_report = evaluate_checkpoint(
        checkpoint,
        data_root=data_root,
        csv="ml_challenge_mock_multi_modal.csv",
        output_dir=output_dir / "eval",
        device="cpu",
    )
    assert eval_report["mock_data"] is True
    assert eval_report["metrics"]["valid_label_count"] == 8
