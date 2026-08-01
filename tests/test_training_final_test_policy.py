from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kd_sensing.engine import data_factory, trainer


def _dataloader_config(final_test: object | None = None) -> dict:
    training = {} if final_test is None else {"final_test": final_test}
    return {
        "training": training,
        "experiment": {"seed": 7},
        "data": {
            "dataloader": {
                "train_batch_size": 2,
                "validation_batch_size": 2,
                "test_batch_size": 2,
            },
        },
    }


@pytest.mark.parametrize(
    ("final_test", "expected_splits"),
    [
        ({"enabled": False}, ["train", "validation"]),
        (None, ["train", "validation", "test"]),
    ],
)
def test_build_dataloaders_respects_final_test_policy(monkeypatch, final_test, expected_splits) -> None:
    seen_splits: list[str] = []

    def build_split_dataset(_cfg, split, **_kwargs):
        seen_splits.append(split)
        return object()

    monkeypatch.setattr(data_factory, "build_split_dataset", build_split_dataset)
    monkeypatch.setattr(data_factory, "fit_gps_scaler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_factory, "gps_scaler_kwargs", lambda _dataset: {})
    monkeypatch.setattr(data_factory, "has_validation_csv", lambda _cfg: True)
    monkeypatch.setattr(
        data_factory,
        "build_dataloader",
        lambda _dataset, _loader_cfg, *, split, **_kwargs: {"split": split},
    )

    dataloaders = data_factory.build_dataloaders(_dataloader_config(final_test))

    assert seen_splits == expected_splits
    assert sorted(dataloaders) == sorted(expected_splits)


def test_mmw_defaults_to_no_final_test() -> None:
    assert data_factory.final_test_enabled({"data": {"dataset": {"type": "mmw"}}, "training": {}}) is False


def test_mmw_test_requires_explicit_runtime_authorization() -> None:
    cfg = {
        "data": {"dataset": {"type": "mmw"}},
        "training": {"final_test": {"enabled": True}},
        "runtime": {"evaluate_test_requested": False},
    }
    with pytest.raises(ValueError, match="explicit --evaluate-test"):
        data_factory.final_test_enabled(cfg)

    cfg["runtime"]["evaluate_test_requested"] = True
    assert data_factory.final_test_enabled(cfg) is True


def _finalize_context(tmp_path, *, final_test: object | None, dataloaders: dict) -> SimpleNamespace:
    return SimpleNamespace(
        cfg={"training": {} if final_test is None else {"final_test": final_test}},
        primary_model=object(),
        dataloaders=dataloaders,
        task_criterion=object(),
        device=object(),
        run_dir=tmp_path,
        state=SimpleNamespace(history={}, epoch_logs=[], checkpoint_loads=[]),
        objective_metadata={},
        artifact_writer=Mock(write_final_artifacts=Mock(return_value={})),
        optimizer_groups=[],
        normalization_artifacts={},
        throughput_metadata={},
        split_metadata={},
        startup_summary={},
    )


def test_finalize_training_run_skips_outer_test_when_disabled(monkeypatch, tmp_path) -> None:
    context = _finalize_context(tmp_path, final_test={"enabled": False}, dataloaders={"train": object()})
    evaluate_final_test = Mock(side_effect=AssertionError("outer test must not be evaluated"))
    monkeypatch.setattr(trainer, "_evaluate_final_test_split", evaluate_final_test)
    monkeypatch.setattr(trainer, "write_complete_status", lambda *_args, **_kwargs: None)

    result = trainer._finalize_training_run(context)

    assert evaluate_final_test.call_count == 0
    assert result["final_test_metrics"] == {
        "status": "not_run",
        "reason": "training.final_test.enabled=false",
    }
    assert context.state.checkpoint_loads == []
    assert context.artifact_writer.write_final_artifacts.call_args.kwargs["final_test_metrics"] == result["final_test_metrics"]


def test_finalize_training_run_keeps_final_test_enabled_by_default(monkeypatch, tmp_path) -> None:
    test_loader = object()
    context = _finalize_context(tmp_path, final_test=None, dataloaders={"test": test_loader})
    metrics = {"evaluation_split": "test"}
    checkpoint_load = {"role": "final-test-last"}
    evaluate_final_test = Mock(return_value=(metrics, checkpoint_load))
    monkeypatch.setattr(trainer, "_evaluate_final_test_split", evaluate_final_test)
    monkeypatch.setattr(trainer, "write_complete_status", lambda *_args, **_kwargs: None)

    result = trainer._finalize_training_run(context)

    assert evaluate_final_test.call_args.args[1] is test_loader
    assert context.state.checkpoint_loads == [checkpoint_load]
    assert result["final_test_metrics"] == metrics
