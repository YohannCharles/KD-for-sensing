import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kd_sensing.engine.trainer_runtime_helpers import _TimingCsvLogger


def _batch_result() -> SimpleNamespace:
    return SimpleNamespace(timings={}, total_loss=torch.tensor(1.25))


def test_timing_is_default_off_without_clock_probe_or_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    logger = _TimingCsvLogger({}, tmp_path, device=torch.device("cpu"))
    monkeypatch.setattr(
        "kd_sensing.engine.trainer_runtime_helpers.time.perf_counter",
        lambda: (_ for _ in ()).throw(AssertionError("disabled timing must not read the clock")),
    )

    assert logger.host_now() is None
    assert logger.start_step() is None
    logger.maybe_log(
        epoch=0,
        batch=0,
        data_time=None,
        step_time=None,
        batch_result=_batch_result(),
        lr=1e-3,
    )
    logger.flush()

    assert not (tmp_path / "timing.csv").exists()


def test_host_timing_buffers_rows_until_flush(tmp_path: Path):
    cfg = {"training": {"timing": {"enabled": True, "profile": "host", "log_interval": 1}}}
    logger = _TimingCsvLogger(cfg, tmp_path, device=torch.device("cpu"))

    wait_start = logger.host_now()
    token = logger.start_step()
    step_time = logger.finish_step(token)
    logger.maybe_log(
        epoch=0,
        batch=0,
        data_time=logger.host_elapsed(wait_start),
        step_time=step_time,
        batch_result=_batch_result(),
        lr=1e-3,
    )

    assert not logger.path.exists()
    logger.flush()
    with logger.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert logger.path == tmp_path / "timing.csv"
    assert len(rows) == 1
    assert rows[0]["profile"] == "host"


def test_enabled_timing_requires_explicit_profile(tmp_path: Path):
    with pytest.raises(ValueError, match="training.timing.profile"):
        _TimingCsvLogger({"training": {"timing": {"enabled": True}}}, tmp_path, device=torch.device("cpu"))


def test_cuda_event_profile_synchronizes_only_sampled_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events = []
    synchronizations = []

    class FakeEvent:
        def __init__(self, **_kwargs):
            events.append(self)

        def record(self):
            return None

        def elapsed_time(self, _other):
            return 5.0

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronizations.append(True))
    cfg = {"training": {"timing": {"enabled": True, "profile": "cuda_event", "log_interval": 2}}}
    logger = _TimingCsvLogger(cfg, tmp_path, device=torch.device("cuda"))

    first = logger.start_step()
    assert logger.finish_step(first) == pytest.approx(0.005)
    logger.maybe_log(
        epoch=0,
        batch=0,
        data_time=0.0,
        step_time=0.005,
        batch_result=_batch_result(),
        lr=1e-3,
    )
    second = logger.start_step()

    assert second is None
    assert len(events) == 2
    assert len(synchronizations) == 1
