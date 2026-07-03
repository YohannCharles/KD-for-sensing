import csv
import json
from pathlib import Path

from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint


def test_resolve_checkpoint_keeps_seed_run_ownership(tmp_path: Path):
    root = tmp_path / "scene31"
    registry = root / "best_checkpoints"
    registry.mkdir(parents=True)
    _write_metrics(root / "main_v3_strong_reliability_btapa_tau1", [(10, 0.39)])
    _write_metrics(root / "main_v3_strong_reliability_btapa_tau1_seed2", [(10, 0.44)])
    tau1 = registry / "main_v3_strong_reliability_btapa_tau1_primary_acc_0.3900.pth"
    seed2 = registry / "main_v3_strong_reliability_btapa_tau1_seed2_primary_acc_0.4400.pth"
    tau1.touch()
    seed2.touch()
    _write_sidecar(tau1, "main_v3_strong_reliability_btapa_tau1", 10, 0.39)
    _write_sidecar(seed2, "main_v3_strong_reliability_btapa_tau1_seed2", 10, 0.44)

    resolved = resolve_checkpoint(root, "main_v3_strong_reliability_btapa_tau1", "best_val_top1")

    assert resolved.path == tau1.resolve()
    assert resolved.epoch == 10
    assert "seed2" not in resolved.path.name


def test_resolve_checkpoint_uses_metrics_epoch_before_higher_filename_acc(tmp_path: Path):
    root = tmp_path / "scene31"
    run = "main_v3_strong_reliability_proto"
    ckpt_dir = root / run / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    _write_metrics(root / run, [(1, 0.30), (5, 0.50), (10, 0.45)])
    best = ckpt_dir / "best_top1.pth"
    latest = ckpt_dir / "last.pth"
    best.touch()
    latest.touch()
    _write_sidecar(best, run, 5, 0.50)
    _write_sidecar(latest, run, 10, 0.45)

    resolved = resolve_checkpoint(root, run, "best_val_top1")

    assert resolved.path == best.resolve()
    assert resolved.epoch == 5


def _write_metrics(run_dir: Path, rows: list[tuple[int, float]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_acc"])
        writer.writeheader()
        for epoch, val_acc in rows:
            writer.writerow({"epoch": epoch, "val_acc": val_acc})


def _write_sidecar(path: Path, run: str, epoch: int, metric: float) -> None:
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "config_slug": run,
                "run_dir": str(path.parents[1] / run),
                "selected_epoch": epoch,
                "metric_value": metric,
                "checkpoint_source": "top1-checkpoint",
                "task_metrics": {"val_acc": metric},
            }
        ),
        encoding="utf-8",
    )
