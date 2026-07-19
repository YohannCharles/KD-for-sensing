import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/update_twc_experiment_status.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_status_ledger_discovers_manifests_and_reports_checkpoint(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    run_dir = tmp_path / "runs/T2/seed1"
    checkpoint = run_dir / "checkpoints/last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "training_manifest_v1.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "test",
                "plan_sha256": "abc",
                "jobs": [
                    {
                        "variant": "T2",
                        "seed": 1,
                        "status": "done",
                        "run_dir": str(run_dir),
                        "evaluation_status": "planned",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: [("0", "1", "2", "3")])

    manifests = module.discover_manifests(tmp_path)
    ledger = module.render_ledger(manifests)

    assert "TWC 实验运行台账" in ledger
    assert "| T2 | MMW-15域 | 1 | done" in ledger
    assert "| 存在 | planned | 不可：固定评估未完成 |" in ledger


def test_status_ledger_groups_posthoc_shards_instead_of_repeating_rows(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: [])
    jobs = [
        {
            "kind": "corruption",
            "method": "T2",
            "seed": seed,
            "corruption": corruption,
            "severity": 1,
            "status": "planned",
            "log_path": f"/tmp/{seed}_{corruption}.log",
        }
        for seed in (1, 2)
        for corruption in ("gps_noise", "radar_noise")
    ]

    ledger = module.render_ledger([(Path("/tmp/posthoc_manifest.json"), {"protocol": "twc_posthoc", "jobs": jobs})])

    assert ledger.count("T2 / corruption (4 shards)") == 1
    assert "planned=4" in ledger
    assert "T2 / gps_noise:S1" not in ledger
