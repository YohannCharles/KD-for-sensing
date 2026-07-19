import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/launch_twc_posthoc_evidence.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reliability_stress_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"manifest_sha256": "protocol-sha"}), encoding="utf-8")
    corruption_manifest = tmp_path / "corruption.json"
    corruption_manifest.write_text(json.dumps({"protocol_id": "test"}), encoding="utf-8")
    monkeypatch.setattr(module, "_prepare_corruption_manifest", lambda _: corruption_manifest)

    default_path = module.prepare_plan(tmp_path / "default", protocol)
    default = json.loads(default_path.read_text(encoding="utf-8"))
    assert default["reliability_stress_enabled"] is False
    assert default["request"]["reliability_stress_policy"] == "explicit_flag_only"
    assert len(default["jobs"]) == len(module.METHODS)
    assert {job["kind"] for job in default["jobs"]} == {"complexity"}

    opted_in_path = module.prepare_plan(tmp_path / "opted-in", protocol, reliability_stress=True)
    opted_in = json.loads(opted_in_path.read_text(encoding="utf-8"))
    expected = len(module.METHODS) + len(module.METHODS) * len(module.SEEDS) * len(module.CORRUPTION_GRID)
    assert opted_in["reliability_stress_enabled"] is True
    assert len(opted_in["jobs"]) == expected
    assert any(job["kind"] == "corruption" for job in opted_in["jobs"])
