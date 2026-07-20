import importlib.util
from pathlib import Path
import sys

from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts/launch_mmw_dynamic_router_decision_screen.py"
    spec = importlib.util.spec_from_file_location("dynamic_router_decision_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decision_screen_has_frozen_two_by_four_matrix() -> None:
    module = _module()
    assert len(module.CANDIDATES) == 8
    assert {item[1] for item in module.CANDIDATES} == {"patr", "h2r"}
    for variant in ("patr", "h2r"):
        assert {item[2] for item in module.CANDIDATES if item[1] == variant} == {
            "expected_utility",
            "joint_hard_ce",
            "power_soft_ce",
            "power_top1_margin",
        }


def test_decision_candidate_changes_only_declared_router_objective(tmp_path: Path) -> None:
    module = _module()
    source = {
        "experiment": {"name": "CurrentControl", "seed": 1},
        "data": {"dataset": {}, "dataloader": {}},
        "temporal_missing": {},
        "model": {"primary": {"fusion_type": "supervised_router", "head_type": "prototype"}},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "use_beam_prototype_alignment": True,
                "router_oracle_weight": 0.1,
                "prototype_topology": {"id": "cyclic_index_v1", "circular": True},
                "prototype_target_circular": True,
            }
        },
        "training": {},
        "scheduler": {},
        "output": {},
    }
    checkpoint = tmp_path / "last.pth"
    checkpoint.touch()
    config = module.build_candidate_config(
        source,
        name="H2R-PowerMargin",
        variant="h2r",
        objective="power_top1_margin",
        output_root=tmp_path / "out",
        panel_path=tmp_path / "panel.json",
        panel_checksum="a" * 64,
        source_checkpoint=checkpoint,
        source_sha256="b" * 64,
        batch_size=64,
        epochs=40,
    )
    dynamic = u_mask_beam_jepa_config(config)["dynamic_router"]
    assert dynamic["fused_decision_objective"] == "power_top1_margin"
    assert dynamic["supervision"] == "beam_power"
    assert dynamic["fused_utility_weight"] == 1.0
    assert config["mmw_dynamic_router_decision_screen"]["claim_eligible"] is False
