import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    path = ROOT / "scripts/launch_mmw_h2r_simplification_screen.py"
    spec = importlib.util.spec_from_file_location("h2r_simplification_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load()


def _load_watcher():
    path = ROOT / "scripts/run_mmw_h2r_simplification_evaluation_after_training.py"
    spec = importlib.util.spec_from_file_location("h2r_simplification_watcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h2r_simplification_matrix_has_fixed_gpu_identity_and_loss_profiles(tmp_path: Path) -> None:
    source = yaml.safe_load(LAUNCHER.DEFAULT_SOURCE_CONFIG.read_text(encoding="utf-8"))
    for gpu, candidate in enumerate(LAUNCHER.CANDIDATES):
        name, profile, supervision, epochs, loss_profile = candidate
        config = LAUNCHER.build_candidate_config(
            source,
            candidate=candidate,
            output_root=tmp_path,
            panel_path=tmp_path / "panel.json",
            panel_checksum="a" * 64,
            source_checkpoint=LAUNCHER.DEFAULT_SOURCE_CHECKPOINT,
            source_sha256=LAUNCHER.DEFAULT_SOURCE_SHA256,
            batch_size=64,
        )
        screen = config["mmw_h2r_simplification_screen"]
        dynamic = config["loss"]["u_mask_beam_jepa"]["dynamic_router"]
        assert gpu == LAUNCHER.CANDIDATES.index(candidate)
        assert screen["candidate"] == name
        assert screen["evidence_profile"] == profile
        assert screen["supervision"] == supervision
        assert screen["calibration_epochs"] == epochs
        assert config["training"]["epochs"] == epochs
        assert config["model"]["primary"]["router_variant_config"]["evidence_profile"] == profile
        assert dynamic["fused_decision_objective"] == "joint_hard_ce"
        assert dynamic["fused_utility_weight"] == 1.0
        if loss_profile == "full":
            assert dynamic["quality_regression_weight"] == 0.2
            assert dynamic["frame_rank_weight"] == 0.2
            assert dynamic["paired_joint"]["monotonic_weight"] == 0.2
        else:
            assert dynamic["quality_regression_weight"] == 0.0
            assert dynamic["frame_rank_weight"] == 0.0
            assert dynamic["residual_anchor_weight"] == 0.0
            expected_monotonic = 0.2 if loss_profile == "mono" else 0.0
            assert dynamic["paired_joint"]["monotonic_weight"] == expected_monotonic


def test_h2r_evaluation_watcher_preserves_candidate_gpu_identity(tmp_path: Path) -> None:
    watcher = _load_watcher()
    training = {
        "jobs": [
            {
                "candidate": candidate[0],
                "gpu": gpu,
                "config_path": str(tmp_path / f"{candidate[0]}.yaml"),
                "config_sha256": "a" * 64,
                "run_dir": str(tmp_path / candidate[0]),
            }
            for gpu, candidate in enumerate(LAUNCHER.CANDIDATES)
        ]
    }
    manifest = watcher._prepare_manifest(tmp_path / "eval/manifest.json", {}, "b" * 64, training)
    assert [(job["candidate"], job["gpu"]) for job in manifest["jobs"]] == [
        (candidate[0], gpu) for gpu, candidate in enumerate(LAUNCHER.CANDIDATES)
    ]
    assert all(job["checkpoint_sha256"] is None for job in manifest["jobs"])
